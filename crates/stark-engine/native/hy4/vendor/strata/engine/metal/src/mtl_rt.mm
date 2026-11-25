// metal/src/mtl_rt.mm - see mtl_rt.h.  Built with ARC (-fobjc-arc).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "stark_hy4_vendor.h"
#include "mtl_rt.h"
#include "stark_hy4_msl.h"

#include <sys/mman.h>
#include <sys/sysctl.h>
#include <time.h>

#include <atomic>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace mtl {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Hy4 Metal runtime failure");
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

struct Context::Impl {
    id<MTLDevice> dev;
    id<MTLCommandQueue> q;
    id<MTLCommandQueue> q2 = nil;   // run_graph_async's queue
    std::vector<id<MTLLibrary>> libs;
    std::vector<id<MTLComputePipelineState>> pipelines;
    id<MTLSharedEvent> ev = nil;   // run_graph_seg's handshake timeline
    uint64_t ev_v = 0;
    id<MTLCounterSampleBuffer> seg_samples = nil;
    id<MTLCounterSampleBuffer> segment_timestamp_buffer(size_t count) {
        if (seg_samples.sampleCount >= count) return seg_samples;
        if (![dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) return nil;
        for (id<MTLCounterSet> set in dev.counterSets) {
            if (![set.name isEqualToString:MTLCommonCounterSetTimestamp]) continue;
            MTLCounterSampleBufferDescriptor* desc = [MTLCounterSampleBufferDescriptor new];
            desc.counterSet = set;
            desc.storageMode = MTLStorageModeShared;
            desc.sampleCount = count;
            seg_samples = [dev newCounterSampleBufferWithDescriptor:desc error:nullptr];
            break;
        }
        return seg_samples;
    }
    // #6078, the GPU keep-alive.  After about a second with no command buffer, macOS lets go of the process's GPU allocations, and the
    // next command buffer waits before it starts (M5 Ultra: 0.9-1.1 s between commit and GPUStartTime; M5 Max, Q2_0, a 21-token
    // follow-up: prompt 155 ms after a gap of 1 s or less, 440 ms after 2-4 s): an agent's turn after a tool call paid it in its
    // conversation park and its prompt read.  While no command buffer was committed for keepalive_ms, and for at most keepalive_win_s
    // after the last one, a one-thread kernel that names every buffer of the context (useResource) goes to the main queue every
    // keepalive_ms.  A residency set alone (requestResidency) did not hold the allocations, nor did a keep-alive that names no buffer
    // (#6078's own trials).  The cost: the allocations stay wired during that window, as during work; so it ends keepalive_win_s
    // after the last command buffer, and it pauses while macOS reports memory pressure (kern.memorystatus_vm_pressure_level above
    // normal), so wired memory is given back when the Mac needs it.  STRATA_METAL_KEEPALIVE_MS (default 250, 0 = off) and
    // STRATA_METAL_KEEPALIVE_WINDOW_S (default 120) set it.
    std::atomic<double> last_commit{0.0};
    std::atomic<bool> ka_stop{false};
    std::thread ka;
    std::mutex all_mu;
    NSMutableArray<id<MTLBuffer>>* all = [NSMutableArray new];
    static double up_s() { return (double) clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9; }
    void touch() { last_commit.store(up_s(), std::memory_order_relaxed); }
    void track(id<MTLBuffer> b) {
        if (b == nil) return;
        std::lock_guard<std::mutex> lk(all_mu);
        [all addObject:b];
    }
    void untrack(id<MTLBuffer> b) {
        if (b == nil) return;
        std::lock_guard<std::mutex> lk(all_mu);
        [all removeObjectIdenticalTo:b];
    }
    static bool under_pressure() {
        int lvl = 0;
        size_t len = sizeof lvl;
        return sysctlbyname("kern.memorystatus_vm_pressure_level", &lvl, &len, nullptr, 0) == 0 && lvl > 1;
    }
    void start_keepalive() {
        const char* e = stark::hy4::vendor_option("STRATA_METAL_KEEPALIVE_MS");
        const double ms = e && *e ? std::atof(e) : 250.0;
        if (!(ms > 0)) return;
        const char* w = stark::hy4::vendor_option("STRATA_METAL_KEEPALIVE_WINDOW_S");
        const double win = w && *w ? std::atof(w) : 120.0;
        if (!(win > 0)) return;
        NSError* err = nil;
        id<MTLLibrary> l = [dev newLibraryWithSource:@"kernel void strata_keepalive(device uint* b [[buffer(0)]]) { b[0] += 1u; }"
                                             options:nil error:&err];
        id<MTLFunction> f = l ? [l newFunctionWithName:@"strata_keepalive"] : nil;
        id<MTLComputePipelineState> p = f ? [dev newComputePipelineStateWithFunction:f error:&err] : nil;
        id<MTLBuffer> one = [dev newBufferWithLength:256 options:MTLResourceStorageModeShared];
        if (p == nil || one == nil) {
            std::fprintf(stderr, "mtl: the GPU keep-alive kernel did not build; keep-alive off\n");
            return;
        }
        id<MTLCommandQueue> qq = q;
        ka = std::thread([this, ms, win, p, one, qq] {
            while (!ka_stop.load()) {
                std::this_thread::sleep_for(std::chrono::microseconds((long long) (ms * 1000)));
                if (ka_stop.load()) break;
                const double idle = up_s() - last_commit.load(std::memory_order_relaxed);
                if (idle * 1000.0 < ms || idle > win || under_pressure()) continue;
                @autoreleasepool {
                    // a retained command buffer: a buffer freed while it is in flight stays alive until it completes
                    id<MTLCommandBuffer> cb = [qq commandBuffer];
                    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
                    [en setComputePipelineState:p];
                    [en setBuffer:one offset:0 atIndex:0];
                    {
                        std::lock_guard<std::mutex> lk(all_mu);
                        for (id<MTLBuffer> x in all) [en useResource:x usage:MTLResourceUsageRead];
                    }
                    [en dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                    [en endEncoding];
                    [cb commit];   // (not a real command buffer: last_commit is not moved)
                    [cb waitUntilCompleted];
                }
            }
        });
        std::fprintf(stderr, "mtl: GPU keep-alive every %.0f ms for %.0f s after the last command buffer (#6078; "
                     "STRATA_METAL_KEEPALIVE_MS=0 turns it off)\n", ms, win);
    }
    MTLSharedEventListener* ev_listener = nil;   // for STRATA_METAL_SEG_WAIT=event (created on first use)
};

static inline id<MTLBuffer> B(const Buffer& b) { return (__bridge id<MTLBuffer>)b.handle; }
// the MTLBuffer and offset that bind byte `off` of b (a chunked buffer: the chunk the offset selects)
static inline id<MTLBuffer> BO(const Buffer& b, size_t off, size_t& o) {
    if (!b.chunks) { o = off + b.bias; return B(b); }   // (bias: a view onto part of a bigger allocation; 0 otherwise)
    if (b.starts) {
        size_t k = b.starts->size() - 1;
        while (k > 0 && (*b.starts)[k] > off) --k;
        o = off - (*b.starts)[k];
        return (__bridge id<MTLBuffer>)(*b.chunks)[k];
    }
    const size_t k = std::min(off / b.stride, b.chunks->size() - 1);
    o = off - k * b.stride;
    return (__bridge id<MTLBuffer>)(*b.chunks)[k];
}
static inline id<MTLComputePipelineState> P(const Pipeline* p) { return (__bridge id<MTLComputePipelineState>)p->handle; }
Context::Context() {
    auto impl = std::make_unique<Impl>();
    impl->dev = MTLCreateSystemDefaultDevice();
    if (impl->dev == nil) throw std::runtime_error("No Metal device is available");
    impl->q = [impl->dev newCommandQueue];
    if (impl->q == nil) throw std::runtime_error("Could not create the Hy4 Metal command queue");
    impl->touch();
    impl->start_keepalive();
    impl_ = impl.release();
}

// ~Context is defined after the GPU filler, which it has to stop first (see set_seg_fill).

std::string Context::device_name() const { return [[impl_->dev name] UTF8String]; }
bool Context::has_family_apple(int n) const {
    return [impl_->dev supportsFamily:(MTLGPUFamily)(MTLGPUFamilyApple1 + (n - 1))];
}
bool Context::supports_metal4() const {
    if (@available(macOS 26.0, *)) return [impl_->dev supportsFamily:MTLGPUFamilyMetal4];
    return false;
}
uint64_t Context::recommended_working_set() const { return [impl_->dev recommendedMaxWorkingSetSize]; }
uint64_t Context::max_buffer_length() const { return [impl_->dev maxBufferLength]; }

// See mtl_rt.h: the switches a deterministic engine must not inherit.
const char* deterministic_env_violation() {
    if (const char* m = stark::hy4::vendor_option("STRATA_METAL_MATH"))
        if (std::string(m) == "fast") return "STRATA_METAL_MATH=fast";
    if (stark::hy4::vendor_option("STRATA_METAL_F16")) return "STRATA_METAL_F16";
    if (stark::hy4::vendor_option("STRATA_METAL_SKIP")) return "STRATA_METAL_SKIP";
    if (stark::hy4::vendor_option("STRATA_METAL_GEMV_NR")) return "STRATA_METAL_GEMV_NR";
    return nullptr;
}

void Context::lock_safe_math() {
    if (const char* bad = deterministic_env_violation()) {
        throw std::runtime_error("Hy4 Metal runtime failure");
    }
    safe_math_locked_ = true;
}

int Context::compile(const std::vector<std::string>& files, const std::string& prelude, bool fast_math) {
    std::string why;
    const int id = compile_try(files, prelude, fast_math, &why);
    if (id < 0) throw std::runtime_error("Hy4 Metal shader compilation failed: " + why);
    return id;
}

// The same, and a failure is returned (-1, the compiler's message in *why) instead of ending the process: for a library that has a fallback.
int Context::compile_try(const std::vector<std::string>& files, const std::string& prelude, bool fast_math, std::string* why) {
    std::string src = prelude + "\n#include <metal_stdlib>\nusing namespace metal;\n";
    if (prelude.find("NO_GGML_COMMON") == std::string::npos) {
        src += "#define GGML_COMMON_DECL_METAL\n#define GGML_COMMON_IMPL_METAL\n";
        src += stark::hy4::msl("strata-quant-common.h");
        src += '\n';
    }
    for (const auto& f : files) {
        const auto name_at = f.find_last_of('/');
        const auto name = f.substr(name_at == std::string::npos ? 0 : name_at + 1);
        const auto code = stark::hy4::msl(name);
        if (code.empty()) {
            if (why) *why = "Missing embedded Metal source " + name;
            return -1;
        }
        src += "#line 1 \"" + name + "\"\n";
        src += code;
        src += '\n';
    }
    MTLCompileOptions* o = [MTLCompileOptions new];
    // Safe math by default: no reassociation, so every token of a window is bitwise what a one-token call gives
    // (the kernels write their contractions as explicit fma).
    fast_math = false;
    if (const char* m = stark::hy4::vendor_option("STRATA_METAL_MATH")) fast_math = std::string(m) == "fast";
    if (safe_math_locked_) {
        // Re-checked at every compile, not only at lock(): a caller that locks safe math must never get a fast-math
        // library, and the switch is an error rather than a downgrade.
        if (const char* bad = deterministic_env_violation()) {
            throw std::runtime_error("Hy4 Metal runtime failure");
        }
        fast_math = false;
    }
    if (@available(macOS 15.0, *)) {
        o.mathMode = fast_math ? MTLMathModeFast : MTLMathModeSafe;
    } else {
        o.fastMathEnabled = fast_math;
    }
    NSError* err = nil;
    id<MTLLibrary> lib = [impl_->dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:o error:&err];
    if (!lib) {
        if (why) *why = err ? [[err localizedDescription] UTF8String] : "unknown";
        return -1;
    }
    impl_->libs.push_back(lib);
    return (int)impl_->libs.size() - 1;
}

Pipeline Context::pipeline(int lib, const std::string& fn, const Constants& c) {
    NSError* err = nil;
    id<MTLFunction> f;
    NSString* name = [NSString stringWithUTF8String:fn.c_str()];
    if (c.vals.empty()) {
        f = [impl_->libs[lib] newFunctionWithName:name];
        // a function with (optional) function constants needs a specialization even when none is set
        if (f && f.functionConstantsDictionary.count > 0)
            f = [impl_->libs[lib] newFunctionWithName:name constantValues:[MTLFunctionConstantValues new] error:&err];
    } else {
        MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
        for (const auto& v : c.vals) {
            int32_t iv = (int32_t)v.v;
            bool bv = v.v != 0;
            uint32_t uv = v.v;
            switch (v.type) {
                case 0: [cv setConstantValue:&iv type:MTLDataTypeInt atIndex:v.index]; break;
                case 1: [cv setConstantValue:&bv type:MTLDataTypeBool atIndex:v.index]; break;
                default: [cv setConstantValue:&uv type:MTLDataTypeUInt atIndex:v.index]; break;
            }
        }
        f = [impl_->libs[lib] newFunctionWithName:name constantValues:cv error:&err];
    }
    if (!f) throw std::runtime_error("Missing Hy4 Metal function " + fn);
    MTLComputePipelineDescriptor* d = [MTLComputePipelineDescriptor new];
    d.computeFunction = f;
    d.supportIndirectCommandBuffers = YES;
    id<MTLComputePipelineState> ps = [impl_->dev newComputePipelineStateWithDescriptor:d
                                                                               options:MTLPipelineOptionNone
                                                                            reflection:nil
                                                                                 error:&err];
    if (!ps) throw std::runtime_error("Could not create Hy4 Metal pipeline " + fn);
    impl_->pipelines.push_back(ps);
    Pipeline p;
    p.handle = (__bridge void*)ps;
    p.max_threads = (int)ps.maxTotalThreadsPerThreadgroup;
    if (const char* cap = stark::hy4::vendor_option("STRATA_METAL_MAX_THREADS_CAP")) {   // test hook, see require_fit
        const int lim = std::atoi(cap);
        if (lim > 0 && lim < p.max_threads) p.max_threads = lim;
    }
    p.simd_width = (int)ps.threadExecutionWidth;
    p.static_tg_mem = (int)ps.staticThreadgroupMemoryLength;
    p.name = fn;
    if (!c.vals.empty()) {
        p.name += "{";
        for (size_t i = 0; i < c.vals.size(); ++i)
            p.name += (i ? "," : "") + std::to_string(c.vals[i].index) + "=" + std::to_string((int)c.vals[i].v);
        p.name += "}";
    }
    return p;
}

void require_fit(const Pipeline& p, uint32_t threads, const char* where) {
    if (threads == 0 || (int64_t)threads <= (int64_t)p.max_threads) return;
    throw std::runtime_error("Unsupported Hy4 Metal threadgroup size for " + p.name + " in " + where);
}

Buffer Context::alloc(size_t bytes) {
    // Metal returns the new buffer with one more reference parked in the calling thread's autorelease pool.  The engine's host
    // threads have no pool (only Context::run opens one), so without the pool below that reference is never given back: every
    // buffer made here would live for the rest of the process, free() or not.
    Buffer r;
    @autoreleasepool {
        id<MTLBuffer> b = [impl_->dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!b) {
            throw std::runtime_error("Hy4 Metal runtime failure");
        }
        std::memset([b contents], 0, bytes);
        impl_->track(b);
        r.handle = (__bridge void*)b;
        r.ptr = [b contents];
    }
    allocated_ += bytes;
    r.bytes = bytes;
    return r;
}

Buffer Context::wrap(void* p, size_t bytes, size_t max_span) {
    const size_t pg = 16384, maxlen = [impl_->dev maxBufferLength] / pg * pg;
    if (bytes <= maxlen || stark::hy4::vendor_option("STRATA_METAL_WRAP_SPLIT") && std::string(stark::hy4::vendor_option("STRATA_METAL_WRAP_SPLIT")) == "0")
        return wrap(p, bytes);
    const size_t span = (max_span + pg - 1) / pg * pg;
    if (span >= maxlen / 2) {
        throw std::runtime_error("Hy4 Metal runtime failure");
    }
    Buffer r;
    r.chunks = std::make_shared<std::vector<void*>>();
    r.stride = maxlen - span;
    for (size_t base = 0; base < bytes; base += r.stride) {
        const size_t len = std::min(maxlen, (bytes - base + pg - 1) / pg * pg);
        @autoreleasepool {   // as in alloc()
            id<MTLBuffer> b = [impl_->dev newBufferWithBytesNoCopy:(uint8_t*) p + base length:len
                                                            options:MTLResourceStorageModeShared deallocator:nil];
            if (!b) {
                throw std::runtime_error("Hy4 Metal runtime failure");
            }
            impl_->track(b);
            r.chunks->push_back((__bridge void*)b);
        }
        if (base + len >= bytes) break;
    }
    allocated_ += bytes;
    r.handle = (*r.chunks)[0];
    r.ptr = p;
    r.bytes = bytes;
    std::fprintf(stderr, "mtl: wrapped %.2f GB as %zu overlapping buffers (maxBufferLength %.2f GB, overlap %.2f GB)\n",
                 bytes / 1e9, r.chunks->size(), maxlen / 1e9, span / 1e9);
    return r;
}

Buffer Context::wrap_ranges(void* p, const std::vector<std::pair<size_t, size_t>>& ranges) {
    Buffer r;
    r.chunks = std::make_shared<std::vector<void*>>();
    r.starts = std::make_shared<std::vector<size_t>>();
    size_t total = 0;
    for (const auto& rg : ranges) {
        @autoreleasepool {   // as in alloc()
            id<MTLBuffer> b = [impl_->dev newBufferWithBytesNoCopy:(uint8_t*) p + rg.first length:rg.second - rg.first
                                                            options:MTLResourceStorageModeShared deallocator:nil];
            if (!b) {
                throw std::runtime_error("Hy4 Metal runtime failure");
            }
            impl_->track(b);   // (the #6078 keep-alive names it like every other buffer of the context)
            r.chunks->push_back((__bridge void*)b);
        }
        r.starts->push_back(rg.first);
        total += rg.second - rg.first;
    }
    allocated_ += total;
    r.handle = (*r.chunks)[0];
    r.ptr = p;
    r.bytes = ranges.back().second;
    std::fprintf(stderr, "mtl: wrapped %.2f GB as %zu separate ranges (of %.2f GB)\n", total / 1e9, ranges.size(), r.bytes / 1e9);
    return r;
}

Buffer Context::wrap(void* p, size_t bytes) {
    Buffer r;
    @autoreleasepool {   // (see alloc: the engine's host threads have no pool of their own)
        id<MTLBuffer> b = [impl_->dev newBufferWithBytesNoCopy:p length:bytes options:MTLResourceStorageModeShared deallocator:nil];
        if (!b) {
            throw std::runtime_error("Hy4 Metal runtime failure");
        }
        impl_->track(b);
        r.handle = (__bridge void*)b;
    }
    allocated_ += bytes;
    r.ptr = p;
    r.bytes = bytes;
    return r;
}

Buffer Context::map(size_t bytes) {
    void* ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (ptr == MAP_FAILED) throw std::runtime_error("Could not map the Hy4 expert arena");
    @autoreleasepool {
        id<MTLBuffer> buffer = [impl_->dev newBufferWithBytesNoCopy:ptr
            length:bytes options:MTLResourceStorageModeShared
            deallocator:^(void* address, NSUInteger length) { munmap(address, length); }];
        if (buffer == nil) {
            munmap(ptr, bytes);
            throw std::runtime_error("Hy4 expert arena Metal mapping failed");
        }
        impl_->track(buffer);
        allocated_ += bytes;
        Buffer out;
        out.handle = (__bridge void*)buffer;
        out.ptr = ptr;
        out.bytes = bytes;
        return out;
    }
}

void Context::stop_keepalive() {
    impl_->ka_stop.store(true);
    if (impl_->ka.joinable()) impl_->ka.join();
}

void Context::free(Buffer& b) {
    if (b.chunks) {
        for (void* h : *b.chunks) { id<MTLBuffer> x = (__bridge id<MTLBuffer>)h; impl_->untrack(x); x = nil; }
        b = Buffer{};
        return;
    }
    if (b.handle) {
        id<MTLBuffer> x = (__bridge id<MTLBuffer>)b.handle;
        if (!b.bias) impl_->untrack(x);   // (a view onto part of an allocation is not the allocation's own entry)
        x = nil;
    }
    b = Buffer{};
}

static void encode(id<MTLComputeCommandEncoder> enc, const Dispatch& d) {
    require_fit(d, "dispatch");
    [enc setComputePipelineState:P(d.p)];
    for (size_t i = 0; i < d.args.size(); ++i) {
        const Arg& a = d.args[i];
        if (a.buf) {
            size_t o;
            id<MTLBuffer> mb = BO(*a.buf, a.off, o);
            [enc setBuffer:mb offset:o atIndex:i];
        }
        else
            [enc setBytes:a.bytes.data() length:a.bytes.size() atIndex:i];
    }
    if (d.tg_mem) [enc setThreadgroupMemoryLength:(d.tg_mem + 15) / 16 * 16 atIndex:0];
    if (d.indirect)
        [enc dispatchThreadgroupsWithIndirectBuffer:B(*d.indirect)
                               indirectBufferOffset:d.ind_off
                              threadsPerThreadgroup:MTLSizeMake(d.tg[0], d.tg[1], d.tg[2])];
    else
        [enc dispatchThreadgroups:MTLSizeMake(d.grid[0], d.grid[1], d.grid[2])
            threadsPerThreadgroup:MTLSizeMake(d.tg[0], d.tg[1], d.tg[2])];
}

double Context::run(const std::vector<Dispatch>& ds, int reps, bool concurrent, bool barrier_between) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [impl_->q commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc =
            [cb computeCommandEncoderWithDispatchType:concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
        for (int r = 0; r < reps; ++r)
            for (const auto& d : ds) {
                encode(enc, d);
                if (barrier_between) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
        [enc endEncoding];
        [cb commit];
        impl_->touch();
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("Hy4 Metal runtime failure");
        }
        return cb.GPUEndTime - cb.GPUStartTime;
    }
}

double Context::run_graph(const std::vector<Dispatch>& ds, size_t count) {
    seg_tm_.n_dis_plain += ds.size();
    // The wait spins on the buffer's status for up to 20 ms before it blocks: waitUntilCompleted's thread wake costs ~0.5 ms
    // a window (M5 Max, Q2_0 decode: +5% tok/s, same tokens).  STRATA_METAL_SPIN=0: the blocking wait.
    static const bool spin = !(stark::hy4::vendor_option("STRATA_METAL_SPIN") && std::string(stark::hy4::vendor_option("STRATA_METAL_SPIN")) == "0");
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [impl_->q commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        const size_t nd = std::min(count, ds.size());
        for (size_t i = 0; i < nd; ++i) {
            if (i > 0 && ds[i].barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode(enc, ds[i]);
        }
        [enc endEncoding];
        [cb commit];
        impl_->touch();
        if (spin) {
            const auto t0 = std::chrono::steady_clock::now();
            while (cb.status < MTLCommandBufferStatusCompleted &&
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 20e-3) {}
        }
        if (cb.status < MTLCommandBufferStatusCompleted) [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("Hy4 Metal runtime failure");
        }
        return cb.GPUEndTime - cb.GPUStartTime;
    }
}

void* Context::run_graph_async(const std::vector<Dispatch>& ds) {
    seg_tm_.n_dis_plain += ds.size();
    @autoreleasepool {
        if (impl_->q2 == nil) impl_->q2 = [impl_->dev newCommandQueue];
        id<MTLCommandBuffer> cb = [impl_->q2 commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        for (size_t i = 0; i < ds.size(); ++i) {
            if (i > 0 && ds[i].barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode(enc, ds[i]);
        }
        [enc endEncoding];
        [cb commit];
        impl_->touch();
        return (__bridge_retained void*) cb;
    }
}

double Context::wait_async(void* token) {
    if (token == nullptr) return 0.0;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge_transfer id<MTLCommandBuffer>) token;
        if (cb.status < MTLCommandBufferStatusCompleted) [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("Hy4 Metal runtime failure");
        }
        return cb.GPUEndTime - cb.GPUStartTime;
    }
}

namespace {
// HOW THE HOST WAITS FOR A CUT.  `spin` is the default and the event path is opt-in.  Neither can alter a value: the
// choice only decides when this thread notices a signal the GPU has already sent.
//   STRATA_METAL_SEG_WAIT=event   short spin, then block on the shared event
//   STRATA_METAL_SEG_SPIN_US=N    the spin budget before blocking (default 50)
// Note STRATA_METAL_SPIN belongs to `run_graph` and does not affect `run_graph_seg`.
bool seg_wait_event() {
    static const bool on = [] {
        const char* e = stark::hy4::vendor_option("STRATA_METAL_SEG_WAIT");
        return e != nullptr && std::string(e) == "event";
    }();
    return on;
}
double spin_budget_s() {
    static const double b = [] {
        const char* e = stark::hy4::vendor_option("STRATA_METAL_SEG_SPIN_US");
        const double us = e != nullptr ? std::atof(e) : 50.0;
        return (us > 0 ? us : 0.0) * 1e-6;
    }();
    return b;
}
}  // namespace

double Context::run_graph_seg(const std::vector<Dispatch>& ds, const std::vector<int>& cuts, void (*at_cut)(void*, int),
                              void* user) {
    return run_graph_seg(ds, cuts, {}, at_cut, user);
}

// ---- the GPU filler (see mtl_rt.h: set_seg_fill) ---------------------------------------------------------------
//
// One per Context, created on the first set_seg_fill(non-zero) and joined in ~Context, so it never outlives the
// queue and objects it uses.  Three states for the thread:
//   disarmed (mode 0)  blocked on the condition variable - no CPU, no GPU.  An engine that never calls
//                      set_seg_fill has no thread at all.
//   cut (1)            while the host is inside a cut, a back-to-back stream of short filler buffers; otherwise it
//                      yield-spins, because it has to start within microseconds of the host entering a cut and a
//                      condition-variable wake costs more than that.  So an armed `cut` filler holds one CPU core;
//                      the caller arms it only for decode (hy4-flash: main.cpp, around the generation loop).
//   always (2)         filler buffers while any segmented run happened in the last 50 ms, else 1 ms sleeps.
// A filler buffer is ~250 us of GPU, so at disarm or exit at most one is still in flight; the thread finishes it
// before it parks or returns, and ~Context waits for the thread.
struct Context::Filler {
    std::mutex mu;
    std::condition_variable cv;
    int mode = 0;                         // guarded by mu; mirrored into `armed` for the run_graph_seg fast path
    bool stop = false;                    // guarded by mu
    std::atomic<int> armed{0};
    // its own cache line: the worker writes `mu`'s line on every spin iteration, and the host's increment around
    // each at_cut should not have to pull that line back
    alignas(64) std::atomic<int> want{0};   // cut mode: > 0 while the host is inside at_cut
    std::atomic<int64_t> last_ns{0};      // always mode: the last segmented run's activity
    std::atomic<uint64_t> n{0}, gpu_ns{0};
    std::thread th;
    bool failed = false;                  // guarded by mu: a filler buffer failed, so it stays off
    double one_us = 0;
    uint32_t iters = 4000;
};

namespace {
int64_t fill_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
// ALU only: an LCG per thread, and a store that happens only if the state lands on one value in 2^32 after n
// steps, into a word nothing reads.  No loads, so it takes no bandwidth from the real work and cannot change a
// value anywhere.
const char* kFillSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void seg_fill(device atomic_uint* sink [[buffer(0)]], constant uint& n [[buffer(1)]],
                     uint t [[thread_position_in_grid]]) {
    uint x = t * 2654435761u + 1u;
    for (uint i = 0; i < n; ++i) x = x * 1664525u + 1013904223u;
    if (x == 0xdeadbeefu) atomic_fetch_add_explicit(sink, 1u, memory_order_relaxed);
}
)MSL";
}  // namespace

std::string Context::set_seg_fill(int mode) {
    // The environment wins over the caller, so it can be switched without a rebuild: `off` disables it
    // outright; `cut` / `always` pick the mode, while the caller still decides WHEN it is armed.
    if (const char* e = stark::hy4::vendor_option("STRATA_METAL_SEG_FILL")) {
        std::string m(e);
        for (char& c : m) c = (char) std::tolower((unsigned char) c);
        // an unrecognised value is an error, not "leave it on": `OFF` or `false` silently keeping the filler on
        // would put it on both sides of a comparison that was meant to turn it off
        if (m == "off" || m == "0" || m == "false" || m == "no" || m == "none") mode = 0;
        else if (m == "cut" || m == "1") mode = mode == 0 ? 0 : 1;
        else if (m == "always" || m == "2") mode = mode == 0 ? 0 : 2;
        else return "STRATA_METAL_SEG_FILL=" + std::string(e) + " is not off|cut|always";
    }
    if (mode < 0 || mode > 2) mode = 0;
    if (mode != 0 && fill_ == nullptr) {
        auto* f = new Filler;
        // Bounded: a filler buffer cannot be interrupted once submitted, so its length is the bound on how long it
        // can sit in front of real work, and on how long a disarm or ~Context waits.  Hence the range check below:
        // a negative value would wrap into billions of steps.
        if (const char* it = stark::hy4::vendor_option("STRATA_METAL_SEG_FILL_ITERS")) {
            const long v = std::atol(it);
            if (v < 1 || v > 100000) {
                delete f;
                return "STRATA_METAL_SEG_FILL_ITERS=" + std::string(it) + " is outside 1..100000";
            }
            f->iters = (uint32_t) v;
        }
        NSError* err = nil;
        MTLCompileOptions* o = [MTLCompileOptions new];
        o.fastMathEnabled = NO;   // nothing here is a value anyone reads, but this runtime never compiles fast-math
        id<MTLLibrary> lib = [impl_->dev newLibraryWithSource:[NSString stringWithUTF8String:kFillSrc] options:o
                                                        error:&err];
        id<MTLComputePipelineState> ps =
            lib ? [impl_->dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"seg_fill"] error:&err]
                : nil;
        if (ps == nil) {
            delete f;
            return std::string("the GPU filler did not compile: ") +
                   (err ? [[err localizedDescription] UTF8String] : "?");
        }
        id<MTLCommandQueue> q = [impl_->dev newCommandQueue];
        id<MTLBuffer> sink = [impl_->dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
        if (q == nil || sink == nil) {
            delete f;
            return "the GPU filler could not create its queue or its sink buffer";
        }
        // < 0 = the buffer did not run.  Checked, because an unchecked failure turns the filler into a tight CPU
        // loop that counts buffers the GPU never executed.
        auto submit = [q, ps, sink](uint32_t iters) -> double {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                if (cb == nil) return -1.0;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                if (enc == nil) return -1.0;
                [enc setComputePipelineState:ps];
                [enc setBuffer:sink offset:0 atIndex:0];
                [enc setBytes:&iters length:4 atIndex:1];
                [enc dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if (cb.status != MTLCommandBufferStatusCompleted) return -1.0;
                return cb.GPUEndTime - cb.GPUStartTime;
            }
        };
        for (int k = 0; k < 3; ++k) {   // the last of three counts, after the first has warmed the pipeline
            const double g = submit(f->iters);
            if (!(g > 0)) {
                delete f;
                return "the GPU filler's test buffer did not run";
            }
            f->one_us = g * 1e6;
        }
        f->th = std::thread([f, submit] {
            for (;;) {
                {
                    std::unique_lock<std::mutex> lk(f->mu);
                    f->cv.wait(lk, [f] { return f->stop || f->mode != 0; });
                    if (f->stop) return;
                }
                const int m = f->armed.load(std::memory_order_acquire);
                bool on = false;
                if (m == 1) on = f->want.load(std::memory_order_acquire) > 0;
                else if (m == 2) on = fill_now_ns() - f->last_ns.load(std::memory_order_relaxed) < 50000000LL;
                if (!on) {
                    if (m == 1) std::this_thread::yield();
                    else std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                const double g = submit(f->iters);
                if (!(g > 0)) {
                    // a failed filler buffer disarms the filler for good rather than spinning on it; decode
                    // carries on without it (it is only a clock aid)
                    std::fprintf(stderr, "mtl: a GPU filler buffer failed; the filler is off for the rest of the run\n");
                    std::lock_guard<std::mutex> lk(f->mu);
                    f->mode = 0;
                    f->armed.store(0, std::memory_order_release);
                    f->failed = true;
                    continue;
                }
                f->n.fetch_add(1, std::memory_order_relaxed);
                f->gpu_ns.fetch_add((uint64_t) (g * 1e9), std::memory_order_relaxed);
            }
        });
        fill_ = f;
    }
    if (fill_ == nullptr) return {};
    {
        std::lock_guard<std::mutex> lk(fill_->mu);
        if (fill_->failed) {
            fill_->mode = 0;
            fill_->armed.store(0, std::memory_order_release);
            return mode == 0 ? std::string() : "a GPU filler buffer failed earlier in this run";
        }
        fill_->mode = mode;
        fill_->armed.store(mode, std::memory_order_release);
        fill_->want.store(0, std::memory_order_release);
    }
    fill_->cv.notify_all();
    return {};
}

int Context::seg_fill_mode() const { return fill_ == nullptr ? 0 : fill_->armed.load(); }
double Context::seg_fill_buffer_us() const { return fill_ == nullptr ? 0.0 : fill_->one_us; }

Context::FillStats Context::fill_stats() const {
    FillStats f;
    if (fill_ != nullptr) {
        f.n = fill_->n.load();
        f.gpu_ms = (double) fill_->gpu_ns.load() / 1e6;
    }
    return f;
}
void Context::reset_fill_stats() {
    if (fill_ == nullptr) return;
    fill_->n = 0;
    fill_->gpu_ns = 0;
}

Context::~Context() {
    // the filler first: its thread uses a queue and buffers that must outlive it (see set_seg_fill)
    if (fill_ != nullptr) {
        {
            std::lock_guard<std::mutex> lk(fill_->mu);
            fill_->stop = true;
            fill_->mode = 0;
            fill_->armed.store(0);
        }
        fill_->cv.notify_all();
        if (fill_->th.joinable()) fill_->th.join();
        delete fill_;
        fill_ = nullptr;
    }
    stop_keepalive();
    delete impl_;
}

double Context::run_graph_seg(const std::vector<Dispatch>& ds, const std::vector<int>& cuts, const std::vector<int>& waits,
                              void (*at_cut)(void*, int), void* user, GraphProfile* profile) {
    if (!waits.empty() && waits.size() != cuts.size()) {
        throw std::runtime_error("Hy4 Metal runtime failure");
    }
    for (size_t s = 0; s < waits.size(); ++s) {
        const int hi = s + 1 < cuts.size() ? cuts[s + 1] : (int) ds.size();
        if (waits[s] < cuts[s] || waits[s] > hi) {
            throw std::runtime_error("Hy4 Metal runtime failure");
        }
    }
    const int fill = fill_ == nullptr ? 0 : fill_->armed.load(std::memory_order_acquire);
    if (fill == 2) fill_->last_ns.store(fill_now_ns(), std::memory_order_relaxed);
    @autoreleasepool {
        id<MTLCounterSampleBuffer> samples = nil;
        std::vector<double*> stage_totals;
        MTLTimestamp cpu0 = 0, gpu0 = 0;
        if (profile != nullptr) {
            profile->sampled = false;
            const size_t span_cap = profile->stages.size() + 2 * cuts.size();
            if (!profile->stages.empty()) samples = impl_->segment_timestamp_buffer(2 * span_cap);
            if (samples != nil) {
                stage_totals.reserve(span_cap);
                [impl_->dev sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];
            }
        }
        if (impl_->ev == nil) impl_->ev = [impl_->dev newSharedEvent];
        id<MTLSharedEvent> ev = impl_->ev;
        const uint64_t base = impl_->ev_v;
        id<MTLCommandBuffer> cb = [impl_->q commandBufferWithUnretainedReferences];
        const double t_enc0 = now_s();
        size_t i = 0;
        size_t stage = 0;
        auto encode_span = [&](size_t end, id<MTLComputeCommandEncoder> enc) {
            const size_t first = i;
            for (; i < end; ++i) {
                if (i > first && ds[i].barrier) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                encode(enc, ds[i]);
            }
            [enc endEncoding];
            ++seg_tm_.n_enc;
        };
        auto encoder = [&](size_t end) {
            // Two host cuts can share a dispatch index. That empty segment has no timestamp span.
            if (i == end && !waits.empty()) return;
            if (samples == nil) {
                encode_span(end, [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]);
                return;
            }
            while (i < end) {
                const auto& stages = profile->stages;
                while (stage + 1 < stages.size() && stages[stage + 1].first <= i) ++stage;
                const size_t next = stage + 1 < stages.size() ? std::min(end, stages[stage + 1].first) : end;
                MTLComputePassDescriptor* pass = [MTLComputePassDescriptor computePassDescriptor];
                pass.dispatchType = MTLDispatchTypeConcurrent;
                auto* attachment = pass.sampleBufferAttachments[0];
                attachment.sampleBuffer = samples;
                attachment.startOfEncoderSampleIndex = 2 * stage_totals.size();
                attachment.endOfEncoderSampleIndex = 2 * stage_totals.size() + 1;
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDescriptor:pass];
                if (enc == nil) throw std::runtime_error("Hy4 Metal runtime failure");
                stage_totals.push_back(stages[stage].total_ms);
                encode_span(next, enc);
            }
        };
        for (size_t s = 0; s <= cuts.size(); ++s) {
            encoder(s < cuts.size() ? (size_t) cuts[s] : ds.size());
            if (s < cuts.size()) {
                [cb encodeSignalEvent:ev value:base + 2 * s + 1];
                // the work between the signal and the wait: its own encoder, run while the host takes its turn
                if (!waits.empty() && waits[s] > cuts[s]) encoder((size_t) waits[s]);
                [cb encodeWaitForEvent:ev value:base + 2 * s + 2];
            }
        }
        seg_tm_.encode_ms += (now_s() - t_enc0) * 1e3;
        const double t_commit0 = now_s();
        [cb commit];
        impl_->touch();
        seg_tm_.commit_ms += (now_s() - t_commit0) * 1e3;
        ++seg_tm_.n_cb;
        seg_tm_.n_dis += ds.size();
        std::exception_ptr host_error;
        for (size_t s = 0; s < cuts.size(); ++s) {
            const uint64_t want = base + 2 * s + 1;
            // THE BUSY-WAIT, timed on its own.  It is the GPU still computing segment s, and it lies INSIDE the
            // command buffer's GPU span - so without this split it reads as "the GPU computed", while
            // `host_cut_ms` covers only the at_cut body below.
            const double t_spin0 = now_s();
            if (seg_wait_event()) {
                // A SHORT SPIN, THEN SLEEP.  The tight loop below burns a core for the whole segment, and on
                // this chip the CPU and the GPU share the memory controller and the power budget, so a spinning
                // core is not free to the thing it is waiting for.  Most cuts are not instant, so the spin
                // budget is short and the common case is the sleep.
                //
                // The listener is created once; `notifyListener:atValue:` fires on its queue when the GPU
                // signals, and the semaphore hands that to this thread.  The wait has a TIMEOUT and rechecks,
                // because a command buffer that failed will never reach `want` and a plain wait would hang
                // the engine instead of reporting the error - the status is only consulted on a timeout, so
                // the fast path does not pay for it.
                const double budget = spin_budget_s();
                while (ev.signaledValue < want && now_s() - t_spin0 < budget) {}
                if (ev.signaledValue < want) {
                    if (impl_->ev_listener == nil)
                        impl_->ev_listener = [[MTLSharedEventListener alloc]
                            initWithDispatchQueue:dispatch_queue_create("strata.seg", DISPATCH_QUEUE_SERIAL)];
                    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
                    [ev notifyListener:impl_->ev_listener
                              atValue:want
                                block:^(id<MTLSharedEvent>, uint64_t) { dispatch_semaphore_signal(sem); }];
                    while (ev.signaledValue < want) {
                        if (dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC)) == 0)
                            break;
                        if (cb.status >= MTLCommandBufferStatusCompleted) break;   // failed: reported below
                    }
                }
            } else {
                while (ev.signaledValue < want) {
                    if (cb.status >= MTLCommandBufferStatusCompleted) break;   // an error: the wait below reports it
                }
            }
            const double spin = (now_s() - t_spin0) * 1e3;
            seg_tm_.spin_ms += spin;
            const double t_cut0 = now_s();
            if (fill == 1) fill_->want.fetch_add(1, std::memory_order_release);
            // A committed command buffer must finish its waits even after a failed host turn.
            if (!host_error && cb.status < MTLCommandBufferStatusCompleted) {
                try { at_cut(user, (int) s); }
                catch (...) { host_error = std::current_exception(); }
            }
            if (fill == 1) fill_->want.fetch_sub(1, std::memory_order_release);
            if (fill == 2) fill_->last_ns.store(fill_now_ns(), std::memory_order_relaxed);
            const double cut = (now_s() - t_cut0) * 1e3;
            seg_tm_.cut_ms += cut;
            ++seg_tm_.n_cut;
            // n_cut has already counted this cut, hence the -1: the index is this cut's position in the
            // window, not the next one's.
            const size_t k = cut_period_ > 0 ? (seg_tm_.n_cut - 1) % (size_t) cut_period_ : s;
            if (k < (size_t) kMaxCuts) {
                seg_tm_.spin_at[k] += spin;
                seg_tm_.cut_at[k] += cut;
                ++seg_tm_.n_at[k];
            }
            ev.signaledValue = base + 2 * s + 2;
        }
        impl_->ev_v = base + 2 * cuts.size() + 2;
        const double t_wait0 = now_s();
        [cb waitUntilCompleted];
        seg_tm_.wait_ms += (now_s() - t_wait0) * 1e3;
        if (host_error) std::rethrow_exception(host_error);
        if (cb.status != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("Hy4 Metal runtime failure");
        }
        const double gpu_start = cb.GPUStartTime;
        const double gpu = cb.GPUEndTime - gpu_start;
        // A missing start can turn the end timestamp into an uptime-sized
        // duration. Leave those samples out and count them for the caller.
        const bool valid_gpu = gpu_start > 0 && gpu >= 0 && gpu <= now_s() - t_enc0 + 0.001;
        if (valid_gpu) seg_tm_.gpu_ms += gpu * 1e3;
        else ++seg_tm_.invalid_gpu_spans;
        if (!stage_totals.empty()) {
            MTLTimestamp cpu1 = 0, gpu1 = 0;
            [impl_->dev sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
            const size_t count = 2 * stage_totals.size();
            NSData* data = [samples resolveCounterRange:NSMakeRange(0, count)];
            if (data.length == count * sizeof(MTLCounterResultTimestamp) && cpu1 > cpu0 && gpu1 > gpu0) {
                const auto* ts = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
                size_t j = 0;
                for (; j < count; j += 2) {
                    const auto start = ts[j].timestamp, end = ts[j + 1].timestamp;
                    if (start == 0 || end == MTLCounterErrorValue || end < start) break;
                }
                if (j == count) {
                    const double ms_per_tick = 1e-6 * (double) (cpu1 - cpu0) / (double) (gpu1 - gpu0);
                    for (size_t k = 0; k < stage_totals.size(); ++k)
                        *stage_totals[k] += (double) (ts[2 * k + 1].timestamp - ts[2 * k].timestamp) * ms_per_tick;
                    profile->sampled = true;
                }
            }
        }
        return valid_gpu ? gpu : 0;
    }
}

double Context::run_wall(const std::vector<Dispatch>& ds, int reps, double* gpu_seconds) {
    double gpu = 0;
    const double t0 = now_s();
    for (int r = 0; r < reps; ++r) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [impl_->q commandBufferWithUnretainedReferences];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (const auto& d : ds) encode(enc, d);
            [enc endEncoding];
            [cb commit];
            impl_->touch();
            [cb waitUntilCompleted];
            gpu += cb.GPUEndTime - cb.GPUStartTime;
        }
    }
    if (gpu_seconds) *gpu_seconds = gpu;
    return now_s() - t0;
}

void Context::warm(const std::vector<Dispatch>& ds, double seconds) {
    const double t0 = now_s();
    while (now_s() - t0 < seconds) run(ds);
}

double Context::run_icb(const std::vector<Dispatch>& ds, int reps, bool barrier_between) {
    @autoreleasepool {
        size_t maxb = 1;
        for (const auto& d : ds) maxb = std::max(maxb, d.args.size());
        MTLIndirectCommandBufferDescriptor* desc = [MTLIndirectCommandBufferDescriptor new];
        desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
        desc.inheritBuffers = NO;
        desc.inheritPipelineState = NO;
        desc.maxKernelBufferBindCount = maxb;
        id<MTLIndirectCommandBuffer> icb = [impl_->dev newIndirectCommandBufferWithDescriptor:desc
                                                                              maxCommandCount:ds.size()
                                                                                      options:0];
        NSMutableSet* used = [NSMutableSet new];
        for (size_t i = 0; i < ds.size(); ++i) {
            const Dispatch& d = ds[i];
            require_fit(d, "indirect-command dispatch");
            id<MTLIndirectComputeCommand> c = [icb indirectComputeCommandAtIndex:i];
            [c setComputePipelineState:P(d.p)];
            for (size_t a = 0; a < d.args.size(); ++a) {
                if (!d.args[a].buf) {
                    throw std::runtime_error("Hy4 Metal runtime failure");
                }
                size_t o;
                id<MTLBuffer> mb = BO(*d.args[a].buf, d.args[a].off, o);
                [c setKernelBuffer:mb offset:o atIndex:a];
                [used addObject:mb];
            }
            if (d.tg_mem) [c setThreadgroupMemoryLength:(d.tg_mem + 15) / 16 * 16 atIndex:0];
            [c concurrentDispatchThreadgroups:MTLSizeMake(d.grid[0], d.grid[1], d.grid[2])
                        threadsPerThreadgroup:MTLSizeMake(d.tg[0], d.tg[1], d.tg[2])];
            if (barrier_between) [c setBarrier];
        }
        id<MTLCommandBuffer> cb = [impl_->q commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        for (id<MTLBuffer> b in used) [enc useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        for (int r = 0; r < reps; ++r) {
            [enc executeCommandsInBuffer:icb withRange:NSMakeRange(0, ds.size())];
            if (barrier_between) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        [enc endEncoding];
        [cb commit];
        impl_->touch();
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            throw std::runtime_error("Hy4 Metal runtime failure");
        }
        return cb.GPUEndTime - cb.GPUStartTime;
    }
}

}  // namespace mtl
