// SPDX-License-Identifier: MIT
#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <span>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "../stq_codec.h"

namespace {

namespace stq = stark::hy4::stq;
constexpr std::uint32_t width = 1536;
constexpr std::uint32_t release_experts = 512;
constexpr std::size_t words_per_expert = width * width / 4;
constexpr std::size_t bytes_per_expert = words_per_expert * sizeof(std::uint16_t);
constexpr std::uint16_t unwritten = 0x7e00;

struct alignas(16) Job {
    std::uint32_t expert;
    std::uint32_t first_row;
    std::uint32_t rows;
    std::uint32_t reserved = 0;
};

struct Case {
    std::uint32_t upstream_expert;
    std::uint32_t rows;
};

struct RowMap {
    std::uint32_t expert;
    std::uint32_t source_row;
};

static_assert(sizeof(Job) == 16 && sizeof(RowMap) == 8);

constexpr std::array cases{
    Case{0, 1}, Case{511, 2}, Case{15, 7}, Case{256, 15}, Case{33, 16},
    Case{477, 17}, Case{94, 23}, Case{127, 32}, Case{201, 31}, Case{312, 0},
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::array<char, 2 * CC_SHA256_DIGEST_LENGTH + 1> sha256(const void* bytes, std::size_t length) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(bytes, static_cast<CC_LONG>(length), digest);
    std::array<char, 2 * CC_SHA256_DIGEST_LENGTH + 1> hex{};
    constexpr char digits[] = "0123456789abcdef";
    for (std::size_t i = 0; i < sizeof(digest); ++i) {
        hex[2 * i] = digits[digest[i] >> 4];
        hex[2 * i + 1] = digits[digest[i] & 15];
    }
    return hex;
}

struct Shard {
    explicit Shard(const char* path) : fd(open(path, O_RDONLY)) {
        require(fd >= 0, "cannot open release layer shard");
    }

    ~Shard() { close(fd); }
    Shard(const Shard&) = delete;
    Shard& operator=(const Shard&) = delete;

    void read(std::uint64_t offset, std::size_t length, void* destination) const {
        auto* dst = static_cast<char*>(destination);
        while (length != 0) {
            const ssize_t got = pread(fd, dst, length, static_cast<off_t>(offset));
            if (got < 0 && errno == EINTR) continue;
            require(got > 0, "cannot read bounded shard slice");
            offset += static_cast<std::size_t>(got);
            dst += got;
            length -= static_cast<std::size_t>(got);
        }
    }

    int fd;
};

NSDictionary* read_header(const Shard& shard, std::uint64_t& data_start) {
    std::uint64_t length = 0;
    shard.read(0, sizeof(length), &length);
    require(length > 0 && length <= 1024 * 1024, "invalid release shard header length");
    NSMutableData* header = [NSMutableData dataWithLength:length];
    shard.read(sizeof(length), length, header.mutableBytes);
    data_start = sizeof(length) + length;
    id parsed = [NSJSONSerialization JSONObjectWithData:header options:0 error:nullptr];
    require([parsed isKindOfClass:NSDictionary.class], "invalid release shard header");
    NSDictionary* meta = parsed[@"__metadata__"];
    require([meta isKindOfClass:NSDictionary.class] &&
            [meta[@"format"] isEqual:@"stq"] && [meta[@"k"] isEqual:@"4"] &&
            [meta[@"window"] isEqual:@"16"] && [meta[@"codebook"] isEqual:@"pair1"] &&
            [meta[@"tile_order"] isEqual:@"pair_major"] &&
            [meta[@"pack_revision"] isEqual:@"capture-2a0815359e97"],
        "expected PAIR1 K4 pair-major release shard");
    return parsed;
}

void load_expert_slices(const Shard& shard, NSDictionary* header, std::uint64_t data_start,
                        NSString* projection, NSString* tensor, NSString* dtype, NSArray* shape,
                        id<MTLBuffer> buffer) {
    NSString* name = [NSString stringWithFormat:@"model.layers.1.mlp.switch_mlp.%@_proj.g0.%@",
                                                projection, tensor];
    NSDictionary* entry = header[name];
    require([entry isKindOfClass:NSDictionary.class] && [entry[@"dtype"] isEqual:dtype] &&
            [entry[@"shape"] isEqual:shape],
        "unexpected packed expert tensor shape");
    NSArray* offsets = entry[@"data_offsets"];
    require([offsets isKindOfClass:NSArray.class] && offsets.count == 2 &&
            [offsets[0] isKindOfClass:NSNumber.class] && [offsets[1] isKindOfClass:NSNumber.class],
        "invalid packed expert tensor offsets");
    const std::uint64_t start = [offsets[0] unsignedLongLongValue];
    const std::uint64_t end = [offsets[1] unsignedLongLongValue];
    const std::size_t bytes = buffer.length / cases.size();
    struct stat info{};
    require(fstat(shard.fd, &info) == 0 && std::uint64_t(info.st_size) >= data_start &&
            start < end && end <= std::uint64_t(info.st_size) - data_start &&
            end - start == release_experts * bytes, "invalid packed expert tensor extent");
    auto* dst = static_cast<char*>(buffer.contents);
    for (std::size_t i = 0; i < cases.size(); ++i) {
        shard.read(data_start + start + cases[i].upstream_expert * bytes, bytes, dst + i * bytes);
    }
}

// A deliberately independent row-major CPU product, with the published CPU
// codec supplying the exact F16 weight bits and F64 accumulating the products.
template <typename Input>
std::vector<double> reference_inner(const stq::TrellisView& trellis, std::span<const Job> jobs,
                                    const Input* input, std::size_t values) {
    std::vector<double> output(values);
    std::array<std::uint16_t, stq::tile_values> decoded;
    for (const auto& job : jobs) {
        const auto* x = input + std::size_t(job.first_row) * width;
        auto* y = output.data() + std::size_t(job.first_row) * width;
        for (std::size_t ot = 0; job.rows != 0 && ot < width / stq::tile_size; ++ot) {
            double sums[32][stq::tile_size] = {};
            for (std::size_t it = 0; it < width / stq::tile_size; ++it) {
                require(stq::decode_tile(trellis, job.expert, it, ot, decoded), "CPU tile decode failed");
                for (std::size_t k = 0; k < stq::tile_size; ++k) {
                    for (std::size_t c = 0; c < stq::tile_size; ++c) {
                        const double w = std::bit_cast<_Float16>(decoded[k * stq::tile_size + c]);
                        for (std::size_t r = 0; r < job.rows; ++r) {
                            sums[r][c] += w * double(x[r * width + it * stq::tile_size + k]);
                        }
                    }
                }
            }
            for (std::size_t r = 0; r < job.rows; ++r) {
                std::copy_n(sums[r], stq::tile_size, y + r * width + ot * stq::tile_size);
            }
        }
    }
    return output;
}

struct ErrorStats {
    std::size_t compared = 0;
    double max_abs = 0;
    double squared_error = 0;
    double squared_reference = 0;

    double add(double actual, double reference) {
        require(std::isfinite(actual) && std::isfinite(reference), "nonfinite live output");
        const double error = std::abs(actual - reference);
        max_abs = std::max(max_abs, error);
        squared_error += error * error;
        squared_reference += reference * reference;
        ++compared;
        return error;
    }

    double relative_rms() const {
        require(squared_reference > 0, "validation reference was empty");
        return std::sqrt(squared_error / squared_reference);
    }

    double peak_over_reference_rms() const {
        return max_abs / std::sqrt(squared_reference / double(compared));
    }
};

ErrorStats check_raw_output(std::span<const Job> jobs, std::span<const double> reference,
                            const std::uint16_t* output, std::uint32_t total_rows, std::size_t& exact) {
    ErrorStats stats;
    std::uint32_t checked_rows = 0;
    for (const auto& job : jobs) {
        for (; checked_rows < job.first_row; ++checked_rows) {
            const auto* row = output + std::size_t(checked_rows) * width;
            require(std::all_of(row, row + width, [](auto b) { return b == unwritten; }),
                "dispatch wrote outside its live row ranges");
        }
        require(job.first_row + job.rows < total_rows, "invalid validation job bounds");
        for (std::size_t r = 0; r < job.rows; ++r) {
            const bool basis_row = r == 0 || r + 1 == job.rows;
            for (std::size_t c = 0; c < width; ++c) {
                const std::size_t i = (job.first_row + r) * width + c;
                const auto got_bits = output[i];
                const auto want_half = _Float16(reference[i]);
                const auto got = double(std::bit_cast<_Float16>(got_bits));
                const auto want = double(want_half);
                if (basis_row) require(got == want, "basis row differed from the CPU weights");
                const double error = stats.add(got, want);
                require(error <= 0.001 * std::abs(want) + 0.000002, "projection exceeded element tolerance");
                exact += got_bits == std::bit_cast<std::uint16_t>(want_half);
            }
        }
        checked_rows += job.rows;
    }
    for (; checked_rows < total_rows; ++checked_rows) {
        const auto* row = output + std::size_t(checked_rows) * width;
        require(std::all_of(row, row + width, [](auto b) { return b == unwritten; }),
            "dispatch wrote beyond the final live row");
    }
    return stats;
}

template <typename Float>
void reference_hadamard(std::span<Float, stq::block_size> block) {
    for (std::size_t step = 1; step < block.size(); step *= 2) {
        for (std::size_t group = 0; group < block.size(); group += 2 * step) {
            for (std::size_t i = group; i < group + step; ++i) {
                const Float lo = block[i], hi = block[i + step];
                block[i] = lo + hi;
                block[i + step] = lo - hi;
            }
        }
    }
    for (Float& value : block) value *= Float(0.08838834764831844055);
}

template <typename Float, typename Result>
std::vector<Result> reference_prepare(const float* input, const _Float16* suh, std::span<const RowMap> rows) {
    std::array<Float, stq::block_size> block;
    std::vector<Result> result(rows.size() * width);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const auto* source = input + std::size_t(rows[r].source_row) * width;
        const auto* scale = suh + std::size_t(rows[r].expert) * width;
        for (std::size_t k = 0; k < width; k += block.size()) {
            for (std::size_t j = 0; j < block.size(); ++j) block[j] = Float(source[k + j]) * Float(scale[k + j]);
            reference_hadamard<Float>(block);
            for (std::size_t j = 0; j < block.size(); ++j) result[r * width + k + j] = Result(block[j]);
        }
    }
    return result;
}

template <typename Float, typename Input>
std::vector<Float> reference_finish(const Input* input, const _Float16* svh, std::span<const RowMap> rows) {
    std::vector<Float> result(rows.size() * width);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const auto* scale = svh + std::size_t(rows[r].expert) * width;
        for (std::size_t k = 0; k < width; k += stq::block_size) {
            const std::size_t start = r * width + k;
            std::span<Float, stq::block_size> block(result.data() + start, stq::block_size);
            for (std::size_t j = 0; j < block.size(); ++j) block[j] = Float(input[start + j]);
            reference_hadamard<Float>(block);
            for (std::size_t j = 0; j < block.size(); ++j) block[j] *= Float(scale[k + j]);
        }
    }
    return result;
}

template <typename Encode>
std::vector<double> time_dispatch(id<MTLCommandQueue> queue, int samples, Encode encode) {
    std::vector<double> timings;
    for (int i = -2; i < samples; ++i) {
        @autoreleasepool {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            require(encoder != nil, "cannot encode validation command");
            encode(encoder);
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            require(command.status == MTLCommandBufferStatusCompleted, "Metal command failed");
            if (i >= 0) {
                const auto ms = (command.GPUEndTime - command.GPUStartTime) * 1e3;
                require(command.GPUStartTime > 0 && ms > 0, "invalid Metal command timestamps");
                timings.push_back(ms);
            }
        }
    }
    std::sort(timings.begin(), timings.end());
    return timings;
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name) {
    id<MTLFunction> kernel = [library newFunctionWithName:name];
    require(kernel != nil, "packed expert kernel absent from library");
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernel error:&error];
    if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
    require(pipeline.threadExecutionWidth == 32 && pipeline.maxTotalThreadsPerThreadgroup >= 128,
            "unsupported packed expert pipeline");
    return pipeline;
}

void encode_grouped(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
                    id<MTLBuffer> weights, id<MTLBuffer> input, id<MTLBuffer> output,
                    std::span<const Job> jobs) {
    const std::uint32_t shape[] = {width, width};
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:weights offset:0 atIndex:0];
    [encoder setBuffer:input offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:jobs.data() length:jobs.size_bytes() atIndex:3];
    [encoder setBytes:shape length:sizeof(shape) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(width / 128, jobs.size(), 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

void encode_io(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
               id<MTLBuffer> input, id<MTLBuffer> scales, id<MTLBuffer> output,
               std::span<const RowMap> rows) {
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:scales offset:0 atIndex:1];
    [encoder setBytes:rows.data() length:rows.size_bytes() atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBytes:&width length:sizeof(width) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(width / 128, rows.size(), 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

void run(const char* source_path, const char* io_path, const char* shard_path, int samples) {
    const Shard shard(shard_path);
    std::uint64_t data_start = 0;
    NSDictionary* header = read_header(shard, data_start);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil && [device supportsFamily:MTLGPUFamilyMetal4], "requires a Metal 4 device");
    NSError* err = nil;
    NSString* source = [NSString stringWithContentsOfFile:@(source_path) encoding:NSUTF8StringEncoding error:&err];
    NSString* io = [NSString stringWithContentsOfFile:@(io_path) encoding:NSUTF8StringEncoding error:&err];
    require(source != nil && io != nil, "cannot load Metal shader sources");
    NSString* combined = [NSString stringWithFormat:@"%@\n%@", source, io];
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    options.mathMode = MTLMathModeSafe;
    id<MTLLibrary> library = [device newLibraryWithSource:combined options:options error:&err];
    if (!library) throw std::runtime_error(err.localizedDescription.UTF8String);
    id<MTLComputePipelineState> grouped = make_pipeline(device, library, @"stark_hy4_stq_pair1_k4_grouped");
    id<MTLComputePipelineState> prepare = make_pipeline(device, library, @"stark_hy4_stq_pair1_prepare");
    id<MTLComputePipelineState> finish = make_pipeline(device, library, @"stark_hy4_stq_pair1_finish");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(queue != nil, "cannot create validation command queue");

    std::array<Job, cases.size()> raw_jobs{}, chain_jobs{};
    std::vector<RowMap> row_map;
    constexpr std::uint32_t source_rows = 37;
    std::uint32_t row_count = 1, live_rows = 0;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto expert = std::uint32_t(i);
        raw_jobs[i] = {expert, row_count, cases[i].rows};
        chain_jobs[i] = {expert, live_rows, cases[i].rows};
        for (std::uint32_t r = 0; r < cases[i].rows; ++r) {
            row_map.push_back({expert, (expert * 19 + r * 13) % source_rows});
        }
        row_count += cases[i].rows + 1;
        live_rows += cases[i].rows;
    }
    const std::size_t values = std::size_t(row_count) * width;
    const std::size_t live_values = std::size_t(live_rows) * width;
    const auto shared = MTLResourceStorageModeShared;
    id<MTLBuffer> weights = [device newBufferWithLength:cases.size() * bytes_per_expert options:shared];
    id<MTLBuffer> input = [device newBufferWithLength:values * sizeof(_Float16) options:shared];
    id<MTLBuffer> output = [device newBufferWithLength:values * sizeof(std::uint16_t) options:shared];
    id<MTLBuffer> suh = [device newBufferWithLength:cases.size() * width * sizeof(_Float16) options:shared];
    id<MTLBuffer> svh = [device newBufferWithLength:suh.length options:shared];
    id<MTLBuffer> source_input = [device newBufferWithLength:source_rows * width * sizeof(float) options:shared];
    id<MTLBuffer> final_output = [device newBufferWithLength:live_values * sizeof(float) options:shared];
    require(weights && input && output && suh && svh && source_input && final_output,
            "cannot allocate validation buffers");
    auto* x = static_cast<_Float16*>(input.contents);
    auto* y = static_cast<std::uint16_t*>(output.contents);
    const stq::TrellisView trellis{
        {static_cast<const std::uint16_t*>(weights.contents), cases.size() * words_per_expert},
        stq::Format::pair1_k4, width, width, cases.size(),
    };
    require(trellis.valid(), "invalid validation trellis view");
    auto* source_values = static_cast<float*>(source_input.contents);
    std::uint32_t source_rng = 0x5499e9u;
    for (std::size_t i = 0; i < source_rows * width; ++i) {
        source_rng ^= source_rng << 13;
        source_rng ^= source_rng >> 17;
        source_rng ^= source_rng << 5;
        source_values[i] = float(int(source_rng & 8191u) - 4096) / 16384.0f;
    }
    std::fill_n(source_values, width, 0.0f);
    source_values[width - 1] = 1.0f;
    for (NSString* projection in @[@"gate", @"up", @"down"]) {
        load_expert_slices(shard, header, data_start, projection, @"trellis_pm", @"U16",
                           @[@512, @48, @96, @2, @64], weights);
        load_expert_slices(shard, header, data_start, projection, @"suh", @"F16", @[@512, @1536], suh);
        load_expert_slices(shard, header, data_start, projection, @"svh", @"F16", @[@512, @1536], svh);
        std::uint32_t rng = 0x73ba9u;
        for (std::size_t i = 0; i < values; ++i) {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            x[i] = _Float16(float(int(rng & 2047u) - 1024) / 16384.0f);
        }
        // Basis rows hit the first and last K tile plus a tail of the second
        // MMA block. Other rows exercise long, dense reductions.
        for (const auto& job : raw_jobs) {
            if (job.rows == 0) continue;
            auto* first = x + std::size_t(job.first_row) * width;
            std::fill_n(first, width, _Float16(0));
            first[job.expert] = _Float16(1);
            if (job.rows >= 2) {
                auto* last = first + std::size_t(job.rows - 1) * width;
                std::fill_n(last, width, _Float16(0));
                last[width - 1 - job.expert] = _Float16(-1);
            }
        }
        std::fill_n(y, values, unwritten);
        const auto timings = time_dispatch(queue, samples, [&](id<MTLComputeCommandEncoder> encoder) {
            encode_grouped(encoder, grouped, weights, input, output, raw_jobs);
        });
        std::size_t exact = 0;
        const auto reference = reference_inner(trellis, raw_jobs, x, values);
        const auto raw = check_raw_output(raw_jobs, reference, y, row_count, exact);
        require(raw.relative_rms() <= 0.00005, "projection exceeded RMS tolerance");
        require(raw.compared == live_values, "validation skipped live outputs");
        const auto weights_hash = sha256(weights.contents, weights.length);
        const auto output_hash = sha256(output.contents, output.length);
        std::printf("%s raw %ux%u experts=%zu live_rows=%u compared=%zu exact=%zu max_abs=%.8g rel_rms=%.8g "
                    "gpu_command_ms_median=%.4f min=%.4f max=%.4f samples=%zu "
                    "weights_sha256=%s output_sha256=%s\n",
            projection.UTF8String, width, width, cases.size() - 1, live_rows, raw.compared, exact, raw.max_abs,
            raw.relative_rms(), timings[timings.size() / 2], timings.front(), timings.back(), timings.size(),
            weights_hash.data(), output_hash.data());

        std::fill_n(y, values, unwritten);
        const auto chain = time_dispatch(queue, samples, [&](id<MTLComputeCommandEncoder> encoder) {
            encode_io(encoder, prepare, source_input, suh, input, row_map);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_grouped(encoder, grouped, weights, input, output, chain_jobs);
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_io(encoder, finish, output, svh, final_output, row_map);
        });
        const auto* suh_values = static_cast<const _Float16*>(suh.contents);
        const auto* svh_values = static_cast<const _Float16*>(svh.contents);
        const auto* final_values = static_cast<const float*>(final_output.contents);
        const auto prepared = reference_prepare<float, _Float16>(source_values, suh_values, row_map);
        require(std::equal(prepared.begin(), prepared.end(), x, [](_Float16 a, _Float16 b) {
            return std::bit_cast<std::uint16_t>(a) == std::bit_cast<std::uint16_t>(b);
        }), "prepare differed from the CPU binary16 reference");
        const auto finish_on_gpu_inner = reference_finish<float>(static_cast<const _Float16*>(output.contents),
                                                                 svh_values, row_map);
        require(std::equal(finish_on_gpu_inner.begin(), finish_on_gpu_inner.end(), final_values),
                "finish differed from the CPU reference on the GPU inner output");
        const auto chain_inner = reference_inner(trellis, chain_jobs, prepared.data(), live_values);
        std::vector<_Float16> rounded_inner(chain_inner.begin(), chain_inner.end());
        const auto boundary = reference_finish<float>(rounded_inner.data(), svh_values, row_map);
        const auto original_input = reference_prepare<double, double>(source_values, suh_values, row_map);
        const auto original_inner = reference_inner(trellis, chain_jobs, original_input.data(), live_values);
        const auto original = reference_finish<double>(original_inner.data(), svh_values, row_map);
        ErrorStats boundary_stats, original_stats;
        for (std::size_t i = 0; i < live_values; ++i) {
            boundary_stats.add(final_values[i], boundary[i]);
            original_stats.add(final_values[i], original[i]);
        }
        const auto suh_hash = sha256(suh.contents, suh.length);
        const auto svh_hash = sha256(svh.contents, svh.length);
        const auto final_hash = sha256(final_output.contents, final_output.length);
        std::printf("%s chain %ux%u experts=%zu live_rows=%u compared=%zu prepare_bits_exact=%zu finish_values_exact=%zu "
                    "boundary_max_abs=%.8g boundary_rel_rms=%.8g boundary_peak_over_ref_rms=%.8g "
                    "original_max_abs=%.8g original_rel_rms=%.8g original_peak_over_ref_rms=%.8g "
                    "gpu_command_ms_median=%.4f min=%.4f max=%.4f samples=%zu "
                    "suh_sha256=%s svh_sha256=%s output_sha256=%s\n",
            projection.UTF8String, width, width, cases.size() - 1, live_rows, live_values, live_values, live_values,
            boundary_stats.max_abs, boundary_stats.relative_rms(), boundary_stats.peak_over_reference_rms(),
            original_stats.max_abs, original_stats.relative_rms(), original_stats.peak_over_reference_rms(),
            chain[chain.size() / 2], chain.front(), chain.back(), chain.size(),
            suh_hash.data(), svh_hash.data(), final_hash.data());
        require(boundary_stats.relative_rms() <= 0.00005 && boundary_stats.peak_over_reference_rms() <= 0.004,
                "chain exceeded the same-boundary CPU tolerance");
        require(original_stats.relative_rms() <= 0.0005 && original_stats.peak_over_reference_rms() <= 0.008,
                "chain exceeded the unrounded original-domain CPU tolerance");
    }
}

}  // namespace

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        if (argc != 4) {
            std::fprintf(stderr, "usage: %s <pair1_k4_grouped.metal> <pair1_k4_io.metal> <model-layer-01.safetensors>\n",
                         argv[0]);
            return 2;
        }
        try {
            run(argv[1], argv[2], argv[3], 11);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "%s\n", error.what());
            return 1;
        }
    }
    return 0;
}
