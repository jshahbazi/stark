// SPDX-License-Identifier: MIT
//! Isolated, serial GLM engine boundary. Model load/forward is never implicit.
const std = @import("std");
const plugin = @import("mlx_stream");
const sdk = @import("sdk");
const mlx = @import("mlx_host").mlx;
const Arch = plugin.glm5_next.arch;
const experimental_prefix = @import("bridge_options").experimental_prefix;
const private_native_caps = @import("bridge_options").private_native_caps;
const allocator = std.heap.c_allocator;
const native_protocol = @import("native_protocol.zig");
const abi_version: u32 = 1;
const max_ceiling: u64 = 112 << 30;
const default_margin: u64 = 2 << 30;
extern "c" fn stark_glm_mlx_get_wired_limit(result: *usize) c_int;

pub const std_options: std.Options = .{ .logFn = silentLog };
fn silentLog(comptime _: std.log.Level, comptime _: @EnumLiteral(), comptime _: []const u8, _: anytype) void {}

const Status = enum(u32) {
    ok = 0, invalid_argument = 1, abi_mismatch = 2, context_limit = 3,
    cancelled = 4, unsupported = 5, out_of_memory = 6, io_failure = 7,
    device_failure = 8, reload_required = 9, busy = 10, invalid_state = 11,
};
const Options = extern struct {
    struct_size: u32, abi_version: u32, context_tokens: u64,
    gpu_ceiling_bytes: u64, wired_margin_bytes: u64,
    expert_rows: u32, expert_prefill_rows: u32,
};
const Info = extern struct {
    struct_size: u32, abi_version: u32, vocab_size: u32, prefill_chunk_tokens: u32,
    context_tokens: u64, position: u64, capabilities: u64, fenced: u32, reserved: u32,
};
const NativeSampling = native_protocol.Sampling;
const NativeRound = native_protocol.Round;
comptime {
    if (@sizeOf(Options) != 40 or @sizeOf(Info) != 48) @compileError("C ABI layout changed");
    if (@sizeOf(NativeSampling) != 32 or @sizeOf(NativeRound) != 48 or
        @alignOf(NativeSampling) != 8 or @alignOf(NativeRound) != 8)
        @compileError("native decode C ABI layout changed");
}

const Model = struct {
    io_runtime: std.Io.Threaded,
    owner: std.Thread.Id,
    module: *Arch.Module,
    config: *Arch.Config,
    weights: sdk.Weights,
    stream: mlx.mlx_stream,
    readout: ?mlx.mlx_array = null,
    cancelled: std.atomic.Value(bool) = .init(false),
    fenced: bool = false,
    request_invalid: bool = false,
    fresh: bool = true,
    wired_capacity: usize = 0,
    previous_wired_capacity: ?usize = null,
    native_loader: bool = false,
    native_head_loaded: bool = false,
    native_mtp_enabled: bool = false,
    native_started: bool = false,
    native_ended: bool = false,
    native_sampling: ?sdk.SamplingParams = null,
    native_pending: ?u32 = null,
};
// The plugin reader and MLX error callback are process-owned, so admit one model.
var process_claimed = std.atomic.Value(bool).init(false);

fn code(s: Status) u32 { return @intFromEnum(s); }
fn statusOf(e: anyerror) Status {
    if (mlx.errorPending()) {
        const memory = if (mlx.peekErrorName()) |n| std.mem.eql(u8, n, "OutOfMemory") else false;
        var discarded: [512]u8 = undefined;
        _ = mlx.takeError(&discarded);
        @memset(&discarded, 0);
        return if (e == error.WiredRestoreFailed) .device_failure else if (memory) .out_of_memory else .device_failure;
    }
    return switch (e) {
        error.OutOfMemory => .out_of_memory,
        error.ContextOverBill, error.NativeBillDoesNotFit, error.NoSlotRows, error.OverBudget => .context_limit,
        error.Glm5NextTokenOutOfRange, error.Glm5NextEmptyPrompt, error.Glm5NextEmptyStep => .invalid_argument,
        error.Cancelled => .cancelled,
        error.UnsupportedPack, error.NoGpu => .unsupported,
        error.FileNotFound, error.AccessDenied, error.InputOutput, error.NoWeightFiles, error.NotDir,
        error.Glm5NextResidentIndex, error.Glm5NextPackDir => .io_failure,
        error.InvalidConfig, error.ConfigNotJson, error.ConfigNotObject, error.NameTooLong => .invalid_argument,
        else => .device_failure,
    };
}
fn owned(m: *Model) Status {
    if (m.owner != std.Thread.getCurrentId()) return .invalid_state;
    return .ok;
}
fn ready(m: *Model) Status {
    const ownership = owned(m);
    if (ownership != .ok) return ownership;
    if (m.fenced) return .reload_required;
    if (m.cancelled.load(.acquire)) {
        _ = reject(m, .cancelled);
        return .cancelled;
    }
    if (m.request_invalid) return .invalid_state;
    return .ok;
}
fn reject(m: *Model, status: Status) u32 {
    // Called only after owned-thread admission. Every failed request invalidates
    // reuse; pure argument errors do not poison the loaded device context.
    m.module.clearPrefixCapture();
    m.module.invalidatePrefix();
    clearReadout(m);
    m.request_invalid = true;
    return code(status);
}
fn clearReadout(m: *Model) void {
    if (m.readout) |x| _ = mlx.mlx_array_free(x);
    m.readout = null;
}
fn flush(m: *Model) !void {
    // Synchronize even when flush reports an error, before releasing ownership.
    var first: ?anyerror = null;
    if (m.readout) |x| mlx.check(mlx.mlx_array_eval(x)) catch |e| { first = e; };
    switch (m.module.src) { inline else => |src| src.flush() catch |e| { first = e; } }
    mlx.check(mlx.mlx_synchronize(m.stream)) catch |e| { if (first == null) first = e; };
    mlx.checkError() catch |e| { if (first == null) first = e; };
    if (first) |e| return e;
}
fn requestDrain(m: *Model) !void {
    var first: ?anyerror = null;
    if (m.native_loader) m.module.starkDrainRoutes() catch |e| { first = e; };
    flush(m) catch |e| { if (first == null) first = e; };
    if (first) |e| return e;
}
fn fail(m: *Model, e: anyerror) u32 {
    const status = statusOf(e);
    m.module.clearPrefixCapture();
    m.module.invalidatePrefix();
    if (status != .cancelled) m.fenced = true;
    // Keep the handle alive for owning-thread drain/destruction after failure.
    return code(status);
}
fn checkResult(m: *Model, row: mlx.mlx_array) !void {
    errdefer _ = mlx.mlx_array_free(row);
    if (mlx.mlx_array_dtype(row) != .float32 or mlx.mlx_array_size(row) != m.config.model.vocab_size)
        return error.InvalidReadout;
    try mlx.check(mlx.mlx_array_eval(row));
    try mlx.checkError();
    if (m.cancelled.load(.acquire)) return error.Cancelled;
    clearReadout(m);
    m.readout = row;
}

fn installWired(m: *Model, target: usize) !void {
    if (target == 0) return error.OverBudget;
    var previous: usize = 0;
    try mlx.check(stark_glm_mlx_get_wired_limit(&previous));
    m.previous_wired_capacity = previous;
    // Both resizes can fail after mutation. The observational getter saves the
    // previous value first, including when it is nonzero.
    var ignored: usize = 0;
    try mlx.check(mlx.mlx_set_wired_limit(&ignored, 0));
    try mlx.check(mlx.mlx_set_wired_limit(&ignored, target));
    m.wired_capacity = target;
}

fn restoreWired(m: *Model) !void {
    const previous = m.previous_wired_capacity orelse return;
    var failed = false;
    // Call only after module/weight handles and the stream have been freed.
    mlx.check(mlx.mlx_clear_cache()) catch { failed = true; };
    var ignored: usize = 0;
    mlx.check(mlx.mlx_set_wired_limit(&ignored, previous)) catch { failed = true; };
    if (failed) return error.WiredRestoreFailed;
    m.previous_wired_capacity = null;
    m.wired_capacity = 0;
}

pub export fn stark_glm_abi_version() u32 { return abi_version; }
pub export fn stark_glm_status_name(status: u32) [*:0]const u8 {
    return switch (status) {
        0 => "OK", 1 => "INVALID_ARGUMENT", 2 => "ABI_MISMATCH", 3 => "CONTEXT_LIMIT",
        4 => "CANCELLED", 5 => "UNSUPPORTED", 6 => "OUT_OF_MEMORY", 7 => "IO_FAILURE",
        8 => "DEVICE_FAILURE", 9 => "RELOAD_REQUIRED", 10 => "BUSY", 11 => "INVALID_STATE",
        else => "UNKNOWN_STATUS",
    };
}

pub export fn stark_glm_load(path_ptr: ?[*]const u8, path_len: usize, options_ptr: ?*const Options, result_ptr: ?*?*Model) u32 {
    return loadEntry(path_ptr, path_len, options_ptr, result_ptr, false, false);
}
pub export fn stark_glm_load_native(path_ptr: ?[*]const u8, path_len: usize, options_ptr: ?*const Options, load_mtp_head: u32, result_ptr: ?*?*Model) u32 {
    if (load_mtp_head > 1) return code(.invalid_argument);
    return loadEntry(path_ptr, path_len, options_ptr, result_ptr, true, load_mtp_head == 1);
}
fn loadEntry(path_ptr: ?[*]const u8, path_len: usize, options_ptr: ?*const Options, result_ptr: ?*?*Model, native_loader: bool, load_head: bool) u32 {
    const options = options_ptr orelse return code(.invalid_argument);
    const result = result_ptr orelse return code(.invalid_argument);
    if (options.struct_size != @sizeOf(Options) or options.abi_version != abi_version) return code(.abi_mismatch);
    if (result.* != null) return code(.invalid_argument);
    if (path_len == 0 or path_len > 4096) return code(.invalid_argument);
    const path = (path_ptr orelse return code(.invalid_argument))[0..path_len];
    if (path[0] != '/' or std.mem.indexOfScalar(u8, path, 0) != null or !std.unicode.utf8ValidateSlice(path))
        return code(.invalid_argument);
    const margin = if (options.wired_margin_bytes == 0) default_margin else options.wired_margin_bytes;
    if (options.context_tokens == 0 or options.context_tokens > 1_048_576 or options.gpu_ceiling_bytes == 0 or
        options.gpu_ceiling_bytes > max_ceiling or margin >= options.gpu_ceiling_bytes or
        options.expert_rows > 288 or options.expert_prefill_rows > 288) return code(.invalid_argument);
    if (process_claimed.cmpxchgStrong(false, true, .acq_rel, .acquire) != null) return code(.busy);
    const m = load(path, options.*, margin, native_loader, load_head) catch |e| {
        process_claimed.store(false, .release);
        return code(statusOf(e));
    };
    result.* = m;
    return code(.ok);
}

fn load(path: []const u8, options: Options, margin: u64, native_loader: bool, load_head: bool) !*Model {
    // This is the first entry allowed to touch MLX. Pure ABI checks return above.
    mlx.installErrorHandler();
    var discarded: [512]u8 = undefined;
    _ = mlx.takeError(&discarded);
    @memset(&discarded, 0);
    var version: [64]u8 = undefined;
    try sdk.negotiate(plugin.plugin, try sdk.host(&version));
    try Arch.claimProcess();
    errdefer Arch.releaseProcess();
    const m = try allocator.create(Model);
    errdefer allocator.destroy(m);
    m.* = .{ .io_runtime = std.Io.Threaded.init(allocator, .{}), .owner = std.Thread.getCurrentId(),
        .module = undefined, .config = undefined, .weights = undefined, .stream = .{},
        .native_loader = native_loader, .native_head_loaded = load_head };
    errdefer m.io_runtime.deinit();
    loadInto(m, path, options, margin) catch |e| {
        // loadInto's errdefers release every model buffer before restoring the
        // process-local residency capacity. Restoration failure has one status.
        restoreWired(m) catch return error.WiredRestoreFailed;
        return e;
    };
    return m;
}

fn loadInto(m: *Model, path: []const u8, options: Options, margin: u64) !void {
    const io = m.io_runtime.io();
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var dir = try std.Io.Dir.openDirAbsolute(io, path, .{});
    defer dir.close(io);
    const text = try dir.readFileAlloc(io, "config.json", arena.allocator(), .limited(16 << 20));
    const peek = try sdk.ConfigPeek.parse(arena.allocator(), path, text);
    var diag: sdk.Diag = .{};
    m.config = try Arch.parse(allocator, &peek, &diag);
    errdefer Arch.freeConfig(allocator, m.config);
    const cfg = m.config;
    if (cfg.expert_family != .exl3) return error.UnsupportedPack;
    // The original entry remains headless. Preload is charged by the upstream
    // bill even when the staged native loader parks the head for an off request.
    cfg.mtp_lane = m.native_head_loaded;
    if (m.native_head_loaded) {
        cfg.draft_depth = 3;
        cfg.draft_gate = 0.5;
        cfg.draft_head = .bf16;
    }
    cfg.kda_verify_named = false;
    cfg.draft_head_named = m.native_head_loaded;
    cfg.mtp_typical_delta_set = false;
    cfg.mtp_acceptance = .exact;
    cfg.dump_dir_len = 0;
    cfg.tally.file = .off;
    cfg.residency.record_len = 0;
    cfg.residency.record_named = false;
    cfg.residency.label_len = 0;
    cfg.residency.label_named = false;
    // Finish the phase before returning from prefill; reset has no pending tail.
    cfg.prompt.ttft_tail = .off;
    const nonfile = sdk.memory.boxUsedBytes() orelse return error.MemoryReadingUnavailable;
    // Budget CPU checkpoint + independent restore/copy peak explicitly. No
    // snapshot payload is included in the upstream inference-only bill.
    const head_snapshot_reserve: u64 = if (experimental_prefix and m.native_head_loaded)
        2 * @as(u64, cfg.model.hidden_size) * 2 +
        2 * 16 * (2 * @as(u64, cfg.model.index_head_dim)) * 2
    else 0;
    const snapshot_reserve: u64 = (if (experimental_prefix)
        2 * cfg.model.kdaStateBytes() + 2 * @as(u64, cfg.model.nDsa()) * 16 * (2 * cfg.model.index_head_dim) * 2 + options.context_tokens * 4
    else 0) + head_snapshot_reserve;
    const stop = std.math.add(u64, margin, snapshot_reserve) catch return error.OverBudget;
    if (stop >= options.gpu_ceiling_bytes) return error.OverBudget;
    cfg.wired_margin_bytes = stop;
    cfg.effective_context = options.context_tokens;
    // Pack-supplied operator pins must never override bridge admission.
    cfg.expert_rows = if (options.expert_rows > 0) options.expert_rows else null;
    cfg.expert_prefill_rows = if (options.expert_prefill_rows > 0) options.expert_prefill_rows else null;
    // Match the normal host: count and evict this pack's file cache, retaining
    // the foreign-file cache charge and the non-file floor in the baseline.
    cfg.* = try Arch.admitted(io, cfg, &.{ .memory_baseline_bytes = nonfile,
        .effective_context = options.context_tokens, .wired_margin_bytes = stop,
        .expert_rows = cfg.expert_rows, .expert_prefill_rows = cfg.expert_prefill_rows });
    const baseline = cfg.memory_baseline_bytes orelse return error.MemoryReadingUnavailable;
    try cfg.validate();
    const bill = try Arch.bill(arena.allocator(), io, &.{ .peek = &peek, .cfg = cfg, .routes = cfg,
        .prompt_tokens = options.context_tokens, .max_tokens = 0, .ceiling = options.gpu_ceiling_bytes, .stop = stop });
    defer bill.free(arena.allocator());
    const pinned_decode = cfg.expert_rows orelse cfg.expert_prefill_rows;
    if (pinned_decode) |decode| {
        const prompt = cfg.expert_prefill_rows orelse decode;
        const target = options.gpu_ceiling_bytes - stop;
        if (prompt == 0 or prompt > decode or decode > cfg.model.n_routed_experts or
            bill.total(.prompt, baseline, prompt) > target or bill.total(.decode, baseline, decode) > target)
            return error.OverBudget;
    }
    const device = mlx.mlx_device_new_type(.gpu, 0);
    defer _ = mlx.mlx_device_free(device);
    var available = false;
    try mlx.check(mlx.mlx_device_is_available(&available, device));
    if (!available) return error.NoGpu;
    try mlx.check(mlx.mlx_set_default_device(device));
    m.stream = mlx.mlx_default_gpu_stream_new();
    errdefer _ = mlx.mlx_stream_free(m.stream);
    if (!mlx.streamIsGpu(m.stream)) return error.NoGpu;
    // Loader arrays are lazy; Module.init drops unread tensors before materialize.
    m.weights = try sdk.loader.dir(io, allocator, path, .{ .nocache = true });
    errdefer m.weights.deinit();
    m.module = try Arch.Module.init(allocator, io, cfg, &m.weights, m.stream, options.gpu_ceiling_bytes, stop);
    errdefer m.module.deinit();
    try flush(m);
    if (m.native_loader) try m.module.starkSetMtpEnabled(false);
    // Residency changes only this process's MLX allocator. The original loader
    // preserves its bill-fit reference policy; native decode follows upstream's
    // recommended-maximum residency premise. Allocation admission is unchanged.
    const process_budget = options.gpu_ceiling_bytes -| stop -| baseline;
    const wired_capacity = if (m.native_loader) mlx.maxRecommendedWorkingSet()
        else @min(bill.processBound(m.module.rows), process_budget, mlx.maxRecommendedWorkingSet());
    try installWired(m, @intCast(wired_capacity));
}

pub export fn stark_glm_prefill(model: ?*Model, ids_ptr: ?[*]const u32, count: usize) u32 {
    const m = model orelse return code(.invalid_argument);
    const state = ready(m);
    if (state != .ok) return code(state);
    if (!m.fresh) return reject(m, .invalid_state);
    if (count == 0 or ids_ptr == null) return reject(m, .invalid_argument);
    if (count > m.module.maxContext()) return reject(m, .context_limit);
    const ids = ids_ptr.?[0..count];
    for (ids) |id| if (id >= m.config.model.vocab_size) return reject(m, .invalid_argument);
    if (m.module.mtp != null) if (m.module.prefix_capture_at) |checkpoint|
        if (checkpoint >= count) return reject(m, .invalid_argument);
    const row = m.module.prefill(ids, .{ .prompt_tokens = count, .max_tokens = 0, .host_context = m.module.maxContext() }) catch |e| return fail(m, e);
    checkResult(m, row) catch |e| return fail(m, e);
    m.fresh = false;
    return code(.ok);
}
pub export fn stark_glm_step(model: ?*Model, token: u32) u32 {
    const m = model orelse return code(.invalid_argument);
    const state = ready(m);
    if (state != .ok) return code(state);
    if (m.native_loader and m.native_started) return reject(m, .invalid_state);
    if (m.fresh or m.readout == null) return reject(m, .invalid_state);
    if (token >= m.config.model.vocab_size) return reject(m, .invalid_argument);
    if (m.module.position() >= m.module.maxContext()) return reject(m, .context_limit);
    const row = m.module.step(&.{token}) catch |e| return fail(m, e);
    checkResult(m, row) catch |e| return fail(m, e);
    return code(.ok);
}
pub export fn stark_glm_readout_f32(model: ?*Model, dst_ptr: ?[*]f32, count: usize) u32 {
    const m = model orelse return code(.invalid_argument);
    const state = ready(m);
    if (state != .ok) return code(state);
    if (dst_ptr == null or count != m.config.model.vocab_size) return reject(m, .invalid_argument);
    const row = m.readout orelse return reject(m, .invalid_state);
    mlx.check(mlx.mlx_array_eval(row)) catch |e| return fail(m, e);
    mlx.checkError() catch |e| return fail(m, e);
    const data = mlx.mlx_array_data_float32(row) orelse return fail(m, error.InvalidReadout);
    const values = data[0..count];
    // Validate every value before writing even one caller-owned word.
    for (values) |v| if (!std.math.isFinite(v)) return fail(m, error.NonFiniteReadout);
    @memcpy(dst_ptr.?[0..count], values);
    return code(.ok);
}

pub export fn stark_glm_set_mtp_enabled(model: ?*Model, enabled: u32) u32 {
    const m = model orelse return code(.invalid_argument);
    const state = ready(m);
    if (state != .ok) return code(state);
    if (!m.native_loader) return reject(m, .unsupported);
    if (enabled > 1 or !m.fresh or m.native_started) return reject(m, .invalid_argument);
    if (enabled == 1 and !m.native_head_loaded) return reject(m, .unsupported);
    const on = enabled == 1;
    if (on != m.native_mtp_enabled) {
        // A checkpoint's state-affecting request mode must match its next use.
        m.module.invalidatePrefix();
        m.module.starkSetMtpEnabled(on) catch |e| return fail(m, e);
        m.native_mtp_enabled = on;
    }
    return code(.ok);
}

fn nativeSampling(s: *const NativeSampling) ?sdk.SamplingParams {
    const policy = native_protocol.sampling(s) orelse return null;
    return .{ .temperature = policy.temperature, .top_p = policy.top_p, .seed = policy.seed };
}

fn nativeChoice(m: *Model) !u32 {
    const row = m.readout orelse return error.InvalidReadout;
    // Copy the MLX handle, not the tensor. The sampler may free its input when
    // replacing a sampled row by a one-hot; the raw F32 readout stays owned here.
    var alias = mlx.mlx_array_new();
    mlx.check(mlx.mlx_array_set(&alias, row)) catch |e| {
        _ = mlx.mlx_array_free(alias);
        return e;
    };
    const chosen = m.module.sampler.chosen(alias, m.module.position()) catch |e| {
        _ = mlx.mlx_array_free(alias);
        return e;
    };
    defer _ = mlx.mlx_array_free(chosen);
    const g = &m.module.g;
    const mark = g.mark();
    defer g.resetTo(mark);
    const selected = try g.astype(try g.argmax(chosen, -1), .uint32);
    var token: [1]u32 = undefined;
    _ = try g.hostU32(selected, &token);
    try mlx.checkError();
    if (token[0] >= m.config.model.vocab_size) return error.InvalidReadout;
    return token[0];
}

pub export fn stark_glm_native_initial(model: ?*Model, sampling_ptr: ?*const NativeSampling, out_ptr: ?*u32) u32 {
    const m = model orelse return code(.invalid_argument);
    const state = ready(m);
    if (state != .ok) return code(state);
    if (!m.native_loader) return reject(m, .unsupported);
    const sampling = sampling_ptr orelse return reject(m, .invalid_argument);
    const out = out_ptr orelse return reject(m, .invalid_argument);
    if (sampling.struct_size != @sizeOf(NativeSampling) or sampling.abi_version != abi_version)
        return reject(m, .abi_mismatch);
    if (m.fresh or m.readout == null or m.native_started or m.native_ended) return reject(m, .invalid_state);
    const policy = nativeSampling(sampling) orelse return reject(m, .invalid_argument);
    m.module.sampler.arm(.{ .greedy = policy.greedy(), .clean = true, .sampling = policy });
    m.module.sampler.begin(Arch.draft_lane.laneName(m.module));
    const pending = nativeChoice(m) catch |e| return fail(m, e);
    m.native_sampling = policy;
    m.native_pending = pending;
    m.native_started = true;
    out.* = pending;
    return code(.ok);
}

pub export fn stark_glm_native_round(model: ?*Model, pending: u32, accepted_cap: u32, out_ptr: ?*NativeRound) u32 {
    const m = model orelse return code(.invalid_argument);
    const state = ready(m);
    if (state != .ok) return code(state);
    if (!m.native_loader) return reject(m, .unsupported);
    const out = out_ptr orelse return reject(m, .invalid_argument);
    if (out.struct_size != @sizeOf(NativeRound) or out.abi_version != abi_version) return reject(m, .abi_mismatch);
    if (!m.native_started or m.native_ended or m.native_sampling == null or m.native_pending == null)
        return reject(m, .invalid_state);
    if (pending != m.native_pending.? or pending >= m.config.model.vocab_size or accepted_cap > 3 or
        (!m.native_mtp_enabled and accepted_cap != 0)) return reject(m, .invalid_argument);
    const before = m.module.position();
    if (before >= m.module.maxContext()) return reject(m, .context_limit);
    out.* = .{ .struct_size = @sizeOf(NativeRound), .abi_version = abi_version, .count = 0,
        .accepted = 0, .verified_drafts = 0, .next_pending = 0, .tokens = @splat(0), .position = before };
    clearReadout(m);
    if (m.native_mtp_enabled) {
        const stats0 = Arch.draft_lane.stats(m.module);
        var round = m.module.mtpRound(allocator, pending, accepted_cap, m.native_sampling.?) catch |e| return fail(m, e);
        defer round.deinit(allocator);
        const stats1 = Arch.draft_lane.stats(m.module);
        if (round.tokens.len == 0 or round.tokens.len > 4 or round.tokens[0] != pending or
            round.accepted + 1 != round.tokens.len or stats1.drafted < stats0.drafted or
            stats1.drafted - stats0.drafted > accepted_cap or round.accepted > stats1.drafted - stats0.drafted or
            round.next_token >= m.config.model.vocab_size or m.module.position() != before + round.tokens.len)
            return fail(m, error.InvalidReadout);
        for (round.tokens) |token| if (token >= m.config.model.vocab_size) return fail(m, error.InvalidReadout);
        out.count = @intCast(round.tokens.len);
        out.accepted = round.accepted;
        out.verified_drafts = @intCast(stats1.drafted - stats0.drafted);
        out.next_pending = round.next_token;
        @memcpy(out.tokens[0..round.tokens.len], round.tokens);
    } else {
        // Truly headless execution, including when a preloaded head is parked.
        const row = m.module.step(&.{pending}) catch |e| return fail(m, e);
        checkResult(m, row) catch |e| return fail(m, e);
        out.count = 1;
        out.tokens[0] = pending;
        out.next_pending = nativeChoice(m) catch |e| return fail(m, e);
        if (m.module.position() != before + 1) return fail(m, error.InvalidReadout);
    }
    // The module has already evaluated/read its decision and flushed consumed
    // routes. Avoid the private benchmark's additional whole-stream fence.
    mlx.checkError() catch |e| return fail(m, e);
    out.position = m.module.position();
    if (!native_protocol.validRound(out, pending, accepted_cap, before, m.config.model.vocab_size))
        return fail(m, error.InvalidReadout);
    m.native_pending = out.next_pending;
    return code(.ok);
}

pub export fn stark_glm_native_end_request(model: ?*Model) u32 {
    const m = model orelse return code(.invalid_argument);
    const state = owned(m);
    if (state != .ok) return code(state);
    if (!m.native_loader) return reject(m, .unsupported);
    // End never draws a bonus token or advances the target. Settle all IO and
    // the whole stream before exposing final success to an iterator or Drop.
    var first: ?anyerror = null;
    requestDrain(m) catch |e| { first = e; };
    m.module.starkDiscardTail();
    clearReadout(m);
    m.native_sampling = null;
    m.native_pending = null;
    m.native_started = false;
    m.native_ended = true;
    if (first) |e| return fail(m, e);
    if (m.fenced) return code(.reload_required);
    if (m.cancelled.load(.acquire)) return reject(m, .cancelled);
    if (m.request_invalid) return code(.invalid_state);
    return code(.ok);
}
pub export fn stark_glm_get_position(model: ?*Model, result: ?*u64) u32 {
    const m = model orelse return code(.invalid_argument);
    const dst = result orelse return code(.invalid_argument);
    const state = owned(m);
    if (state != .ok) return code(state);
    dst.* = m.module.position();
    return code(.ok);
}
pub export fn stark_glm_get_info(model: ?*Model, info_ptr: ?*Info) u32 {
    const m = model orelse return code(.invalid_argument);
    const dst = info_ptr orelse return code(.invalid_argument);
    const state = owned(m);
    if (state != .ok) return code(state);
    if (dst.struct_size != @sizeOf(Info) or dst.abi_version != abi_version) return code(.abi_mismatch);
    dst.* = .{ .struct_size = @sizeOf(Info), .abi_version = abi_version, .vocab_size = m.config.model.vocab_size,
        .prefill_chunk_tokens = 16_384, .context_tokens = m.module.maxContext(), .position = m.module.position(),
        // Default builds keep native bits off. An explicit private stage may
        // expose the implementation for worker qualification, even while the
        // admitted head is parked before request-mode selection.
        .capabilities = native_protocol.capabilities(experimental_prefix, m.module.mtp != null,
            m.native_loader, m.native_head_loaded, private_native_caps),
        .fenced = @intFromBool(m.fenced), .reserved = 0 };
    return code(.ok);
}
pub export fn stark_glm_get_rows(model: ?*Model, prompt_ptr: ?*u32, decode_ptr: ?*u32) u32 {
    const m = model orelse return code(.invalid_argument);
    const prompt = prompt_ptr orelse return code(.invalid_argument);
    const decode = decode_ptr orelse return code(.invalid_argument);
    const state = owned(m);
    if (state != .ok) return code(state);
    // Module.rows is assigned from the completed admission plan and used to
    // construct the expert source. Requested config pins are not authoritative.
    prompt.* = m.module.rows.prompt;
    decode.* = m.module.rows.decode;
    return code(.ok);
}
pub export fn stark_glm_get_wired_capacity(model: ?*Model, capacity_ptr: ?*u64, previous_ptr: ?*u64) u32 {
    const m = model orelse return code(.invalid_argument);
    const capacity = capacity_ptr orelse return code(.invalid_argument);
    const previous = previous_ptr orelse return code(.invalid_argument);
    const state = owned(m);
    if (state != .ok) return code(state);
    capacity.* = m.wired_capacity;
    previous.* = m.previous_wired_capacity orelse return code(.invalid_state);
    return code(.ok);
}
pub export fn stark_glm_cancel(model: ?*Model) u32 {
    const m = model orelse return code(.invalid_argument);
    m.cancelled.store(true, .release);
    return code(.ok);
}
pub export fn stark_glm_drain(model: ?*Model) u32 {
    const m = model orelse return code(.invalid_argument);
    const state = owned(m);
    if (state != .ok) return code(state);
    requestDrain(m) catch |e| return fail(m, e);
    return code(.ok);
}
pub export fn stark_glm_reset(model: ?*Model) u32 {
    const m = model orelse return code(.invalid_argument);
    const state = owned(m);
    if (state != .ok) return code(state);
    requestDrain(m) catch |e| return fail(m, e);
    if (m.fenced) return code(.reload_required);
    if (m.cancelled.load(.acquire)) m.module.invalidatePrefix();
    m.module.clearPrefixCapture();
    clearReadout(m);
    if (m.native_loader) m.module.starkDiscardTail() else m.module.requestEnd();
    m.module.cache.restart(&m.module.g);
    // Eager-tail policy guarantees canonical routes, but restore explicitly.
    m.module.prompt_pass = m.module.installed.prompt;
    m.module.step_pass = m.module.installed.step;
    m.module.round_pass = m.module.installed.round;
    m.module.tail = .{};
    m.module.decode = .{};
    m.fresh = true;
    m.native_started = false;
    m.native_ended = false;
    m.native_sampling = null;
    m.native_pending = null;
    m.request_invalid = false;
    m.cancelled.store(false, .release);
    return code(.ok);
}
// CPU-owned snapshots; the capability reports compiled implementation only.
// Qualification must compare fresh/restored full readouts and all state hashes.
pub export fn stark_glm_capture_prefix_at(model: ?*Model, position: u64) u32 {
    const m = model orelse return code(.invalid_argument);
    const state = ready(m);
    if (state != .ok) return code(state);
    if (!experimental_prefix) return reject(m, .unsupported);
    if (!m.fresh or position > std.math.maxInt(u32)) return reject(m, .invalid_argument);
    m.module.capturePrefixAt(@intCast(position)) catch return reject(m, .invalid_argument);
    return code(.ok);
}
pub export fn stark_glm_take_prefix_snapshot(model: ?*Model, out: ?*?*Arch.PrefixSnapshot) u32 {
    const m = model orelse return code(.invalid_argument);
    const dst = out orelse return code(.invalid_argument);
    const state = ready(m);
    if (state != .ok) return code(state);
    if (!experimental_prefix) return code(.unsupported);
    if (dst.* != null) return code(.invalid_argument);
    dst.* = m.module.takePrefixSnapshot();
    return code(.ok);
}
pub export fn stark_glm_restore_prefix(model: ?*Model, snapshot: ?*const Arch.PrefixSnapshot, ids_ptr: ?[*]const u32, count: usize) u32 {
    const m = model orelse return code(.invalid_argument);
    const state = ready(m);
    if (state != .ok) return code(state);
    const s = snapshot orelse return reject(m, .invalid_argument);
    if (!experimental_prefix) return reject(m, .unsupported);
    if (!m.fresh or count == 0 or count > m.module.maxContext() or ids_ptr == null) return reject(m, .invalid_argument);
    if (!m.module.canResumePrefix(s, ids_ptr.?[0..count])) return reject(m, .invalid_argument);
    const row = m.module.resumePrefix(s, ids_ptr.?[0..count], .{ .prompt_tokens = count, .max_tokens = 0, .host_context = m.module.maxContext() }) catch |e| return fail(m, e);
    checkResult(m, row) catch |e| return fail(m, e);
    m.fresh = false;
    return code(.ok);
}
pub export fn stark_glm_prefix_snapshot_free(snapshot_ptr: ?*?*Arch.PrefixSnapshot) u32 {
    const slot = snapshot_ptr orelse return code(.invalid_argument);
    if (slot.*) |s| s.deinit();
    slot.* = null;
    return code(.ok);
}
pub export fn stark_glm_validation_state_sha256(model: ?*Model, dst_ptr: ?[*]u8, count: usize) u32 {
    const m = model orelse return code(.invalid_argument);
    const state = ready(m);
    if (state != .ok) return code(state);
    if (!experimental_prefix) return code(.unsupported);
    if (dst_ptr == null or count != 32) return code(.invalid_argument);
    var digest: [32]u8 = undefined;
    m.module.validationStateFingerprint(&digest) catch |e| return fail(m, e);
    @memcpy(dst_ptr.?[0..32], &digest);
    return code(.ok);
}
pub export fn stark_glm_destroy(model_ptr: ?*?*Model) u32 {
    const slot = model_ptr orelse return code(.invalid_argument);
    const m = slot.* orelse return code(.ok);
    const state = owned(m);
    if (state != .ok) return code(state);
    var status: Status = .ok;
    requestDrain(m) catch |e| { status = statusOf(e); };
    clearReadout(m);
    // Native teardown synchronizes again and joins the source's reader before
    // its owned slot/mmap buffers are released. Keep config/weights alive until then.
    m.module.deinit();
    m.weights.deinit();
    Arch.freeConfig(allocator, m.config);
    _ = mlx.mlx_stream_free(m.stream);
    restoreWired(m) catch { status = .device_failure; };
    m.io_runtime.deinit();
    allocator.destroy(m);
    slot.* = null;
    Arch.releaseProcess();
    process_claimed.store(false, .release);
    return code(status);
}
