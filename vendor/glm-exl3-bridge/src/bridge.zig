// SPDX-License-Identifier: MIT
//! Isolated, serial GLM engine boundary. Model load/forward is never implicit.
const std = @import("std");
const plugin = @import("mlx_stream");
const sdk = @import("sdk");
const mlx = @import("mlx_host").mlx;
const Arch = plugin.glm5_next.arch;
const experimental_prefix = @import("bridge_options").experimental_prefix;
const allocator = std.heap.c_allocator;
const abi_version: u32 = 1;
const max_ceiling: u64 = 112 << 30;
const default_margin: u64 = 2 << 30;

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
comptime {
    if (@sizeOf(Options) != 40 or @sizeOf(Info) != 48) @compileError("C ABI layout changed");
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
        return if (memory) .out_of_memory else .device_failure;
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
    const m = load(path, options.*, margin) catch |e| {
        process_claimed.store(false, .release);
        return code(statusOf(e));
    };
    result.* = m;
    return code(.ok);
}

fn load(path: []const u8, options: Options, margin: u64) !*Model {
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
        .module = undefined, .config = undefined, .weights = undefined, .stream = .{} };
    errdefer m.io_runtime.deinit();
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
    // Explicit prototype policy: serial only, no payload dumps or durable request data.
    cfg.mtp_lane = false;
    cfg.kda_verify_named = false;
    cfg.draft_head_named = false;
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
    const baseline = sdk.memory.boxUsedBytes() orelse return error.MemoryReadingUnavailable;
    cfg.memory_baseline_bytes = baseline;
    // Budget CPU checkpoint + independent restore/copy peak explicitly. No
    // snapshot payload is included in the upstream inference-only bill.
    const snapshot_reserve: u64 = if (experimental_prefix)
        2 * cfg.model.kdaStateBytes() + 2 * @as(u64, cfg.model.nDsa()) * 16 * (2 * cfg.model.index_head_dim) * 2 + options.context_tokens * 4
    else 0;
    const stop = std.math.add(u64, margin, snapshot_reserve) catch return error.OverBudget;
    if (stop >= options.gpu_ceiling_bytes) return error.OverBudget;
    cfg.wired_margin_bytes = stop;
    cfg.effective_context = options.context_tokens;
    // Pack-supplied operator pins must never override bridge admission.
    cfg.expert_rows = if (options.expert_rows > 0) options.expert_rows else null;
    cfg.expert_prefill_rows = if (options.expert_prefill_rows > 0) options.expert_prefill_rows else null;
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
    return m;
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
    const row = m.module.prefill(ids, .{ .prompt_tokens = count, .max_tokens = 0, .host_context = m.module.maxContext() }) catch |e| return fail(m, e);
    checkResult(m, row) catch |e| return fail(m, e);
    m.fresh = false;
    return code(.ok);
}
pub export fn stark_glm_step(model: ?*Model, token: u32) u32 {
    const m = model orelse return code(.invalid_argument);
    const state = ready(m);
    if (state != .ok) return code(state);
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
        .capabilities = 3, .fenced = @intFromBool(m.fenced), .reserved = 0 };
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
pub export fn stark_glm_cancel(model: ?*Model) u32 {
    const m = model orelse return code(.invalid_argument);
    m.cancelled.store(true, .release);
    return code(.ok);
}
pub export fn stark_glm_drain(model: ?*Model) u32 {
    const m = model orelse return code(.invalid_argument);
    const state = owned(m);
    if (state != .ok) return code(state);
    flush(m) catch |e| return fail(m, e);
    return code(.ok);
}
pub export fn stark_glm_reset(model: ?*Model) u32 {
    const m = model orelse return code(.invalid_argument);
    const state = owned(m);
    if (state != .ok) return code(state);
    flush(m) catch |e| return fail(m, e);
    if (m.fenced) return code(.reload_required);
    if (m.cancelled.load(.acquire)) m.module.invalidatePrefix();
    m.module.clearPrefixCapture();
    clearReadout(m);
    m.module.requestEnd();
    m.module.cache.restart(&m.module.g);
    // Eager-tail policy guarantees canonical routes, but restore explicitly.
    m.module.prompt_pass = m.module.installed.prompt;
    m.module.step_pass = m.module.installed.step;
    m.module.round_pass = m.module.installed.round;
    m.module.tail = .{};
    m.module.decode = .{};
    m.fresh = true;
    m.request_invalid = false;
    m.cancelled.store(false, .release);
    return code(.ok);
}
// Experimental ABI: CPU-owned snapshots; no prefix capability is advertised.
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
    flush(m) catch |e| { status = statusOf(e); };
    clearReadout(m);
    // Native teardown synchronizes again and joins the source's reader before
    // its owned slot/mmap buffers are released. Keep config/weights alive until then.
    m.module.deinit();
    m.weights.deinit();
    Arch.freeConfig(allocator, m.config);
    _ = mlx.mlx_stream_free(m.stream);
    m.io_runtime.deinit();
    allocator.destroy(m);
    slot.* = null;
    Arch.releaseProcess();
    process_claimed.store(false, .release);
    return code(status);
}
