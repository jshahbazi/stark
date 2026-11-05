// SPDX-License-Identifier: MIT
// Pure ABI/admission rules. This module never imports MLX or creates a device.
const std = @import("std");
pub const api_version: u32 = 1;
pub const Sampling = extern struct {
    struct_size: u32, abi_version: u32, mode: u32, reserved: u32,
    temperature: f32, top_p: f32, seed: u64,
};
pub const Round = extern struct {
    struct_size: u32, abi_version: u32, count: u32, accepted: u32,
    verified_drafts: u32, next_pending: u32, tokens: [4]u32, position: u64,
};
pub const Policy = struct { temperature: f32, top_p: f32, seed: u64 };

pub const StageConfig = struct { prefix: bool, native_profile: bool };

/// The usable native profile is explicit and experimental. Private candidates
/// retain their separate validation gate; the reference build stays selectable.
pub fn stageConfig(profile: []const u8, prefix: ?bool, private_caps: bool) ?StageConfig {
    const native = std.mem.eql(u8, profile, "bf16-native-experimental");
    if (!native and !std.mem.eql(u8, profile, "headless-reference")) return null;
    if (native and (private_caps or (prefix != null and !prefix.?))) return null;
    return .{ .prefix = prefix orelse native, .native_profile = native };
}

pub fn capabilities(prefix: bool, active_head: bool, native_loader: bool, head_loaded: bool, native_available: bool) u64 {
    var bits: u64 = if (prefix and !active_head) 7 else 3;
    if (native_available and native_loader and head_loaded) {
        bits |= 8;
        if (prefix) bits |= 16;
    }
    return bits;
}

pub fn sampling(s: *const Sampling) ?Policy {
    if (s.reserved != 0 or !std.math.isFinite(s.temperature) or !std.math.isFinite(s.top_p)) return null;
    if (s.mode == 0) {
        if (s.temperature != 0 or s.top_p != 1) return null;
    } else if (s.mode == 1) {
        if (s.temperature < 0.01 or s.top_p <= 0 or s.top_p > 1) return null;
    } else return null;
    return .{ .temperature = s.temperature, .top_p = s.top_p, .seed = s.seed };
}

pub fn validRound(r: *const Round, pending: u32, cap: u32, before: u64, vocab: u32) bool {
    if (r.struct_size != @sizeOf(Round) or r.abi_version != api_version or cap > 3 or
        r.count == 0 or r.count > 4 or r.accepted > 3 or r.count != r.accepted + 1 or
        r.verified_drafts > cap or r.accepted > r.verified_drafts or
        r.tokens[0] != pending or r.next_pending >= vocab or before > std.math.maxInt(u64) - @as(u64, r.count) or
        r.position != before + r.count) return false;
    for (r.tokens[0..r.count]) |id| if (id >= vocab) return false;
    return true;
}

test "native protocol fixed C layout" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Sampling));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(Sampling));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(Sampling, "seed"));
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(Round));
    try std.testing.expectEqual(@as(usize, 8), @alignOf(Round));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(Round, "tokens"));
    try std.testing.expectEqual(@as(usize, 40), @offsetOf(Round, "position"));
}

test "native stage profiles preserve the reference and reject conflicting availability" {
    const reference = stageConfig("headless-reference", null, false).?;
    try std.testing.expect(!reference.prefix and !reference.native_profile);
    try std.testing.expect(stageConfig("headless-reference", true, true) != null);
    const native = stageConfig("bf16-native-experimental", null, false).?;
    try std.testing.expect(native.prefix and native.native_profile);
    try std.testing.expect(stageConfig("bf16-native-experimental", true, false) != null);
    try std.testing.expect(stageConfig("bf16-native-experimental", false, false) == null);
    try std.testing.expect(stageConfig("bf16-native-experimental", true, true) == null);
    try std.testing.expect(stageConfig("unknown", null, false) == null);
}

test "native capabilities stay default off and explicit loaded-head availability survives parking" {
    try std.testing.expectEqual(@as(u64, 7), capabilities(true, false, true, true, false));
    try std.testing.expectEqual(@as(u64, 3), capabilities(true, true, true, true, false));
    try std.testing.expectEqual(@as(u64, 31), capabilities(true, false, true, true, true));
    try std.testing.expectEqual(@as(u64, 27), capabilities(true, true, true, true, true));
    try std.testing.expectEqual(@as(u64, 11), capabilities(false, false, true, true, true));
    try std.testing.expectEqual(@as(u64, 7), capabilities(true, false, false, false, true));
    try std.testing.expectEqual(@as(u64, 7), capabilities(true, false, true, false, true));
}

test "native protocol sampling admits supported finite policies only" {
    var s: Sampling = .{ .struct_size = 32, .abi_version = 1, .mode = 0, .reserved = 0,
        .temperature = 0, .top_p = 1, .seed = 7 };
    try std.testing.expectEqual(@as(u64, 7), sampling(&s).?.seed);
    s.mode = 1; s.temperature = 1; s.top_p = 0.95;
    try std.testing.expect(sampling(&s) != null);
    s.reserved = 1;
    try std.testing.expect(sampling(&s) == null);
    s.reserved = 0; s.temperature = std.math.nan(f32);
    try std.testing.expect(sampling(&s) == null);
    s.temperature = 1; s.top_p = 0;
    try std.testing.expect(sampling(&s) == null);
    s.top_p = 1.01;
    try std.testing.expect(sampling(&s) == null);
    s.top_p = 1; s.mode = 2;
    try std.testing.expect(sampling(&s) == null);
}

test "native protocol committed blocks reject bonus counts and position drift" {
    for (0..4) |accepted| {
        var r: Round = .{ .struct_size = 48, .abi_version = 1, .count = @intCast(accepted + 1),
            .accepted = @intCast(accepted), .verified_drafts = 3, .next_pending = 9,
            .tokens = .{ 1, 2, 3, 4 }, .position = 100 + accepted + 1 };
        try std.testing.expect(validRound(&r, 1, 3, 100, 16));
        r.position += 1;
        try std.testing.expect(!validRound(&r, 1, 3, 100, 16));
        r.position -= 1; r.count += 1;
        try std.testing.expect(!validRound(&r, 1, 3, 100, 16));
    }
    var r: Round = .{ .struct_size = 48, .abi_version = 1, .count = 1, .accepted = 0,
        .verified_drafts = 0, .next_pending = 2, .tokens = .{ 1, 0, 0, 0 }, .position = 101 };
    try std.testing.expect(validRound(&r, 1, 0, 100, 16));
    r.verified_drafts = 1;
    try std.testing.expect(!validRound(&r, 1, 0, 100, 16));
}
