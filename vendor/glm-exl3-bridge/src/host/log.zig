// SPDX-License-Identifier: MIT
//! Deliberately silent adapter. The C boundary emits stable status codes only.
pub fn info(comptime _: []const u8, _: anytype) void {}
pub fn warn(comptime _: []const u8, _: anytype) void {}
pub fn err(comptime _: []const u8, _: anytype) void {}
pub fn debug(comptime _: []const u8, _: anytype) void {}
