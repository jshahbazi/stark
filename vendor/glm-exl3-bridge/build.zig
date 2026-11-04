// SPDX-License-Identifier: MIT
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{ .default_target = .{
        .os_version_min = .{ .semver = .{ .major = 26, .minor = 2, .patch = 0 } },
    } });
    if (target.result.os.tag != .macos or target.result.cpu.arch != .aarch64)
        @panic("stark-glm-bridge supports macOS arm64 only");
    const optimize = b.standardOptimizeOption(.{});
    const host_dir = b.pathResolve(&.{ b.root.root_dir.path orelse ".", b.option([]const u8, "host-dir", "Pinned mlx-serve source") orelse @panic("pass -Dhost-dir with the pinned ignored source checkout") });
    const plugin_dir = b.pathResolve(&.{ b.root.root_dir.path orelse ".", b.option([]const u8, "plugin-dir", "Matched mlx-stream source with local snapshot patch") orelse @panic("pass -Dplugin-dir with the prepared patched source copy") });
    const stage = b.option([]const u8, "mlx-stage", "Pinned MLX/MLX-C stage") orelse b.pathJoin(&.{ host_dir, "lib/mlx" });
    const sushi_dir = b.option([]const u8, "sushi-dir", "Pinned private sushi source archive") orelse b.pathJoin(&.{ host_dir, "lib/sushi" });
    const host = b.createModule(.{ .root_source_file = b.path("src/host/root.zig"), .target = target, .optimize = optimize, .link_libc = true, .link_libcpp = true });
    const sushi = b.createModule(.{ .root_source_file = .{ .cwd_relative = b.pathJoin(&.{ sushi_dir, "src/exl3/root.zig" }) }, .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "mlx_host", .module = host }} });
    host.addImport("sushi_exl3", sushi);
    const sdk = b.createModule(.{ .root_source_file = .{ .cwd_relative = b.pathJoin(&.{ plugin_dir, "sdk/root.zig" }) }, .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "mlx_host", .module = host }} });
    const plugin = b.createModule(.{ .root_source_file = .{ .cwd_relative = b.pathJoin(&.{ plugin_dir, "src/root.zig" }) }, .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "mlx_host", .module = host }, .{ .name = "sdk", .module = sdk } } });
    const bridge = b.createModule(.{ .root_source_file = b.path("src/bridge.zig"), .target = target, .optimize = optimize, .link_libc = true, .link_libcpp = true, .imports = &.{ .{ .name = "mlx_stream", .module = plugin }, .{ .name = "mlx_host", .module = host }, .{ .name = "sdk", .module = sdk } } });
    const options = b.addOptions();
    options.addOption(bool, "experimental_prefix", b.option(bool, "experimental-prefix", "Compile model-local prefix support; artifact qualification is separate") orelse false);
    options.addOption(bool, "private_native_caps", b.option(bool, "private-native-caps", "Advertise native/head-prefix implementation only in a private qualification stage") orelse false);
    bridge.addOptions("bridge_options", options);
    const csrc = b.pathJoin(&.{ plugin_dir, "csrc" });
    bridge.addCSourceFile(.{ .file = b.path("src/mlx_wired.cpp"), .flags = &.{ "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter", "-Wno-deprecated-copy" } });
    bridge.addCSourceFile(.{ .file = .{ .cwd_relative = b.pathJoin(&.{ csrc, "q3_lookahead4_exl3.c" }) }, .flags = &.{ "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread" } });
    bridge.addCSourceFile(.{ .file = .{ .cwd_relative = b.pathJoin(&.{ csrc, "q3_event_shim.mm" }) }, .flags = &.{ "-O2", "-Wall", "-Wextra", "-Werror", "-fno-objc-arc" } });
    for ([_][]const u8{ "mlx_event_shim.cpp", "mlx_alloc_shim.cpp" }) |name| bridge.addCSourceFile(.{ .file = .{ .cwd_relative = b.pathJoin(&.{ csrc, name }) }, .flags = &.{ "-std=c++20", "-O2", "-D_METAL_", "-DACCELERATE_NEW_LAPACK", "-fno-sanitize=all", "-Wall", "-Wno-unused-parameter", "-Wno-deprecated-declarations" } });
    bridge.addIncludePath(.{ .cwd_relative = csrc });
    bridge.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ stage, "include" }) });
    bridge.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ stage, "include/metal_cpp" }) });
    bridge.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ host_dir, "lib/mlxc-src" }) });
    const libdir = b.pathJoin(&.{ stage, "lib" });
    bridge.addLibraryPath(.{ .cwd_relative = libdir });
    bridge.addRPath(.{ .cwd_relative = libdir });
    bridge.linkSystemLibrary("mlxc", .{ .use_pkg_config = .no });
    bridge.linkSystemLibrary("mlx", .{ .use_pkg_config = .no });
    var exit_code: u8 = 0;
    const sdk_path = std.mem.trim(u8, b.runAllowFail(&.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" }, &exit_code, .inherit) catch @panic("macOS SDK is required"), " \n\r\t");
    const frameworks = b.pathJoin(&.{ sdk_path, "System/Library/Frameworks" });
    bridge.addFrameworkPath(.{ .cwd_relative = frameworks });
    for ([_][]const u8{ "IOKit", "CoreFoundation", "Foundation", "Metal", "IOSurface" }) |name| bridge.linkFramework(name, .{});
    const lib = b.addLibrary(.{ .name = "stark_glm", .linkage = .static, .root_module = bridge });
    b.installArtifact(lib);
    b.installFile("include/stark_glm.h", "include/stark_glm.h");
    const smoke = cExe(b, target, optimize, lib, libdir, frameworks, "stark-glm-abi-smoke", "tests/abi_smoke.c");
    b.installArtifact(smoke);
    const test_step = b.step("test", "Run model-free C ABI admission checks (no model or device calls)");
    test_step.dependOn(&b.addRunArtifact(smoke).step);
    const protocol_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("src/native_protocol.zig"), .target = target, .optimize = optimize,
    }) });
    // This root imports only std, so its test discovery cannot start MLX tests.
    test_step.dependOn(&b.addRunArtifact(protocol_tests).step);
    const bench = cExe(b, target, optimize, lib, libdir, frameworks, "stark-glm-bench", "tools/bench.c");
    b.installArtifact(bench);
    b.step("bench", "Build the standalone benchmark without executing inference").dependOn(&bench.step);
}

fn cExe(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, lib: *std.Build.Step.Compile, libdir: []const u8, frameworks: []const u8, name: []const u8, source: []const u8) *std.Build.Step.Compile {
    const mod = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true, .link_libcpp = true });
    mod.addIncludePath(b.path("include"));
    mod.addCSourceFile(.{ .file = b.path(source), .flags = &.{ "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror" } });
    mod.linkLibrary(lib);
    mod.addLibraryPath(.{ .cwd_relative = libdir });
    mod.addRPath(.{ .cwd_relative = libdir });
    mod.addFrameworkPath(.{ .cwd_relative = frameworks });
    return b.addExecutable(.{ .name = name, .root_module = mod });
}
