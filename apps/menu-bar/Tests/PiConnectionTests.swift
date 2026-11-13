import AppKit
import Foundation

@main
struct PiConnectionTests {
    static func require(_ condition: @autoclosure () -> Bool, line: Int = #line) {
        if !condition() {
            FileHandle.standardError.write(Data("pi_connection_check_failed_\(line)\n".utf8))
            exit(1)
        }
    }

    static func main() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("stark-pi-ui-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("resources/src"), withIntermediateDirectories: true)
        try Data().write(to: directory.appendingPathComponent("resources/src/desktop-setup.mjs"))
        defer { try? FileManager.default.removeItem(at: directory) }
        for file in [".nvm/versions/node/v20.0.0/bin/node", ".nvm/versions/node/v22.19.0/bin/node",
                     ".nvm/versions/node/v26.13.0/bin/node", ".local/bin/pi"] {
            let url = directory.appendingPathComponent(file)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("#!/bin/sh\nexit 0\n".utf8).write(to: url)
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: url.path)
        }
        let detected = PiConnectionController.findTools(home: directory.path)
        require(detected?.node.path == directory.appendingPathComponent(".nvm/versions/node/v26.13.0/bin/node").path)
        require(detected?.pi.path == directory.appendingPathComponent(".local/bin/pi").path)
        let suite = "stark.pi.tests." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let root = directory.appendingPathComponent("profiles")
        let profile = root.appendingPathComponent("profile-fixture").path
        let baseURL = "http://127.0.0.1:12345/v1"
        let key = "synthetic-pi-test-key"
        let tools = PiConnectionController.Tools(node: URL(fileURLWithPath: "/synthetic/node"), pi: URL(fileURLWithPath: "/synthetic/pi"))
        var calls = 0
        var launchArguments: [String] = []
        var deferred: PiConnectionController.Completion?
        let controller = PiConnectionController(resources: directory.appendingPathComponent("resources"), root: root,
            defaults: defaults, tools: tools, runner: { node, args, environment, completion in
                calls += 1
                require(node == tools.node)
                require(!args.contains(where: { $0.contains(key) }))
                require(Set(environment.keys).isSubset(of: ["HOME", "PATH", "STARK_API_KEY"]))
                require(args.first?.hasSuffix("src/desktop-setup.mjs") == true)
                if args.contains("inspect") {
                    require(environment["STARK_API_KEY"] == nil)
                    completion(["ok": true, "version": "1.5.2"])
                } else if args.contains("setup") {
                    require(environment["STARK_API_KEY"] == key)
                    require(args.contains(baseURL) && args.contains("public-model"))
                    deferred = completion
                } else if args.contains("prepare-launch") {
                    launchArguments = args
                    require(environment["STARK_API_KEY"] == nil)
                    require(args.contains(profile))
                    completion(["ok": true, "command": profile + "/Open Pi-fixture.command"])
                } else { fatalError("unexpected_pi_helper_action") }
            })
        controller.refresh()
        require(controller.installed && !controller.busy && calls == 1)
        require(!controller.matches(baseURL: baseURL, modelID: "public-model"))
        controller.setup(baseURL: baseURL, modelID: "public-model", key: key)
        require(controller.busy && calls == 2)
        controller.setup(baseURL: baseURL, modelID: "public-model", key: key)
        require(calls == 2) // The actual caller refuses overlapping installations.
        deferred?(["ok": true, "profile": profile, "modelId": "public-model", "baseUrl": baseURL])
        require(!controller.busy && controller.matches(baseURL: baseURL, modelID: "public-model"))
        require(!controller.matches(baseURL: "http://127.0.0.1:12346/v1", modelID: "public-model"))
        require(!controller.matches(baseURL: baseURL, modelID: "another-model"))
        var copied = false, opened = false
        controller.prepareLaunch(workspace: directory, key: key, openTerminal: { command, completion in
            require(copied)
            require(command.path == profile + "/Open Pi-fixture.command")
            opened = true; completion(true)
        }, copyKey: { supplied in copied = supplied == key; return copied })
        require(copied && opened && calls == 3)

        // The actual Open action launches directly. Folder selection is optional.
        copied = false; opened = false
        controller.open(baseURL: baseURL, modelID: "public-model", key: key,
            openTerminal: { _, completion in require(copied); opened = true; completion(true) },
            copyKey: { supplied in copied = supplied == key; return copied })
        require(copied && opened && calls == 4 && !launchArguments.contains("--workspace"))

        _ = NSApplication.shared
        let window = NSWindow(contentRect: .zero, styleMask: [], backing: .buffered, defer: false)
        defer { window.close() }
        var choose: ((URL?) -> Void)?
        controller.chooseWorkingFolder(in: window, chooser: { parent, current, completion in
            require(parent === window && current == nil); choose = completion
        })
        require(controller.busy && calls == 4)
        choose?(directory)
        require(!controller.busy && controller.workingFolder == directory)
        controller.open(baseURL: baseURL, modelID: "public-model", key: key,
            openTerminal: { _, completion in completion(true) }, copyKey: { _ in true })
        require(launchArguments.contains("--workspace") && launchArguments.contains(directory.path))
        controller.chooseWorkingFolder(in: window, chooser: { _, current, completion in
            require(current == directory); completion(nil)
        })
        require(!controller.busy && controller.workingFolder == directory)
        controller.useDefaultWorkspace()
        require(controller.workingFolder == nil)
        controller.open(baseURL: baseURL, modelID: "public-model", key: key,
            openTerminal: { _, completion in completion(true) }, copyKey: { _ in true })
        require(!launchArguments.contains("--workspace"))

        // Saved setup restoration uses the same verified helper, and an unsafe
        // helper path cannot be saved or opened even if it reports success.
        var restored = false
        let next = PiConnectionController(resources: directory.appendingPathComponent("resources"), root: root,
            defaults: defaults, tools: tools, runner: { _, args, environment, completion in
                require(environment["STARK_API_KEY"] == nil)
                require(args.contains(profile))
                restored = true
                completion(["ok": true, "profile": profile, "modelId": "public-model", "baseUrl": baseURL])
            })
        next.refresh()
        require(restored && next.matches(baseURL: baseURL, modelID: "public-model"))

        var failed = false
        let refused = PiConnectionController(resources: directory.appendingPathComponent("resources"), root: root,
            defaults: UserDefaults(suiteName: suite + ".refused")!, tools: tools, runner: { _, args, _, completion in
                if args.contains("inspect") { completion(["ok": true, "version": "1.5.2"]) }
                else { failed = true; completion(["ok": true, "profile": directory.appendingPathComponent("unrelated").path,
                    "modelId": "public-model", "baseUrl": baseURL]) }
            })
        defer { UserDefaults.standard.removePersistentDomain(forName: suite + ".refused") }
        refused.refresh(); refused.setup(baseURL: baseURL, modelID: "public-model", key: key)
        require(failed && refused.profile == nil)

        var discovered = tools
        var executions: [String] = []
        let recovering = PiConnectionController(resources: directory.appendingPathComponent("resources"), root: root,
            defaults: UserDefaults(suiteName: suite + ".recovery")!, tools: tools, finder: { discovered },
            runner: { node, _, _, completion in
                executions.append(node.path); completion(["ok": true, "version": "1.5.2"])
            })
        defer { UserDefaults.standard.removePersistentDomain(forName: suite + ".recovery") }
        recovering.refresh()
        discovered = .init(node: URL(fileURLWithPath: "/replacement/node"), pi: URL(fileURLWithPath: "/replacement/pi"))
        recovering.refresh(force: true)
        require(executions == [tools.node.path, discovered.node.path] && recovering.installed)

        let pasteboard = NSPasteboard.withUniqueName()
        defer { pasteboard.releaseGlobally() }
        var expirations: [() -> Void] = []
        let lease = ConnectionClipboardLease(pasteboard: pasteboard, schedule: { delay, action in
            require(delay == 120); expirations.append(action)
        })
        require(lease.write(key))
        expirations.removeFirst()()
        require(pasteboard.string(forType: .string) == nil)
        require(lease.write(key))
        require(ConnectionClipboard.write("replacement user copy", to: pasteboard))
        lease.clear()
        require(pasteboard.string(forType: .string) == "replacement user copy")
        require(lease.write(key))
        lease.clear() // Same operation used by normal application termination.
        require(pasteboard.string(forType: .string) == nil)
        require(lease.write("first synthetic copy"))
        let oldExpiration = expirations.last!
        require(lease.write(key))
        oldExpiration()
        require(pasteboard.string(forType: .string) == key)
        lease.clear()
        require(pasteboard.string(forType: .string) == nil)
        print("Pi controller checks passed")
    }
}
