"""Model-free checks of the production beginner session and interface preference.

Uses only synthetic identities and an isolated UserDefaults suite. No backend,
model, network, credential storage, or installed app is started. Compiler and
runtime output stay private; failures report only the named invariant.
"""

from __future__ import annotations

import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VIEW_CHECK = r"""
extension BeginnerSetupView {
    @MainActor static func checkStartPolicy(restart: Bool) -> Bool {
        let session = BeginnerSetupSession()
        let choice = BeginnerModelChoice(id: "recipe-a", name: "Fixture model", summary: "Fixture",
                                         installedModelID: "installed-a")
        var snapshot = BeginnerSetupSnapshot()
        snapshot.choices = [choice]
        snapshot.runningModelID = "installed-a"
        snapshot.modelReady = true
        snapshot.choiceRequiresRestart = restart
        snapshot.canStartModel = true
        var calls = 0
        var actions = BeginnerSetupActions()
        actions.startModel = { received in
            if received.id == choice.id && received.installedModelID == "installed-a" { calls += 1 }
        }
        let view = BeginnerSetupView(session: session, snapshot: snapshot, actions: actions)
        view.start(choice)
        return restart
            ? calls == 1 && session.startRequestedModelID == "installed-a" && session.stage == .installation
            : calls == 0 && session.startRequestedModelID == nil && session.stage == .openWork
    }
}
"""

DRIVER = r"""
import Foundation

@main
struct BeginnerChecks {
    @MainActor static func ready() -> (BeginnerSetupSession, BeginnerSetupSnapshot) {
        let session = BeginnerSetupSession()
        session.task = .text
        var snapshot = BeginnerSetupSnapshot()
        snapshot.choices = [
            BeginnerModelChoice(id: "recipe-a", name: "Current model", summary: "Fixture",
                                installedModelID: "installed-a"),
            BeginnerModelChoice(id: "recipe-b", name: "Another model", summary: "Fixture")
        ]
        snapshot.runningModelID = "installed-a"
        snapshot.runningModelName = "Current model"
        snapshot.modelReady = true
        snapshot.runningSupportsTools = true
        snapshot.runningContextKnown = true
        snapshot.openWorkInstalled = true
        snapshot.openWorkRunning = true
        snapshot.openWorkApproved = true
        snapshot.openWorkClientActive = true
        snapshot.openWorkScope = "connection-a"
        session.synchronize(with: snapshot)
        session.confirmWorkspace()
        session.confirmModelSelection(snapshot: snapshot)
        session.finish(snapshot: snapshot)
        return (session, snapshot)
    }

    @MainActor static func check(_ scenario: String) -> Bool {
        switch scenario {
        case "installation-identity":
            let (session, base) = ready()
            var snapshot = base
            let candidate = snapshot.choices[1]
            snapshot.installChoiceID = "unrelated-recipe"
            snapshot.installPhase = .ready
            snapshot.installedResultID = "installed-b"
            guard session.installedID(for: candidate, in: snapshot) == nil else { return false }
            snapshot.installChoiceID = candidate.id
            snapshot.installPhase = .preparing
            guard session.installedID(for: candidate, in: snapshot) == nil else { return false }
            snapshot.installPhase = .ready
            guard session.installedID(for: candidate, in: snapshot) == "installed-b" else { return false }
            return session.installedID(for: snapshot.choices[0], in: snapshot) == "installed-a"
        case "installation-cancel-fence":
            let (session, base) = ready()
            var snapshot = base
            session.requestInstall(choiceID: "recipe-b")
            snapshot.installChoiceID = "recipe-b"
            snapshot.installPhase = .cancelling
            session.synchronize(with: snapshot)
            guard session.installingChoiceID == "recipe-b" else { return false }
            snapshot.installChoiceID = "unrelated-recipe"
            snapshot.installPhase = .cancelled
            session.synchronize(with: snapshot)
            guard session.installingChoiceID == "recipe-b" else { return false }
            snapshot.installChoiceID = "recipe-b"
            session.synchronize(with: snapshot)
            return session.installingChoiceID == nil && session.stage == .installation
                && snapshot.runningModelID == "installed-a" && snapshot.modelReady
        case "retry-preserves-current":
            let (session, snapshot) = ready()
            let unchanged = snapshot
            session.beginAnotherModel(snapshot: snapshot)
            guard session.stage == .retry, session.choosingAnotherModel,
                  session.excludedModelID == "installed-a",
                  session.choices(in: snapshot).map(\.id) == ["recipe-b"],
                  session.workspaceConfirmed,
                  session.confirmedOpenWorkModelID == "installed-a" else { return false }
            session.task = .code
            session.keepCurrentModel()
            session.synchronize(with: snapshot)
            return session.stage == .ready && session.task == .text && !session.choosingAnotherModel
                && session.canFinish(snapshot: snapshot) && snapshot == unchanged
        case "retry-before-first-guide":
            let (_, snapshot) = ready()
            let session = BeginnerSetupSession()
            session.synchronize(with: snapshot)
            session.beginAnotherModel(snapshot: snapshot)
            return session.stage == .retry && session.task == .unsure
                && session.excludedModelID == "installed-a"
        case "start-exact-readiness":
            let (session, base) = ready()
            var snapshot = base
            session.beginAnotherModel(snapshot: snapshot)
            session.requestStart(modelID: "installed-b")
            session.synchronize(with: snapshot)
            guard session.stage == .installation else { return false }
            snapshot.runningModelID = "unrelated-model"
            session.synchronize(with: snapshot)
            guard session.stage == .installation else { return false }
            snapshot.runningModelID = "installed-b"
            snapshot.modelReady = false
            snapshot.modelStarting = true
            session.synchronize(with: snapshot)
            guard session.stage == .installation else { return false }
            snapshot.modelReady = true
            snapshot.modelStarting = false
            session.synchronize(with: snapshot)
            return session.stage == .connection && session.startRequestedModelID == nil
                && !session.choosingAnotherModel && session.workspaceConfirmed
                && session.confirmedOpenWorkModelID == nil
        case "failed-start-can-retry":
            let (session, base) = ready()
            var snapshot = base
            session.requestStart(modelID: "installed-b")
            snapshot.engineError = "Synthetic startup failure"
            snapshot.modelStarting = false
            session.synchronize(with: snapshot)
            return session.startRequestedModelID == nil && session.stage == .installation
        case "failed-switch-can-choose-again":
            let (session, base) = ready()
            var snapshot = base
            session.beginAnotherModel(snapshot: snapshot)
            session.requestStart(modelID: "installed-b")
            snapshot.runningModelID = ""
            snapshot.modelReady = false
            snapshot.engineError = "Synthetic startup failure"
            session.synchronize(with: snapshot)
            session.chooseAgain()
            return session.stage == .task && session.task != nil && !session.choosingAnotherModel
                && session.excludedModelID == nil && session.selectedChoiceID == nil
                && session.startRequestedModelID == nil && snapshot.runningModelID.isEmpty
                && session.choices(in: snapshot).contains(where: { $0.installedModelID == "installed-a" })
        case "confirmations-and-facts":
            let session = BeginnerSetupSession()
            let (_, snapshot) = ready()
            session.synchronize(with: snapshot)
            guard !session.canFinish(snapshot: snapshot) else { return false }
            session.confirmWorkspace()
            guard !session.canFinish(snapshot: snapshot) else { return false }
            session.confirmModelSelection(snapshot: snapshot)
            var noClient = snapshot
            noClient.openWorkClientActive = false
            guard !session.canFinish(snapshot: noClient) else { return false }
            return session.canFinish(snapshot: snapshot)
        case "scope-invalidation":
            let (session, base) = ready()
            var snapshot = base
            snapshot.openWorkScope = "connection-b"
            session.synchronize(with: snapshot)
            return !session.workspaceConfirmed && session.confirmedOpenWorkModelID == nil
                && session.stage == .openWork && !session.canFinish(snapshot: snapshot)
        case "first-approval-preserves-workspace":
            let session = BeginnerSetupSession()
            let (_, base) = ready()
            var snapshot = base
            snapshot.openWorkApproved = false
            session.synchronize(with: snapshot)
            session.confirmWorkspace()
            snapshot.openWorkApproved = true
            session.synchronize(with: snapshot)
            return session.workspaceConfirmed && session.confirmedOpenWorkModelID == nil
        case "model-confirmation-scoped":
            let (session, base) = ready()
            var snapshot = base
            snapshot.runningModelID = "installed-b"
            session.synchronize(with: snapshot)
            guard session.stage == .connection, session.workspaceConfirmed,
                  session.confirmedOpenWorkModelID == nil,
                  !session.canFinish(snapshot: snapshot) else { return false }
            session.confirmModelSelection(snapshot: snapshot)
            session.finish(snapshot: snapshot)
            return session.stage == .ready && session.canFinish(snapshot: snapshot)
        case "ready-regresses":
            let mutations: [(inout BeginnerSetupSnapshot) -> Void] = [
                { $0.modelReady = false }, { $0.runningSupportsTools = false },
                { $0.runningContextKnown = false }, { $0.openWorkRunning = false },
                { $0.openWorkApproved = false }, { $0.openWorkClientActive = false },
                { $0.localAPIAvailable = false }
            ]
            for mutate in mutations {
                let (session, base) = ready()
                var snapshot = base
                mutate(&snapshot)
                session.synchronize(with: snapshot)
                guard session.stage != .ready, !session.canFinish(snapshot: snapshot) else { return false }
            }
            return true
        case "mode-defaults":
            let suite = "stark.beginner-test." + UUID().uuidString
            let defaults = UserDefaults(suiteName: suite)!
            defer { defaults.removePersistentDomain(forName: suite) }
            let preferences = StarkInterfacePreferences(defaults: defaults)
            guard preferences.mode == .advanced else { return false }
            preferences.resolveInitialMode(hasInstalledModels: false)
            guard preferences.mode == .beginner else { return false }
            preferences.resolveInitialMode(hasInstalledModels: true)
            guard preferences.mode == .beginner else { return false }
            let restored = StarkInterfacePreferences(defaults: defaults)
            guard restored.mode == .beginner else { return false }
            restored.select(.advanced)
            let saved = defaults.persistentDomain(forName: suite) ?? [:]
            return StarkInterfacePreferences(defaults: defaults).mode == .advanced
                && Set(saved.keys) == [StarkInterfacePreferences.key]
        case "existing-library-default":
            let suite = "stark.beginner-test." + UUID().uuidString
            let defaults = UserDefaults(suiteName: suite)!
            defer { defaults.removePersistentDomain(forName: suite) }
            let preferences = StarkInterfacePreferences(defaults: defaults)
            preferences.resolveInitialMode(hasInstalledModels: true)
            return preferences.mode == .advanced
        case "explicit-mode-before-inventory":
            let suite = "stark.beginner-test." + UUID().uuidString
            let defaults = UserDefaults(suiteName: suite)!
            defer { defaults.removePersistentDomain(forName: suite) }
            let preferences = StarkInterfacePreferences(defaults: defaults)
            preferences.select(.beginner)
            preferences.resolveInitialMode(hasInstalledModels: true)
            return preferences.mode == .beginner
        case "start-reuses-current": return BeginnerSetupView.checkStartPolicy(restart: false)
        case "start-applies-restart-policy": return BeginnerSetupView.checkStartPolicy(restart: true)
        default: return false
        }
    }

    @MainActor static func main() {
        exit(check(CommandLine.arguments.dropFirst().first ?? "") ? 0 : 1)
    }
}
"""


@unittest.skipUnless(platform.system() == "Darwin", "macOS native state checks")
class BeginnerSetupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="stark-beginner-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        directory = Path(cls.temporary.name)
        sources = ROOT / "apps/menu-bar/Sources"
        view = directory / "BeginnerSetupView.swift"
        view.write_text((sources / view.name).read_text() + VIEW_CHECK)
        driver = directory / "BeginnerChecks.swift"
        driver.write_text(DRIVER)
        cls.executable = directory / "beginner-checks"
        compiled = subprocess.run(
            [
                "xcrun",
                "swiftc",
                "-parse-as-library",
                "-target",
                "arm64-apple-macosx26.4",
                str(sources / "BeginnerSetupState.swift"),
                str(sources / "StarkInterfacePreferences.swift"),
                str(view),
                str(driver),
                "-o",
                str(cls.executable),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
            check=False,
        )
        if compiled.returncode:
            raise RuntimeError("beginner_state_check_compilation_failed")

    def run_check(self, scenario):
        result = subprocess.run(
            [str(self.executable), scenario],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, f"beginner_{scenario}_check_failed")

    def test_installation_identity_and_cancellation(self):
        for scenario in ["installation-identity", "installation-cancel-fence"]:
            with self.subTest(scenario=scenario):
                self.run_check(scenario)

    def test_retry_preserves_current_model_and_confirmations(self):
        for scenario in ["retry-preserves-current", "retry-before-first-guide"]:
            with self.subTest(scenario=scenario):
                self.run_check(scenario)

    def test_start_waits_for_exact_model_readiness(self):
        self.run_check("start-exact-readiness")

    def test_failed_start_can_be_retried(self):
        for scenario in ["failed-start-can-retry", "failed-switch-can-choose-again"]:
            with self.subTest(scenario=scenario):
                self.run_check(scenario)

    def test_readiness_requires_detected_facts_and_user_confirmations(self):
        self.run_check("confirmations-and-facts")

    def test_confirmation_scopes_and_ready_regression(self):
        for scenario in [
            "scope-invalidation",
            "first-approval-preserves-workspace",
            "model-confirmation-scoped",
            "ready-regresses",
        ]:
            with self.subTest(scenario=scenario):
                self.run_check(scenario)

    def test_isolated_interface_defaults_and_persistence(self):
        for scenario in [
            "mode-defaults",
            "existing-library-default",
            "explicit-mode-before-inventory",
        ]:
            with self.subTest(scenario=scenario):
                self.run_check(scenario)

    def test_start_reuse_and_explicit_restart_policy(self):
        for scenario in ["start-reuses-current", "start-applies-restart-policy"]:
            with self.subTest(scenario=scenario):
                self.run_check(scenario)


if __name__ == "__main__":
    unittest.main()
