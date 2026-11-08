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
    @MainActor static func checkInstalledSpacePresentation() -> Bool {
        let session = BeginnerSetupSession()
        let existing = BeginnerModelChoice(id: "existing", name: "Existing", summary: "Fixture",
                                          setupBytes: 111_000_000_000, installedModelID: "installed-a")
        let fresh = BeginnerModelChoice(id: "fresh", name: "Fresh", summary: "Fixture", setupBytes: 10_000_000_000)
        var snapshot = BeginnerSetupSnapshot()
        snapshot.choices = [existing, fresh]
        let view = BeginnerSetupView(session: session, snapshot: snapshot, actions: BeginnerSetupActions())
        return view.setupSpaceDescription(existing) == nil
            && view.setupSpaceDescription(fresh)?.hasPrefix("Free space needed:") == true
    }

    @MainActor static func checkFilesTaskSelection() -> Bool {
        let session = BeginnerSetupSession()
        session.task = .files
        var received: [(BeginnerTask, String?, String)] = []
        var actions = BeginnerSetupActions()
        actions.refreshChoices = { received.append(($0, $1, $2)) }
        let view = BeginnerSetupView(session: session, snapshot: BeginnerSetupSnapshot(), actions: actions)
        guard BeginnerTask.allCases.contains(.files), BeginnerTask.files.rawValue == "files",
              BeginnerTask.files.title == "Work with files",
              BeginnerTask.files.detail == "Read or edit a folder in OpenWork" else { return false }
        view.requestChoices()
        return received.count == 1 && received[0].0 == .files && received[0].0.rawValue == "files"
    }

    @MainActor static func checkRecommendationPreferences() -> Bool {
        let session = BeginnerSetupSession()
        session.task = .code
        session.modelPriority = .speed
        var received: [(BeginnerTask, String?, String)] = []
        var actions = BeginnerSetupActions()
        actions.refreshChoices = { received.append(($0, $1, $2)) }
        let view = BeginnerSetupView(session: session, snapshot: BeginnerSetupSnapshot(), actions: actions)
        view.requestChoices()
        session.retryPreference = .answers
        view.requestChoices()
        return received.count == 2 && received[0].0 == .code && received[0].2 == "speed"
            && received[1].2 == "answers" && session.stage == .recommendation
    }

    @MainActor static func checkStartPolicy(restart: Bool, accepted: Bool = true) -> Bool {
        let session = BeginnerSetupSession()
        session.stage = .recommendation
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
            return accepted
        }
        let view = BeginnerSetupView(session: session, snapshot: snapshot, actions: actions)
        view.start(choice)
        if restart && !accepted {
            return calls == 1 && session.startRequestedModelID == nil && session.stage == .recommendation
        }
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
        case "navigation-policy":
            let (session, base) = ready()
            session.selectedChoiceID = "recipe-b"
            for target in BeginnerSetupStage.allCases {
                guard session.canNavigate(to: target, snapshot: base) else { return false }
            }
            session.navigate(to: .task, snapshot: base)
            guard session.task == .text, session.selectedChoiceID == "recipe-b" else { return false }
            session.navigate(to: .ready, snapshot: base)
            guard session.stage == .ready else { return false }
            var blocked = base
            blocked.installPhase = .downloading
            guard !session.canNavigate(to: .task, snapshot: blocked) else { return false }
            session.requestStart(modelID: "installed-b")
            guard !session.canNavigate(to: .task, snapshot: base) else { return false }
            let incomplete = BeginnerSetupSession()
            incomplete.task = .code
            guard !incomplete.canNavigate(to: .openWork, snapshot: BeginnerSetupSnapshot())
                && !incomplete.canNavigate(to: .ready, snapshot: BeginnerSetupSnapshot()) else { return false }
            return true
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
        case "completed-setup-survives-generation":
            let (session, base) = ready()
            var snapshot = base
            snapshot.modelReady = false
            snapshot.engineBusy = true
            snapshot.modelGenerating = true
            session.synchronize(with: snapshot)
            guard session.stage == .ready, !session.canFinish(snapshot: snapshot),
                  session.keepsCompletedSetupWhileGenerating(snapshot: snapshot) else { return false }
            session.synchronize(with: base)
            guard session.stage == .ready, session.canFinish(snapshot: base) else { return false }
            session.beginAnotherModel(snapshot: base)
            return session.stage == .retry && session.excludedModelID == base.runningModelID
        case "generation-cannot-hide-invalidated-setup":
            let mutations: [(inout BeginnerSetupSnapshot) -> Void] = [
                { $0.runningModelID = "" }, { $0.runningModelID = "different-model" },
                { $0.runningSupportsTools = false }, { $0.runningContextKnown = false },
                { $0.localAPIAvailable = false }, { $0.openWorkRunning = false },
                { $0.openWorkApproved = false }, { $0.openWorkClientActive = false },
                { $0.openWorkScope = "another-client" }, { $0.modelGenerating = false }
            ]
            for mutate in mutations {
                let (session, base) = ready()
                var snapshot = base
                snapshot.modelReady = false; snapshot.engineBusy = true; snapshot.modelGenerating = true
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
        case "files-task-selection": return BeginnerSetupView.checkFilesTaskSelection()
        case "start-reuses-current": return BeginnerSetupView.checkStartPolicy(restart: false)
        case "recommendation-preferences": return BeginnerSetupView.checkRecommendationPreferences()
        case "start-applies-restart-policy": return BeginnerSetupView.checkStartPolicy(restart: true)
        case "start-rejection-does-not-wait": return BeginnerSetupView.checkStartPolicy(restart: true, accepted: false)
        case "memory-report":
            var raw: [String: Any] = [
                "id": "fixture", "name": "Fixture", "repo": "example/model",
                "revision": String(repeating: "a", count: 40), "artifact_mode": "text", "quantization": "4-bit",
                "download_bytes": 80, "installed_bytes_estimate": 90, "setup_disk_bytes_estimate": 100,
                "runtime_memory_bytes_estimate": 100, "context_tokens": 512, "installed_ids": [String](),
                "license": "test", "qualification": "Synthetic fixture", "caveats": [String]()
            ]
            guard let legacy = BeginnerModelReport.Item(raw), legacy.gpuRuntimeBytes == 100,
                  legacy.memoryDetails.count == 1, legacy.benchmarkDetails.isEmpty else { return false }
            raw["benchmark_details"] = ["Published upstream reference: 42/100"]
            guard BeginnerModelReport.Item(raw)?.benchmarkDetails.count == 1 else { return false }
            for invalid in [true, [1], Array(repeating: "reference", count: 17),
                            [String(repeating: "x", count: 2_001)]] as [Any] {
                raw["benchmark_details"] = invalid
                if BeginnerModelReport.Item(raw) != nil { return false }
            }
            raw.removeValue(forKey: "benchmark_details")
            raw["gpu_memory_bytes_estimate"] = 50
            guard let paged = BeginnerModelReport.Item(raw), paged.gpuRuntimeBytes == 50,
                  paged.memoryDetails.count == 3 else { return false }
            raw["setup_disk_bytes_remaining_estimate"] = 20
            guard BeginnerModelReport.Item(raw)?.setupBytes == 20 else { return false }
            raw["setup_disk_bytes_remaining_estimate"] = 101
            guard BeginnerModelReport.Item(raw) == nil else { return false }
            raw.removeValue(forKey: "setup_disk_bytes_remaining_estimate")
            for invalid in [0, -1, 101, true, 1.5, "50"] as [Any] {
                raw["gpu_memory_bytes_estimate"] = invalid
                if BeginnerModelReport.Item(raw) != nil { return false }
            }
            return true
        case "beginner-presentation":
            let benchmark: [String: Any] = [
                "benchmark": "livecodebench", "score": "91.9", "used_for_ranking": true,
                "model_id": "Qwen/Qwen3.8-Flash-Next",
                "evidence_kind": "upstream_reference", "settings": "TECHNICAL_HARNESS_SETTINGS",
                "source": ["url": "https://example.com/published-score"]
            ]
            let raw: [String: Any] = [
                "id": "fixture", "name": "Fixture", "repo": "example/model",
                "revision": String(repeating: "a", count: 40), "artifact_mode": "text", "quantization": "4-bit",
                "download_bytes": 80, "installed_bytes_estimate": 90, "setup_disk_bytes_estimate": 100,
                "runtime_memory_bytes_estimate": 111 * 1_073_741_824,
                "gpu_memory_bytes_estimate": 82 * 1_073_741_824,
                "context_tokens": 32768, "installed_ids": ["installed-a"], "license": "other",
                "qualification": "TECHNICAL_NATIVE_QUALIFICATION", "caveats": ["TECHNICAL_VALIDATION_CAVEAT"],
                "benchmark_details": ["TECHNICAL_BENCHMARK_DETAILS"],
                "recommendation_reason": "TECHNICAL_CAPACITY_FORMULA", "benchmarks": [benchmark]
            ]
            guard let item = BeginnerModelReport.Item(raw), let reference = item.benchmarkReference,
                  reference.name == "LiveCodeBench", reference.usedForRanking,
                  reference.sourceURL?.absoluteString == "https://huggingface.co/Qwen/Qwen3.8-Flash-Next",
                  item.beginnerDetails.split(separator: "\n").count == 3,
                  item.beginnerDetails.contains("111"), item.beginnerDetails.contains("LiveCodeBench:"),
                  item.beginnerDetails.contains("Published model score"),
                  !item.beginnerDetails.contains("TECHNICAL_"), !item.beginnerDetails.contains("https://"),
                  !item.beginnerDetails.contains("License:"), !item.beginnerDetails.contains("Revision:"),
                  item.beginnerSummary(availableMemoryBytes: nil).contains("Recommended for coding"),
                  !item.beginnerSummary(availableMemoryBytes: nil).contains("TECHNICAL_"),
                  item.beginnerSummary(availableMemoryBytes: 1).contains("Free up memory") else { return false }
            var unscored = raw; unscored.removeValue(forKey: "benchmarks")
            guard let fallback = BeginnerModelReport.Item(unscored), fallback.benchmarkReference == nil,
                  fallback.beginnerDetails.split(separator: "\n").count == 1,
                  fallback.beginnerSummary(availableMemoryBytes: nil).contains("No benchmark result is recorded") else { return false }
            var unranked = benchmark; unranked["used_for_ranking"] = false
            var unrankedRaw = raw; unrankedRaw["benchmarks"] = [unranked]
            guard let unrankedItem = BeginnerModelReport.Item(unrankedRaw),
                  !unrankedItem.beginnerSummary(availableMemoryBytes: nil).contains("Recommended for coding"),
                  unrankedItem.beginnerDetails.contains("Reference only") else { return false }
            for score in ["NaN", "Infinity", "-1", "101", "garbage"] {
                var invalid = benchmark; invalid["score"] = score
                if BeginnerModelReport.BenchmarkReference(invalid) != nil { return false }
            }
            var arbitrarySource = benchmark; arbitrarySource["source"] = ["url": "file:///private/fixture"]
            guard BeginnerModelReport.BenchmarkReference(arbitrarySource)?.sourceURL?.absoluteString
                    == "https://huggingface.co/Qwen/Qwen3.8-Flash-Next" else { return false }
            for model in ["Qwen", "../model", "Qwen/../file", "Qwen/model?redirect=x", "https://example.com/model", "Qwen/model\n"] {
                var invalid = benchmark; invalid["model_id"] = model
                guard let reference = BeginnerModelReport.BenchmarkReference(invalid), reference.sourceURL == nil else { return false }
            }
            return true
        case "installed-space-presentation": return BeginnerSetupView.checkInstalledSpacePresentation()
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
                str(sources / "BeginnerModelReport.swift"),
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

    def test_memory_report_preserves_legacy_and_bounds_gpu_estimates(self):
        self.run_check("memory-report")

    def test_sidebar_navigation_preserves_choices_and_enforces_prerequisites(self):
        self.run_check("navigation-policy")

    def test_retry_preserves_current_model_and_confirmations(self):
        for scenario in ["retry-preserves-current", "retry-before-first-guide"]:
            with self.subTest(scenario=scenario):
                self.run_check(scenario)

    def test_start_waits_for_exact_model_readiness(self):
        self.run_check("start-exact-readiness")

    def test_model_priority_and_retry_preference_reach_recommendations(self):
        self.run_check("recommendation-preferences")

    def test_files_task_is_visible_and_reaches_recommendation_action(self):
        self.run_check("files-task-selection")

    def test_beginner_display_keeps_short_benchmark_evidence_and_omits_technical_metadata(self):
        self.run_check("beginner-presentation")

    def test_installed_models_do_not_display_new_setup_space_requirements(self):
        self.run_check("installed-space-presentation")

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
            "completed-setup-survives-generation",
            "generation-cannot-hide-invalidated-setup",
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
        for scenario in [
            "start-reuses-current",
            "start-applies-restart-policy",
            "start-rejection-does-not-wait",
        ]:
            with self.subTest(scenario=scenario):
                self.run_check(scenario)


if __name__ == "__main__":
    unittest.main()
