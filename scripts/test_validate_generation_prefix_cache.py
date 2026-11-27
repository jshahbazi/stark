"""Model-free regression checks for the model onboarding qualification harness."""

import importlib.util
import json
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

_spec = importlib.util.spec_from_file_location(
    "prefix_qualification",
    Path(__file__).with_name("validate_generation_prefix_cache.py"),
)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


HYBRID_KEY = "model-instance+context-tokens+cache-scope+mtp-enabled"


class SyntheticTemplate:
    """Renders a growing synthetic conversation whose earlier renders are prefixes of later ones."""

    def render(self, **arguments):
        return "x" * (100 * len(arguments["messages"]))


class FakeWorker:
    """A model-free worker with the documented boundary-snapshot reuse rule.

    Greedy output depends only on the input; seeded stochastic output also
    depends on the draft count and scheduling selector. The baseline predates
    the selectors and refuses them as an unknown request field. ``faults``
    injects candidate defects for negative controls.
    """

    chunk = 64
    log: ClassVar[list] = []
    faults = frozenset()
    # The candidate's model default (policy, lookup); None is the fixed default.
    default = None
    # Prompt tokens always prefilled after a boundary, and the drafts a request
    # naming no MTP setting runs with. Readiness advertises both unless a fault
    # hides them.
    tail = 1
    default_drafts = 0

    def __init__(self, binary, model, *, prefix_cache=False, progress=False, vision=False):
        self.binary = binary
        self.prefix_cache = prefix_cache
        self.vision = vision
        self.entry = None
        mtp = {"supported": True, "max_drafts": 3, "recommended_drafts": 2}
        if self.default_drafts and "default_drafts_not_advertised" not in self.faults:
            mtp["default_drafts"] = self.default_drafts
        if binary == "candidate":
            mtp.update(policies=["fixed", "adaptive"], default_policy="fixed", lookup=True)
            if self.default is not None:
                mtp.update(default_policy=self.default[0], default_lookup=self.default[1])
        self.ready = {"ready": True, "mtp": mtp, "prefix_cache": {
            "enabled": prefix_cache, "chunk_tokens": self.chunk, "key": HYBRID_KEY}}
        if self.tail != 1 and "tail_not_advertised" not in self.faults:
            self.ready["prefix_cache"]["tail_tokens"] = self.tail

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def tokenize(self, text):
        return [index % 50 + 1 for index in range(len(text))]

    def request(self, payload, *, cancel_after_first_progress=False):
        operation = payload["operation"]
        if operation in ("reset_cache", "discard_image"):
            self.entry = None
            return {"_progress": []}
        if operation == "decode":
            return {"text": "synthetic", "_progress": []}
        self.log.append((self.binary, self.prefix_cache, payload))
        selector = {k: payload[k] for k in ("mtp_policy", "mtp_lookup") if k in payload}
        if selector and self.binary != "candidate":
            return {"error": "invalid_request", "_progress": []}
        ids, scope = payload["input_token_ids"], payload.get("cache_scope")
        drafts = payload.get("mtp_drafts")
        if drafts is None:
            drafts = self.default_drafts
        if cancel_after_first_progress or len(ids) + payload["max_new_tokens"] > payload["context_tokens"]:
            self.entry = None
            return {"error": "generation_cancelled" if cancel_after_first_progress else "context_overflow",
                    "_progress": []}
        key = (scope, payload["context_tokens"], drafts > 0)
        if "selector_in_key" in self.faults:
            key += (tuple(sorted(selector.items())),)
        reused = 0
        if self.prefix_cache and scope is not None and self.entry is not None and self.entry[0] == key:
            prefix = self.entry[1]
            if len(ids) >= len(prefix) + max(2, self.tail) and ids[: len(prefix)] == prefix:
                reused = len(prefix)
        boundary = max(len(ids) - self.tail, 0) // self.chunk * self.chunk
        self.entry = (key, ids[:boundary]) if self.prefix_cache and scope is not None and boundary else None
        sampling = payload["sampling"]
        default = self.default if self.binary == "candidate" else None
        if "default_not_applied" in self.faults:
            default = None
        policy, lookup = default or ("fixed", False)
        if "optout_ignored" not in self.faults or selector != {"mtp_policy": "fixed", "mtp_lookup": False}:
            policy, lookup = payload.get("mtp_policy", policy), payload.get("mtp_lookup", lookup)
        if "ignore_selector" in self.faults:
            policy, lookup = "fixed", False
        weight = (policy == "adaptive") + int(lookup) if drafts else 0
        generated = [sum(ids) % 97, len(ids) % 89]
        if sampling["mode"] != "greedy":
            generated.append(payload["seed"] * 7 + drafts + 3 * weight)
        if "selector_hit_drift" in self.faults and selector and reused:
            generated.append(5)
        reply = {"generated_token_ids": generated, "text": str(generated), "finish_reason": "length",
                 "reused_input_tokens": reused, "_progress": []}
        if drafts:
            stats = {"rounds": 2, "proposed": 3, "accepted": 1, "verified_rows": 4, "draft_seconds": 0.5}
            reply["mtp"] = {"enabled": True, "drafts": drafts, "stats": stats}
            if (policy, lookup) != ("fixed", False):
                reply["mtp"].update(policy=policy, lookup=lookup)
                stats["schedule"] = {"policy": policy, "lookup": lookup, "limit": drafts,
                                     "head_rounds_by_depth": [0, 1, 1][: drafts + 1],
                                     "head_round_seconds_by_depth": [0.0, 0.1, 0.2][: drafts + 1],
                                     "lookup_rounds": int(lookup), "lookup_proposed": 2 * int(lookup),
                                     "lookup_accepted": int(lookup), "lookup_round_seconds": 0.1,
                                     "head_advances": 0, "head_advance_seconds": 0.0}
                if policy == "adaptive":
                    stats["schedule"]["cost_table"] = "synthetic-table"
        return reply


def selector_args(**overrides):
    values = {"candidate": "candidate", "baseline": "baseline", "model": "model", "progress": False,
              "vision": False, "seed": 11, "first_tokens": 420, "tools": 1, "instruction_paragraphs": 1,
              "follow_up_sentences": (4, 4, 8), "context": 2000, "max_new": 4, "reuse": "boundary",
              "mtp_drafts": 2, "mtp_policy": "adaptive", "mtp_lookup": True}
    values.update(overrides)
    return SimpleNamespace(**values)


class PrefixQualificationTests(unittest.TestCase):
    def run_synthetic(self, faults=frozenset(), default=None, **overrides):
        FakeWorker.log = []
        with (patch.object(harness, "Worker", FakeWorker), patch.object(FakeWorker, "faults", faults),
              patch.object(FakeWorker, "default", default)):
            return harness.run_exactness(selector_args(**overrides), SyntheticTemplate(), {}, None)

    def test_readiness_default_schedule_is_none_for_fixed_and_labels_opt_out(self):
        self.assertIsNone(harness.default_schedule({}))
        self.assertIsNone(harness.default_schedule({"mtp": {"default_policy": "fixed"}}))
        self.assertIsNone(harness.default_schedule({"mtp": {"default_policy": "fixed", "default_lookup": False}}))
        ready = {"mtp": {"default_policy": "adaptive", "default_lookup": True}}
        self.assertEqual(harness.default_schedule(ready), ("adaptive", True))
        self.assertEqual(harness.default_schedule({"mtp": {"default_lookup": True}}), ("fixed", True))
        self.assertEqual(harness.selector_label(harness.OPT_OUT), "fixed_nolookup")
        self.assertEqual(harness.effective_schedule(harness.OPT_OUT, ("adaptive", True)), ("fixed", False))
        self.assertEqual(harness.effective_schedule({}, ("adaptive", True)), ("adaptive", True))
        self.assertEqual(harness.effective_schedule({"mtp_lookup": False}, ("adaptive", True)), ("adaptive", False))

    def test_fixed_acknowledgement_must_be_plain(self):
        plain = {"mtp": {"enabled": True, "drafts": 2, "stats": {"rounds": 2}}}
        self.assertTrue(harness.schedule_record(plain, harness.OPT_OUT, ("adaptive", True))["acknowledged"])
        self.assertFalse(harness.schedule_record(plain, {}, ("adaptive", True))["acknowledged"])
        named = {"mtp": {"enabled": True, "policy": "adaptive", "lookup": True, "stats": {
            "rounds": 2, "schedule": {"policy": "adaptive", "lookup": True, "cost_table": "synthetic"}}}}
        self.assertTrue(harness.schedule_record(named, {}, ("adaptive", True))["acknowledged"])
        self.assertFalse(harness.schedule_record(named, harness.OPT_OUT, ("adaptive", True))["acknowledged"])
        self.assertNotIn("schedule_cost_table", harness.schedule_record(named, {}, ("adaptive", True))["counts"])

    def test_non_fixed_default_plan_adds_opt_out_cases_only_then(self):
        turns = [[1, 2] * 550] * 3 + [[1, 2] * 600]
        default = harness.plan(turns, 1024, 4096, 2)
        self.assertEqual(default, harness.plan(turns, 1024, 4096, 2, default_schedule=None))
        steps = harness.plan(turns, 1024, 4096, 2, default_schedule=("adaptive", True))
        labels = [label for _, label, _, _ in steps]
        added = [label for label in labels if label not in {row[1] for row in default}]
        self.assertEqual(added, [
            "optout_after_plain", "optout_warm", "optout_repeat_stochastic", "default_after_optout",
            "optout_after_default", "plain_after_optout", "optout_stochastic_after_plain",
            "optout_stochastic_warm", "plain_after_optout_stochastic", "plain_after_optout_warm"])
        cases = {label: options for kind, label, _, options in steps if kind == "case"}
        for label in ("optout_after_plain", "optout_warm", "optout_repeat_stochastic", "optout_stochastic_warm"):
            self.assertEqual(cases[label]["selector"], harness.OPT_OUT)
        self.assertNotIn("selector", cases["default_after_optout"])
        self.assertEqual(cases["optout_stochastic_after_plain"]["sampling"]["mode"], "stochastic")
        # The selector block's fixed step becomes the explicit opt-out under a non-fixed default.
        with_selectors = harness.plan(turns, 1024, 4096, 2, default_schedule=("adaptive", True),
                                      selectors=harness.selector_combinations("adaptive", True))
        cases = {label: options for kind, label, _, options in with_selectors if kind == "case"}
        self.assertEqual(cases["fixed_after_adaptive_lookup"]["selector"], harness.OPT_OUT)
        legacy = harness.plan(turns, 1024, 4096, 2, selectors=harness.selector_combinations("adaptive", True))
        self.assertNotIn("selector", {label: o for k, label, _, o in legacy if k == "case"}["fixed_after_adaptive"])

    def test_adaptive_default_run_passes_with_and_without_selector_flags(self):
        for flags in ({"mtp_policy": None, "mtp_lookup": False}, {}):
            with self.subTest(flags=flags):
                result = self.run_synthetic(default=("adaptive", True), **flags)
                self.assertTrue(harness.exactness_passed(result))
                self.assertEqual(result["mtp_default_schedule"], {"policy": "adaptive", "lookup": True})
                rows = {check["case"]: check for check in result["checks"]}
                for label in ("mtp_after_plain", "mtp_warm", "default_after_optout"):
                    self.assertEqual(rows[label]["mtp_selector"], "default")
                    self.assertTrue(rows[label]["identical_to_baseline_worker"])
                    self.assertTrue(rows[label]["selector_acknowledged"])
                # A stochastic default case is not compared with the fixed-schedule baseline.
                self.assertIsNone(rows["mtp_repeat_stochastic"]["identical_to_baseline_worker"])
                self.assertTrue(rows["mtp_repeat_stochastic"]["schedule_identical_to_fresh"])
                for label in ("optout_after_plain", "optout_warm", "optout_repeat_stochastic",
                              "optout_stochastic_after_plain", "optout_stochastic_warm"):
                    self.assertTrue(rows[label]["expects_fixed_schedule"])
                    self.assertTrue(rows[label]["identical_to_baseline_worker"])
                self.assertEqual(rows["optout_after_plain"]["reused_input_tokens"], 0)
                self.assertEqual(rows["optout_stochastic_after_plain"]["reused_input_tokens"], 0)
                self.assertGreater(rows["optout_warm"]["reused_input_tokens"], 0)
                self.assertGreater(rows["optout_stochastic_warm"]["reused_input_tokens"], 0)
                self.assertGreater(rows["default_after_optout"]["reused_input_tokens"], 0)
                self.assertNotIn("mtp_selector", rows["plain_warm"])
                baseline = [p for binary, _, p in FakeWorker.log if binary == "baseline"]
                self.assertEqual(sum("mtp_policy" in p or "mtp_lookup" in p for p in baseline),
                                 len(result.get("mtp_selectors", {}).get("combinations", [])))

    def test_default_schedule_defects_fail_qualification(self):
        for fault, field in (("default_not_applied", "selector_acknowledged"),
                             ("optout_ignored", "identical_to_baseline_worker")):
            with self.subTest(fault=fault):
                result = self.run_synthetic(faults=frozenset({fault}), default=("adaptive", True),
                                            mtp_policy=None, mtp_lookup=False)
                self.assertFalse(harness.exactness_passed(result))
                self.assertTrue([check for check in result["checks"] if check.get(field) is False])

    def test_stochastic_default_row_compared_with_baseline_cannot_pass(self):
        row = {"successful": True, "identical_to_fresh": True, "reuse_matches_rule": True,
               "mtp_selector": "default", "greedy": False, "selector_acknowledged": True,
               "schedule_identical_to_fresh": True, "identical_to_fixed_schedule": None,
               "identical_to_baseline_worker": None}
        self.assertTrue(harness.check_passed(row))
        for value in (True, False):
            self.assertFalse(harness.check_passed({**row, "identical_to_baseline_worker": value}))
        opt_out = {**row, "mtp_selector": "fixed_nolookup", "expects_fixed_schedule": True}
        self.assertFalse(harness.check_passed(opt_out))
        self.assertTrue(harness.check_passed({**opt_out, "identical_to_baseline_worker": True}))
        self.assertFalse(harness.check_passed({**opt_out, "identical_to_baseline_worker": False}))

    def test_selector_combinations_are_opt_in_and_labelled(self):
        self.assertEqual(harness.selector_combinations(), [])
        self.assertEqual(harness.selector_combinations("fixed", False), [])
        self.assertEqual(harness.selector_combinations("adaptive"), [{"mtp_policy": "adaptive"}])
        self.assertEqual(harness.selector_combinations(None, True), [{"mtp_lookup": True}])
        both = harness.selector_combinations("adaptive", True)
        self.assertEqual([harness.selector_label(s) for s in both], ["adaptive", "lookup", "adaptive_lookup"])
        self.assertEqual(both[-1], {"mtp_policy": "adaptive", "mtp_lookup": True})

    def test_default_plan_is_unchanged_without_selectors(self):
        turns = [[1, 2] * 550] * 4
        default = harness.plan(turns, 1024, 4096, 2)
        self.assertEqual(default, harness.plan(turns, 1024, 4096, 2, selectors=()))
        self.assertFalse(any("selector" in options for kind, _, _, options in default if kind == "case"))
        self.assertFalse(any("adaptive" in label or "lookup" in label for _, label, _, _ in default))
        # Without MTP no selector case can be planned.
        self.assertEqual(harness.plan(turns, 1024, 4096, None, selectors=[{"mtp_lookup": True}]),
                         harness.plan(turns, 1024, 4096, None))

    def test_selector_plan_covers_plain_warm_stochastic_switch_and_plain_after(self):
        turns = [[1, 2] * 550] * 3 + [[1, 2] * 600]
        steps = harness.plan(turns, 1024, 4096, 2, selectors=[{"mtp_policy": "adaptive"}])
        default = harness.plan(turns, 1024, 4096, 2)
        labels = [label for _, label, _, _ in steps]
        added = [label for label in labels if label not in {row[1] for row in default}]
        self.assertEqual(added, [
            "adaptive_after_plain", "adaptive_warm", "adaptive_repeat_stochastic", "fixed_after_adaptive",
            "adaptive_after_fixed", "adaptive_shorter", "adaptive_regrow", "adaptive_regrow_stochastic",
            "plain_after_adaptive", "plain_after_adaptive_warm"])
        self.assertEqual(labels.index("adaptive_after_plain"), labels.index("plain_warm") + 1)
        cases = {label: (ids, options) for kind, label, ids, options in steps if kind == "case"}
        selector = {"mtp_policy": "adaptive"}
        for label in ("adaptive_after_plain", "adaptive_warm", "adaptive_after_fixed", "adaptive_regrow"):
            self.assertEqual(cases[label][1]["selector"], selector)
            self.assertEqual(cases[label][1]["mtp_drafts"], 2)
            self.assertTrue(harness.is_greedy(cases[label][1]))
        self.assertEqual(cases["adaptive_repeat_stochastic"][1]["sampling"]["mode"], "stochastic")
        self.assertNotIn("selector", cases["fixed_after_adaptive"][1])
        self.assertEqual(cases["fixed_after_adaptive"][1]["mtp_drafts"], 2)
        self.assertNotIn("mtp_drafts", cases["plain_after_adaptive"][1])
        self.assertEqual(cases["adaptive_shorter"][0], turns[-2])
        # The fixed-schedule counterpart of a greedy selector case is the existing warm MTP request.
        self.assertEqual(harness.comparison_key(turns[-1], harness.fixed_schedule(cases["adaptive_warm"][1])),
                         harness.comparison_key(turns[-1], cases["mtp_warm"][1]))
        self.assertEqual(harness.case_request(cases["adaptive_warm"][1], 8)["mtp_policy"], "adaptive")
        self.assertNotIn("mtp_policy", harness.case_request(cases["mtp_warm"][1], 8))

    def test_selector_readiness_requires_mtp_and_advertised_selectors(self):
        ready = {"mtp": {"supported": True, "max_drafts": 3, "policies": ["fixed", "adaptive"], "lookup": True}}
        both = harness.selector_combinations("adaptive", True)
        harness.selector_readiness(ready, 2, both)
        harness.selector_readiness({}, None, [])
        with self.assertRaisesRegex(SystemExit, "positive MTP draft count"):
            harness.selector_readiness(ready, 0, both)
        with self.assertRaisesRegex(SystemExit, "adaptive MTP policy"):
            harness.selector_readiness({"mtp": {"lookup": True}}, 2, both)
        with self.assertRaisesRegex(SystemExit, "prompt lookup"):
            harness.selector_readiness({"mtp": {"policies": ["fixed", "adaptive"]}}, 2, both)

    def test_schedule_record_keeps_only_integer_counts_and_checks_acknowledgement(self):
        reply = {"mtp": {"enabled": True, "policy": "adaptive", "lookup": True, "stats": {
            "rounds": 3, "draft_seconds": 0.5, "schedule": {
                "policy": "adaptive", "lookup": True, "limit": 2, "head_rounds_by_depth": [1, 0, 2],
                "head_round_seconds_by_depth": [0.1, 0.0, 0.2], "lookup_rounds": 1}}}}
        record = harness.schedule_record(reply, {"mtp_policy": "adaptive", "mtp_lookup": True})
        self.assertTrue(record["acknowledged"])
        self.assertEqual(record["counts"], {"rounds": 3, "schedule_limit": 2,
                                            "schedule_head_rounds_by_depth": [1, 0, 2],
                                            "schedule_lookup_rounds": 1})
        self.assertFalse(harness.schedule_record(reply, {"mtp_policy": "adaptive"})["acknowledged"])
        self.assertEqual(harness.schedule_record({"error": "invalid_request"}, {"mtp_lookup": True}),
                         {"acknowledged": False, "counts": None})

    def test_synthetic_selector_run_passes_and_never_sends_selector_cases_to_baseline(self):
        result = self.run_synthetic()
        self.assertTrue(harness.exactness_passed(result))
        self.assertEqual(result["mtp_selectors"]["combinations"], ["adaptive", "lookup", "adaptive_lookup"])
        self.assertEqual(result["mtp_selectors"]["baseline_refusals"],
                         {"adaptive": "invalid_request", "lookup": "invalid_request",
                          "adaptive_lookup": "invalid_request"})
        baseline_selector_requests = [payload for binary, _, payload in FakeWorker.log
                                      if binary == "baseline" and ("mtp_policy" in payload or "mtp_lookup" in payload)]
        # Only the three refusal probes, each one output token.
        self.assertEqual([payload["max_new_tokens"] for payload in baseline_selector_requests], [1, 1, 1])
        rows = {check["case"]: check for check in result["checks"]}
        for name in ("adaptive", "lookup", "adaptive_lookup"):
            self.assertEqual(rows[f"{name}_after_plain"]["reused_input_tokens"], 0)
            for label in ("warm", "repeat_stochastic", "after_fixed", "regrow", "regrow_stochastic"):
                self.assertGreater(rows[f"{name}_{label}"]["reused_input_tokens"], 0)
            self.assertGreater(rows[f"fixed_after_{name}"]["reused_input_tokens"], 0)
            self.assertEqual(rows[f"plain_after_{name}"]["reused_input_tokens"], 0)
            greedy, stochastic = rows[f"{name}_warm"], rows[f"{name}_repeat_stochastic"]
            self.assertTrue(greedy["identical_to_fixed_schedule"] and greedy["identical_to_baseline_worker"])
            self.assertIsNone(stochastic["identical_to_baseline_worker"])
            self.assertFalse(stochastic["identical_to_fixed_schedule"])
            self.assertTrue(stochastic["identical_to_fresh"] and stochastic["schedule_identical_to_fresh"])
            self.assertNotIn("draft_seconds", greedy["schedule_counts"])
        self.assertNotIn("mtp_selector", rows["mtp_warm"])

    def test_default_synthetic_run_sends_no_selector_and_keeps_its_schema(self):
        result = self.run_synthetic(mtp_policy=None, mtp_lookup=False)
        self.assertTrue(harness.exactness_passed(result))
        self.assertNotIn("mtp_selectors", result)
        self.assertFalse(any("mtp_policy" in payload or "mtp_lookup" in payload for _, _, payload in FakeWorker.log))
        self.assertEqual({frozenset(check) for check in result["checks"]}, {frozenset({
            "case", "input_tokens", "reused_input_tokens", "reuse_matches_rule", "identical_to_fresh",
            "identical_to_baseline_worker", "output_sha256", "tool_calls", "finish_reason", "output_tokens",
            "successful"})})

    def test_selector_defects_fail_qualification(self):
        for fault, field in (("selector_hit_drift", "identical_to_fresh"),
                             ("selector_in_key", "reuse_matches_rule"),
                             ("ignore_selector", "selector_acknowledged")):
            with self.subTest(fault=fault):
                result = self.run_synthetic(faults=frozenset({fault}), mtp_lookup=False)
                self.assertFalse(harness.exactness_passed(result))
                failing = [check for check in result["checks"] if not check.get(field, True)]
                self.assertTrue(failing)
                self.assertTrue(all("mtp_selector" in check or "adaptive" in check["case"] for check in failing))

    def test_selector_pass_rule_requires_greedy_independence_and_no_stochastic_baseline(self):
        greedy = {"successful": True, "identical_to_fresh": True, "reuse_matches_rule": True,
                  "mtp_selector": "adaptive", "greedy": True, "selector_acknowledged": True,
                  "schedule_identical_to_fresh": True, "identical_to_fixed_schedule": True,
                  "identical_to_baseline_worker": True}
        self.assertTrue(harness.check_passed(greedy))
        for field in ("identical_to_fixed_schedule", "identical_to_baseline_worker"):
            self.assertFalse(harness.check_passed({**greedy, field: None}))
            self.assertFalse(harness.check_passed({**greedy, field: False}))
        stochastic = {**greedy, "greedy": False, "identical_to_fixed_schedule": False,
                      "identical_to_baseline_worker": None}
        self.assertTrue(harness.check_passed(stochastic))
        self.assertFalse(harness.check_passed({**stochastic, "identical_to_baseline_worker": True}))
        self.assertFalse(harness.check_passed({**stochastic, "schedule_identical_to_fresh": False}))
        self.assertFalse(harness.check_passed({**stochastic, "identical_to_fresh": False}))

    def test_vision_worker_staging_is_private_empty_and_removed_on_close(self):
        process = MagicMock()
        with (patch.object(harness.subprocess, "Popen", return_value=process) as spawn,
              patch.object(harness.Worker, "read", return_value={"ready": True})):
            with harness.Worker("worker", "model", prefix_cache=True, vision=True) as worker:
                root = worker.staging_root
                self.assertTrue(root.is_absolute())
                self.assertEqual(root, root.resolve())
                self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
                self.assertEqual(list(root.iterdir()), [])
                self.assertEqual(spawn.call_args.args[0], ["worker", "--model-path", "model", "--prefix-cache",
                                                          "--vision-staging-root", str(root)])
            self.assertFalse(root.exists())

    def test_vision_staging_is_removed_when_spawn_or_readiness_fails(self):
        for startup_failure in (True, False):
            roots = []

            def spawn(command, roots=roots, startup_failure=startup_failure, **_kwargs):
                roots.append(Path(command[command.index("--vision-staging-root") + 1]))
                if startup_failure:
                    raise OSError("synthetic spawn failure")
                return MagicMock()

            with (self.subTest(spawn_failure=startup_failure),
                  patch.object(harness.subprocess, "Popen", side_effect=spawn),
                  patch.object(harness.Worker, "read", return_value={"ready": False, "error": "synthetic"}),
                  self.assertRaises((OSError, SystemExit))):
                harness.Worker("worker", "model", vision=True)
            self.assertEqual(len(roots), 1)
            self.assertFalse(roots[0].exists())

    def test_vision_image_invalidation_completes_private_prepare_and_generation(self):
        process = MagicMock()
        prepared_id = "b" * 32
        requests = []
        template = MagicMock()
        template.render.return_value = "private synthetic image prompt"

        def request(payload):
            requests.append(payload)
            if payload["operation"] == "prepare_image":
                path = worker.staging_root / payload["image"]["id"]
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                self.assertEqual(path.read_bytes(), harness.synthetic_png())
                self.assertEqual(path.stat().st_size, payload["image"]["byte_length"])
                return {"prepared_id": prepared_id}
            self.assertEqual(payload["operation"], "generate_image")
            self.assertEqual(payload["prepared_id"], prepared_id)
            return {"output_tokens": 1}

        with (patch.object(harness.subprocess, "Popen", return_value=process),
              patch.object(harness.Worker, "read", return_value={"ready": True}),
              patch.object(harness, "load_template", return_value=({}, template)),
              harness.Worker("worker", "model", vision=True) as worker,
              patch.object(worker, "request", side_effect=request)):
            result, keeps = harness.act(worker, "image_operation", [1, 2], 4096, "scope")
            self.assertEqual(result, {"output_tokens": 1})
            self.assertFalse(keeps)
            self.assertEqual(list(worker.staging_root.iterdir()), [])
            self.assertEqual([row["context_tokens"] for row in requests], [4096, 4096])

    def test_failed_image_preparation_does_not_generate_and_cleans_fixture(self):
        template = MagicMock()
        template.render.return_value = "private prompt"
        with (patch.object(harness.subprocess, "Popen", return_value=MagicMock()),
              patch.object(harness.Worker, "read", return_value={"ready": True}),
              patch.object(harness, "load_template", return_value=({}, template)),
              harness.Worker("worker", "model", vision=True) as worker,
              patch.object(worker, "request", return_value={"error": "invalid_image_request"}) as request):
            self.assertEqual(worker.image_transaction(4096), {"error": "invalid_image_request"})
            request.assert_called_once()
            self.assertEqual(list(worker.staging_root.iterdir()), [])

    def test_exact_draft_selection_and_seeded_mtp_are_qualified(self):
        dense = SimpleNamespace(ready={"prefix_cache": {"key": "model-instance+mtp-selection"}})
        hybrid = SimpleNamespace(ready={"prefix_cache": {"key": "model-instance+mtp-enabled"}})
        self.assertNotEqual(harness.mtp_cache_value(dense, 0), harness.mtp_cache_value(dense, 7))
        self.assertEqual(harness.mtp_cache_value(hybrid, 1), harness.mtp_cache_value(hybrid, 3))
        steps = harness.plan([[1, 2] * 550] * 4, 1024, 4096, 7, mtp_selection_key=True)
        cases = {label: options for kind, label, _, options in steps if kind == "case"}
        self.assertEqual(cases["mtp_repeat_stochastic"]["sampling"]["mode"], "stochastic")
        self.assertEqual(cases["mtp_repeat_stochastic"]["mtp_drafts"], 7)
        self.assertEqual(cases["mtp_different_depth"]["mtp_drafts"], 1)
        self.assertEqual(cases["mtp_depth_restored"]["mtp_drafts"], 7)

    def test_explicit_mtp_depth_is_readiness_bounded_and_default_is_recommended(self):
        ready = {"mtp": {"supported": True, "max_drafts": 7, "recommended_drafts": 3}}
        self.assertEqual(harness.selected_mtp_drafts(ready, None), 3)
        self.assertEqual(harness.selected_mtp_drafts(ready, 7), 7)
        self.assertEqual(harness.selected_mtp_drafts(ready, 0), 0)
        with self.assertRaisesRegex(SystemExit, "requested MTP depth exceeds worker readiness"):
            harness.selected_mtp_drafts(ready, 8)
        with self.assertRaisesRegex(SystemExit, "requested MTP depth exceeds worker readiness"):
            harness.selected_mtp_drafts(
                {"mtp": {"supported": True, "max_drafts": 3, "recommended_drafts": 3}}, 7
            )

    def streaming_pack(self, root):
        model = root / "model"
        model.mkdir()
        for name in harness.EXL3_FILES:
            (model / name).write_bytes(b"synthetic")
        config = {"model_type": "glm5_next", "residents_format": "fp8_block128",
                  "text_config": {"vocab_size": 8, "eos_token_id": 1},
                  "expert_streaming": {"family": "mlx-stream", "format": "mtplx-expert-manifest-v2",
                                       "manifest": "expert-manifest-v2.json",
                                       "runtime_manifest": "expert-manifest.json",
                                       "quant": "exl3-k3-mul1"}}
        (model / "config.json").write_text(json.dumps(config))
        (model / "tokenizer_config.json").write_text(json.dumps({"eos_token": {"content": "<eos>"}}))
        (model / "generation_config.json").write_text('{"eos_token_id":[1,2]}')
        (model / "chat_template.jinja").write_text("{{ bos_token }}{{ messages[0].content }}{{ eos_token }}")
        files = {}
        for path in model.iterdir():
            current = path.stat()
            files[path.name] = {"bytes": current.st_size, "mtime_ns": current.st_mtime_ns,
                                "inode": current.st_ino, "sha256": harness.file_sha256(path), "matched": True}
        receipt = {"passed": True, "destination": str(model), "repository": harness.EXL3_REPOSITORY,
                   "revision": harness.EXL3_REVISION, "files": files,
                   "total_bytes": sum(row["bytes"] for row in files.values())}
        path = root / "download-receipt.json"
        path.write_text(json.dumps(receipt))
        return model, path, receipt

    def test_verified_download_receipt_binds_both_banks_and_all_metadata_without_payload_rehash(self):
        with tempfile.TemporaryDirectory() as directory:
            model, receipt_path, receipt = self.streaming_pack(Path(directory))
            original_hash = harness.file_sha256
            with patch.object(harness, "file_sha256", wraps=original_hash) as hashing:
                identity = harness.download_receipt_identity(model, receipt_path)
            self.assertEqual([call.args[0] for call in hashing.call_args_list], [receipt_path])
            self.assertEqual(set(identity["files"]), harness.EXL3_FILES)
            self.assertIn("experts.bin", identity["files"])
            self.assertIn("mtp-experts-nvfp4.bin", identity["files"])
            self.assertIn("residents-fp8-manifest.json", identity["files"])
            first = harness.weights_sha256(model, receipt_path)
            receipt["files"]["experts.bin"]["sha256"] = "0" * 64
            receipt_path.write_text(json.dumps(receipt))
            self.assertNotEqual(first, harness.weights_sha256(model, receipt_path))
            with self.assertRaisesRegex(ValueError, "complete verified download receipt"):
                harness.weights_sha256(model)

    def test_download_receipt_rejects_stale_incomplete_wrong_revision_and_missing_expert(self):
        for mutation in ("stale", "incomplete", "revision", "missing_manifest_entry", "missing_expert"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                model, path, receipt = self.streaming_pack(Path(directory))
                if mutation == "stale":
                    (model / "experts.bin").write_bytes(b"changed expert bytes")
                elif mutation == "incomplete":
                    receipt["passed"] = False
                elif mutation == "revision":
                    receipt["revision"] = "0" * 40
                elif mutation == "missing_manifest_entry":
                    del receipt["files"]["experts.bin"]
                else:
                    (model / "experts.bin").unlink()
                path.write_text(json.dumps(receipt))
                with self.assertRaises(ValueError):
                    harness.download_receipt_identity(model, path)

    def test_download_receipt_rejects_unlisted_files_and_symlink_weights(self):
        for linked in (False, True):
            with self.subTest(linked=linked), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model, path, _ = self.streaming_pack(root)
                if linked:
                    (root / "external.bin").write_bytes(b"synthetic")
                    (model / "experts.bin").unlink()
                    (model / "experts.bin").symlink_to(root / "external.bin")
                else:
                    (model / "extra.safetensors").write_bytes(b"synthetic")
                with self.assertRaisesRegex(ValueError, "unlisted or linked"):
                    harness.download_receipt_identity(model, path)

    def test_raw_streaming_template_uses_native_evaluator_without_modifying_pack(self):
        try:
            harness.template_runtime._template_executable()
        except harness.template_runtime.TemplateHelperMissing:
            self.skipTest("native template helper has not been built")
        with tempfile.TemporaryDirectory() as directory:
            model, receipt_path, _ = self.streaming_pack(Path(directory))
            before = harness.download_receipt_identity(model, receipt_path)
            config = (model / "config.json").read_bytes()
            codec, template, provenance = harness.load_validation_template(model, receipt_path)
            self.assertIsInstance(template, harness.template_runtime.IsolatedTemplate)
            self.assertEqual(template.render(messages=[{"role": "user", "content": "synthetic"}],
                                             bos_token=codec.get("bos_token", ""), eos_token=codec["eos_token"]),
                             "synthetic<eos>")
            self.assertNotIn("bos_token", codec)
            self.assertEqual(provenance["source_template_sha256"], harness.file_sha256(model / "chat_template.jinja"))
            self.assertEqual(provenance["native_template_executable_sha256"],
                             harness.file_sha256(template._executable))
            self.assertEqual(config, (model / "config.json").read_bytes())
            self.assertEqual(before, harness.download_receipt_identity(model, receipt_path))

    def test_streaming_contract_and_prepared_template_paths_remain_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            model, _receipt_path, _ = self.streaming_pack(Path(directory))
            with self.assertRaisesRegex(ValueError, "pinned download receipt"):
                harness.load_validation_template(model)
            config = json.loads((model / "config.json").read_text())
            config["expert_streaming"]["quant"] = "other"
            self.assertFalse(harness.streaming_contract(config))
            (model / "config.json").write_text('{"token_codec_config":{}}')
            sentinel = ({"type": "ChatCodecConfig"}, object())
            with patch.object(harness, "load_template", return_value=sentinel) as prepared:
                self.assertEqual(harness.load_validation_template(model), (*sentinel, None))
                prepared.assert_called_once_with(model)

    def test_native_weight_digest_binds_gguf_and_external_ngram_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "weights-00001.gguf").write_bytes(b"synthetic-weight-one")
            first = harness.weights_sha256(model)
            (model / "weights-00002.gguf").write_bytes(b"synthetic-weight-two")
            second = harness.weights_sha256(model)
            self.assertNotEqual(first, second)
            (model / "ngram_table.bin").write_bytes(b"synthetic-ngram-one")
            third = harness.weights_sha256(model)
            self.assertNotEqual(second, third)
            (model / "ngram_table.bin").write_bytes(b"synthetic-ngram-two")
            self.assertNotEqual(third, harness.weights_sha256(model))

    def test_missing_weights_cannot_produce_an_empty_manifest_receipt(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(SystemExit, "no supported weight files"),
        ):
            harness.weights_sha256(Path(directory))

    def test_identical_errors_cannot_qualify_reuse(self):
        exactness = {
            "checks": [
                {
                    "successful": False,
                    "identical_to_fresh": True,
                    "identical_to_baseline_worker": True,
                    "reuse_matches_rule": True,
                }
            ],
            "invalidations": [{"as_documented": True}],
            "follow_ups_reuse": True,
        }
        self.assertFalse(harness.exactness_passed(exactness))
        exactness["checks"][0]["successful"] = True
        self.assertTrue(harness.exactness_passed(exactness))
        exactness["checks"] = []
        self.assertFalse(harness.exactness_passed(exactness))

    def test_boundary_snapshot_requires_two_remaining_tokens_and_matching_key(self):
        expected = harness.Expectation(1024, boundary_only=True)
        ids = list(range(1100))
        key = ("synthetic-scope", 4096, False)
        expected.record(ids, key, {})
        self.assertEqual(expected.predict(ids[:1025], key), 0)
        self.assertEqual(expected.predict(ids[:1026], key), 1024)
        self.assertEqual(expected.predict(ids, (key[0], 4095, False)), 0)
        self.assertEqual(expected.predict(ids, (key[0], 4096, True)), 0)
        ids[512] += 1
        self.assertEqual(expected.predict(ids, key), 0)

    def test_advertised_tail_moves_the_boundary_and_the_reuse_edge(self):
        expected = harness.Expectation(2048, boundary_only=True, tail=4)
        ids = list(range(9096))
        key = ("synthetic-scope", 16384, True)
        expected.record(ids, key, {})
        self.assertEqual(len(expected.entry[1]), 8192)
        self.assertEqual(expected.predict(ids[:8195], key), 0)
        self.assertEqual(expected.predict(ids[:8196], key), 8192)
        # One to three tokens past a chunk multiple keeps the earlier boundary.
        expected.record(ids[:8195], key, {})
        self.assertEqual(len(expected.entry[1]), 6144)
        self.assertEqual(harness.reuse_boundary(2051, 2048, 4), 0)
        self.assertEqual(harness.reuse_boundary(2052, 2048, 4), 2048)
        self.assertEqual(harness.reuse_boundary(1025, 1024), 1024)
        self.assertEqual(harness.reuse_boundary(0, 1024), 0)
        self.assertEqual(harness.advertised_tail({}, 1024), 1)
        self.assertEqual(harness.advertised_tail({"tail_tokens": 4}, 2048), 4)
        for invalid in (0, 2048, True, "4"):
            with self.subTest(invalid=invalid), self.assertRaises(SystemExit):
                harness.advertised_tail({"tail_tokens": invalid}, 2048)

    def test_midpoint_restore_geometry_changes_restore_only_off_the_physical_grid(self):
        settings = {"physical_chunk_tokens": 4096, "midpoint_restore_min_suffix_tokens": 17}
        midpoint = harness.advertised_midpoint(settings, 2048, 1)
        self.assertIsNone(harness.advertised_midpoint({}, 4096, 1))
        expected = harness.Expectation(2048, boundary_only=True, midpoint=midpoint)
        ids = list(range(4200))
        key = ("synthetic-scope", 8192, False)
        self.assertEqual(expected.boundary(2049), 2048)
        for boundary, suffix, reuse in ((2048, 16, 0), (2048, 17, 2048), (4096, 2, 4096)):
            with self.subTest(boundary=boundary, suffix=suffix):
                expected.record(ids[:boundary + 64], key, {})
                self.assertEqual(expected.predict(ids[:boundary + suffix], key), reuse)
        for invalid in ({"physical_chunk_tokens": 4096}, {**settings, "physical_chunk_tokens": 3000},
                        {**settings, "midpoint_restore_min_suffix_tokens": True}):
            with self.subTest(settings=invalid), self.assertRaises(SystemExit):
                harness.advertised_midpoint(invalid, 2048, 1)

    def test_tail_and_default_draft_cases_are_added_only_when_advertised(self):
        def cases(steps):
            return {label: (ids, options) for kind, label, ids, options in steps if kind == "case"}

        turns = [[1, 2] * 550] * 4
        default = harness.plan(turns, 1024, 4096, 2)
        self.assertEqual(default, harness.plan(turns, 1024, 4096, 2, tail=1, default_drafts=0))
        self.assertEqual(default, harness.plan(turns, 1024, 4096, 2, tail=2))
        tailed = cases(harness.plan(turns, 1024, 4096, 2, tail=4))
        self.assertEqual([label for label in tailed if label not in cases(default)],
                         ["restore_before_tail_edges", "snapshot_plus_tail_less_one",
                          "restore_between_tail_edges", "snapshot_plus_tail"])
        self.assertEqual(len(tailed["snapshot_plus_tail_less_one"][0]), 1024 + 3)
        self.assertEqual(len(tailed["snapshot_plus_tail"][0]), 1024 + 4)
        drafted = cases(harness.plan(turns, 1024, 4096, 2, default_drafts=3))
        self.assertEqual([label for label in drafted if label not in cases(default)],
                         ["explicit_plain_after_default", "explicit_plain_warm",
                          "default_after_explicit_plain", "default_warm"])
        self.assertEqual(drafted["explicit_plain_after_default"][1]["mtp_drafts"], 0)
        self.assertNotIn("mtp_drafts", drafted["default_after_explicit_plain"][1])

    def test_default_drafting_keys_a_plain_request_as_drafted(self):
        def worker(default_drafts, key=HYBRID_KEY):
            return SimpleNamespace(ready={"mtp": {"default_drafts": default_drafts}, "prefix_cache": {"key": key}})

        self.assertIs(harness.mtp_cache_value(worker(3), None), True)
        self.assertIs(harness.mtp_cache_value(worker(3), 0), False)
        self.assertIs(harness.mtp_cache_value(worker(3), 2), True)
        self.assertIs(harness.mtp_cache_value(worker(0), None), False)
        self.assertIs(harness.mtp_cache_value(SimpleNamespace(ready={}), None), False)
        self.assertEqual(harness.mtp_cache_value(worker(3, HYBRID_KEY + "+mtp-selection"), None), 3)
        self.assertEqual(harness.mtp_cache_value(worker(0, HYBRID_KEY + "+mtp-selection"), None), 0)

    def test_synthetic_run_with_a_long_tail_and_default_drafting_passes(self):
        with patch.object(FakeWorker, "tail", 4), patch.object(FakeWorker, "default_drafts", 3):
            result = self.run_synthetic(mtp_policy=None, mtp_lookup=False, mtp_drafts=None)
        self.assertTrue(harness.exactness_passed(result))
        self.assertEqual((result["tail_tokens"], result["mtp_default_drafts"]), (4, 3))
        rows = {check["case"]: check for check in result["checks"]}
        self.assertEqual(rows["snapshot_plus_tail_less_one"]["reused_input_tokens"], 0)
        self.assertGreater(rows["snapshot_plus_tail"]["reused_input_tokens"], 0)
        # A plain request drafts by default, so naming the drafts keeps the entry
        # and only an explicit zero drops it.
        self.assertGreater(rows["mtp_after_plain"]["reused_input_tokens"], 0)
        self.assertGreater(rows["plain_after_mtp"]["reused_input_tokens"], 0)
        self.assertEqual(rows["explicit_plain_after_default"]["reused_input_tokens"], 0)
        self.assertGreater(rows["explicit_plain_warm"]["reused_input_tokens"], 0)
        self.assertEqual(rows["default_after_explicit_plain"]["reused_input_tokens"], 0)
        self.assertGreater(rows["default_warm"]["reused_input_tokens"], 0)
        self.assertTrue(all(check["reused_input_tokens"] % FakeWorker.chunk == 0 for check in result["checks"]))

    def test_an_unadvertised_tail_or_default_drafting_fails_the_reuse_rule(self):
        for fault, attribute, value in (("tail_not_advertised", "tail", 4),
                                        ("default_drafts_not_advertised", "default_drafts", 3)):
            with self.subTest(fault=fault), patch.object(FakeWorker, attribute, value):
                result = self.run_synthetic(faults=frozenset({fault}), mtp_policy=None, mtp_lookup=False,
                                            mtp_drafts=None)
                self.assertFalse(harness.exactness_passed(result))
                self.assertTrue(all(check["identical_to_fresh"] for check in result["checks"]))
                self.assertFalse(all(check["reuse_matches_rule"] for check in result["checks"]))

    def test_timing_mismatch_or_missing_warm_reuse_fails_qualification(self):
        def turn(reused):
            return {
                "outputs_identical": True,
                "baseline": {"reused_input_tokens": [0]},
                "prefix_cache": {"reused_input_tokens": [reused]},
            }

        timing = {"turns": [turn(0), turn(1024)]}
        self.assertTrue(harness.timing_passed(timing))
        timing["turns"][1]["outputs_identical"] = False
        self.assertFalse(harness.timing_passed(timing))
        timing["turns"][1] = turn(0)
        self.assertFalse(harness.timing_passed(timing))

    def test_divergence_changes_tokens_even_when_neighbors_are_equal(self):
        ids = [1, 1, 1, 2]
        changed = harness.diverge(ids, 1)
        self.assertEqual(changed, [1, 2, 1, 2])
        self.assertEqual(ids, [1, 1, 1, 2])

    def test_template_preflight_rejects_unsupported_completed_tool_cycles(self):
        class RejectCompletedToolCycle:
            def render(self, **arguments):
                if (
                    sum(message["role"] == "tool" for message in arguments["messages"])
                    > 1
                ):
                    raise ValueError("synthetic rejection")
                return "synthetic rendering"

        args = SimpleNamespace(
            seed=11,
            mode="exactness",
            tools=2,
            instruction_paragraphs=2,
            follow_up_sentences=(4, 3, 8),
            progress=False,
        )
        with self.assertRaisesRegex(ValueError, "synthetic rejection"):
            harness.preflight_sessions(args, RejectCompletedToolCycle(), {})

    def test_opt_in_user_assistant_sessions_never_use_tools_and_default_is_unchanged(self):
        class AlternatingOnly:
            """Accepts one leading system message, then strict user/assistant alternation, and no tools."""

            def __init__(self):
                self.renders = []

            def render(self, **arguments):
                messages = arguments["messages"]
                if arguments.get("tools") is not None:
                    raise ValueError("synthetic tools rejection")
                roles = [message["role"] for message in messages if message["role"] != "system"]
                if any(role != ("user" if index % 2 == 0 else "assistant") for index, role in enumerate(roles)):
                    raise ValueError("synthetic role rejection")
                if any("tool_calls" in message for message in messages):
                    raise ValueError("synthetic tool-call rejection")
                self.renders.append(len(messages))
                return "synthetic rendering"

        base = {
            "seed": 11,
            "mode": "exactness",
            "tools": 2,
            "instruction_paragraphs": 2,
            "follow_up_sentences": (4, 3, 8),
            "progress": False,
        }
        template = AlternatingOnly()
        harness.preflight_sessions(SimpleNamespace(**base, session_kind="user-assistant"), template, {})
        self.assertEqual(template.renders, [2, 4, 6, 8])
        with self.assertRaisesRegex(ValueError, "synthetic tools rejection"):
            harness.preflight_sessions(SimpleNamespace(**base, session_kind="tools"), AlternatingOnly(), {})
        # The default and explicit "tools" kinds produce the same session and record.
        default, explicit = harness.Session(11, tools=2), harness.Session(11, tools=2, session_kind="tools")
        default.add_follow_up(4)
        explicit.add_tool_step(4)
        self.assertEqual(default.messages, explicit.messages)
        self.assertEqual(harness.session_options(SimpleNamespace(**base)), harness.session_options(
            SimpleNamespace(**base, session_kind="tools")))
        self.assertNotIn("session_kind", harness.session_options(SimpleNamespace(**base)))
        self.assertEqual(
            harness.session_options(SimpleNamespace(**base, session_kind="user-assistant"))["session_kind"],
            "user-assistant",
        )
        # The system and first user text are identical across kinds.
        plain = harness.Session(11, tools=2, session_kind="user-assistant")
        self.assertEqual(plain.messages, harness.Session(11, tools=2).messages)
        self.assertEqual(plain.tools, [])
        with self.assertRaises(ValueError):
            harness.Session(11, session_kind="unknown")


if __name__ == "__main__":
    unittest.main()
