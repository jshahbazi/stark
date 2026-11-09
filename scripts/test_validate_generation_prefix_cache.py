"""Model-free regression checks for the model onboarding qualification harness."""

import importlib.util
import json
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_spec = importlib.util.spec_from_file_location(
    "prefix_qualification",
    Path(__file__).with_name("validate_generation_prefix_cache.py"),
)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


class PrefixQualificationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
