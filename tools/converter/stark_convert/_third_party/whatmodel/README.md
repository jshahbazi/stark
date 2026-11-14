# whatmodel.sh snapshot data notices

Source checkout revision: `2769f5ba536d5d78fc70a772048512471a499802`.
Source model catalogue: `data/models.json` (SHA-256 `b1a6b655bc71b6120842a5ad4c771cdc61918953a54827d08f32a88c36b79f56`).

## License scope

The numerical compilation, adaptations, and derived scores originate from whatmodel.sh and are distributed under CC BY-SA 4.0, as described in `DATA-LICENSE.md`. Underlying source grants remain applicable to their included material; the full source license texts and notices are copied unchanged from the pinned checkout under `licenses/`. The repository `LICENSE` covers software code, not the numerical data compilation. Model weights, trademarks, and application code are not included or relicensed by this snapshot.

## Snapshot transformation

The refresh script consumes saved whatmodel.sh JSON responses (or downloads them only with explicit `--fetch`). For each task and RAM bucket it retains source-order rows whose `where` value is `local` or `local-slow`, including rows with null scores. It copies only model ID, display name, score, uncertainty, estimated memory, quantization, and the source speed object; commands, code, prompts, provider prose, and model-card text are omitted. Exact Hugging Face repository mappings come only from `model_repo` URLs in the pinned `data/models.json`; IDs are never matched by name. Raw response hashes, query specs, score metadata, and source attributions are preserved in the generated snapshot.
