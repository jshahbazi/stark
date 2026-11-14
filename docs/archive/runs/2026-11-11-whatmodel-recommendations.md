# whatmodel.sh recommendations snapshot

Captured 2026-11-11 from whatmodel.sh source revision `2769f5ba536d5d78fc70a772048512471a499802`.
The tables show source ranking scores, not Stark measurements or compatibility claims.

## General

| Mac RAM | Rank | Model | Score | Uncertainty | Estimated memory | Quantization |
|---:|---:|---|---:|---:|---:|---|
| 8 GB | 1 | Spark-X2.6 4B | 67.2 | 1 | 5.2 GB | 4-bit |
| 8 GB | 2 | Spark-X2.5 4B | 64.7 | 0.7 | 3.8 GB | 4-bit |
| 8 GB | 3 | MiniCPM5 2B | 63.1 | 1 | 3.7 GB | 8-bit |
| 16 GB | 1 | Spark-X2.6 4B | 67.2 | 1 | 10.1 GB | 16-bit |
| 16 GB | 2 | Spark-X2.5 4B | 64.7 | 0.7 | 9.4 GB | 16-bit |
| 16 GB | 3 | MiMo-V2.6-Distill-Qwen-9B | 64.4 | 2.2 | 8.6 GB | 4-bit |
| 24 GB | 1 | Spark-X2.6 4B | 67.2 | 1 | 10.1 GB | 16-bit |
| 24 GB | 2 | Spark-X2.5 4B | 64.7 | 0.7 | 9.4 GB | 16-bit |
| 24 GB | 3 | MiMo-V2.6-Distill-Qwen-9B | 64.4 | 2.2 | 12.2 GB | 8-bit |
| 32 GB | 1 | Qwen3.8 27B | 71.8 | 0.7 | 20.4 GB | 4-bit |
| 32 GB | 2 | KAT-Coder-V2.5-Dev (35B-A3B) | 69.4 | 1.5 | 20.5 GB | 4-bit |
| 32 GB | 3 | Ornith 1.5 35B-A3B | 69.2 | 1.4 | 20.5 GB | 4-bit |
| 48 GB | 1 | Qwen3.8 27B | 71.8 | 0.7 | 34.6 GB | 8-bit |
| 48 GB | 2 | KAT-Coder-V2.5-Dev (35B-A3B) | 69.4 | 1.5 | 22.4 GB | 4-bit |
| 48 GB | 3 | Ornith 1.5 35B-A3B | 69.2 | 1.4 | 24 GB | 4-bit |
| 64 GB | 1 | Qwen3.8 27B | 71.8 | 0.7 | 34.6 GB | 8-bit |
| 64 GB | 2 | KAT-Coder-V2.5-Dev (35B-A3B) | 69.4 | 1.5 | 37.9 GB | 8-bit |
| 64 GB | 3 | Ornith 1.5 35B-A3B | 69.2 | 1.4 | 38.8 GB | 8-bit |
| 96 GB | 1 | Qwen3.8-Flash-Next | 73.4 | 0.9 | 58.7 GB | 2-bit |
| 96 GB | 2 | Qwen3.8 27B | 71.8 | 0.7 | 34.6 GB | 8-bit |
| 96 GB | 3 | Ling 3.0 Flash (124B-A5B) | 69.9 | 1.5 | 71 GB | 4-bit |
| 128 GB | 1 | Qwen3.8-Flash-Next | 73.4 | 0.9 | 58.7 GB | 2-bit |
| 128 GB | 2 | DeepSeek V4 Flash (0731) | 72.4 | 0.6 | 91.5 GB | 2-bit |
| 128 GB | 3 | Hunyuan Hy3 (295B-A21B) | 72 | 0.8 | 93.5 GB | 2-bit |
| 192 GB | 1 | MiMo-V2.6-Flash (309B-A15B) | 75.6 | 1 | 135.7 GB | 2-bit |
| 192 GB | 2 | DeepSeek V4 Flash Vision (Exp) | 73.6 | 2.3 | 104.1 GB | 2-bit |
| 192 GB | 3 | Qwen3.8-Flash-Next | 73.4 | 0.9 | 115.4 GB | 4-bit |
| 256 GB | 1 | MiMo-V2.6-Flash (309B-A15B) | 75.6 | 1 | 180.1 GB | MXFP4 |
| 256 GB | 2 | Muse Spark Open | 74.4 | 0.9 | 190.2 GB | 1-bit |
| 256 GB | 3 | DeepSeek V4 Flash Vision (Exp) | 73.6 | 2.3 | 165.3 GB | 4-bit |
| 512 GB | 1 | Hy4 preview (770B-A49B) | 76.7 | 1 | 248.4 GB | 2-bit |
| 512 GB | 2 | Qwen3.8 Max (2.4T-A95B) | 76.4 | 0.6 | 380 GB | 1-bit |
| 512 GB | 3 | MiMo-V2.6-Flash (309B-A15B) | 75.6 | 1 | 180.1 GB | MXFP4 |

## Coding

| Mac RAM | Rank | Model | Score | Uncertainty | Estimated memory | Quantization |
|---:|---:|---|---:|---:|---:|---|
| 8 GB | 1 | Spark-X2.6 4B | 63.3 | 2.4 | 5.2 GB | 4-bit |
| 8 GB | 2 | Spark-X2.5 4B | 58.1 | 1.6 | 3.8 GB | 4-bit |
| 8 GB | 3 | MiniCPM5 2B | 56.4 | 1.7 | 3.7 GB | 8-bit |
| 16 GB | 1 | Spark-X2.6 4B | 63.3 | 2.4 | 10.1 GB | 16-bit |
| 16 GB | 2 | MiMo-V2.6-Distill-Qwen-9B | 60.2 | 3.3 | 8.6 GB | 4-bit |
| 16 GB | 3 | Gemma 4 12B | 59.4 | 3.2 | 10.2 GB | 4-bit |
| 24 GB | 1 | Spark-X2.6 4B | 63.3 | 2.4 | 10.1 GB | 16-bit |
| 24 GB | 2 | MiMo-V2.6-Distill-Qwen-9B | 60.2 | 3.3 | 12.2 GB | 8-bit |
| 24 GB | 3 | Gemma 4 12B | 59.4 | 3.2 | 15.2 GB | 8-bit |
| 32 GB | 1 | Qwen3.8 27B | 73.7 | 0.7 | 20.4 GB | 4-bit |
| 32 GB | 2 | Ornith 1.5 35B-A3B | 67.5 | 2.9 | 20.5 GB | 4-bit |
| 32 GB | 3 | KAT-Coder-V2.5-Dev (35B-A3B) | 66.3 | 2.5 | 20.5 GB | 4-bit |
| 48 GB | 1 | Qwen3.8 27B | 73.7 | 0.7 | 34.6 GB | 8-bit |
| 48 GB | 2 | Ornith 1.5 35B-A3B | 67.5 | 2.9 | 24 GB | 4-bit |
| 48 GB | 3 | KAT-Coder-V2.5-Dev (35B-A3B) | 66.3 | 2.5 | 22.4 GB | 4-bit |
| 64 GB | 1 | Qwen3.8 27B | 73.7 | 0.7 | 34.6 GB | 8-bit |
| 64 GB | 2 | Ornith 1.5 35B-A3B | 67.5 | 2.9 | 38.8 GB | 8-bit |
| 64 GB | 3 | KAT-Coder-V2.5-Dev (35B-A3B) | 66.3 | 2.5 | 37.9 GB | 8-bit |
| 96 GB | 1 | Qwen3.8-Flash-Next | 76.1 | 0.7 | 58.7 GB | 2-bit |
| 96 GB | 2 | Qwen3.8 27B | 73.7 | 0.7 | 34.6 GB | 8-bit |
| 96 GB | 3 | Ornith 1.5 35B-A3B | 67.5 | 2.9 | 38.8 GB | 8-bit |
| 128 GB | 1 | Qwen3.8-Flash-Next | 76.1 | 0.7 | 58.7 GB | 2-bit |
| 128 GB | 2 | Qwen3.8 27B | 73.7 | 0.7 | 34.6 GB | 8-bit |
| 128 GB | 3 | DeepSeek V4 Flash (0731) | 73.5 | 0.7 | 91.5 GB | 2-bit |
| 192 GB | 1 | MiMo-V2.6-Flash (309B-A15B) | 76.7 | 0.8 | 135.7 GB | 2-bit |
| 192 GB | 2 | Qwen3.8-Flash-Next | 76.1 | 0.7 | 115.4 GB | 4-bit |
| 192 GB | 3 | GLM-5.3-Flash | 74.9 | 0.8 | 102.8 GB | 2-bit |
| 256 GB | 1 | MiMo-V2.6-Flash (309B-A15B) | 76.7 | 0.8 | 180.1 GB | MXFP4 |
| 256 GB | 2 | Qwen3.8-Flash-Next | 76.1 | 0.7 | 115.4 GB | 4-bit |
| 256 GB | 3 | GLM-5.3-Flash | 74.9 | 0.8 | 102.8 GB | 2-bit |
| 512 GB | 1 | Qwen3.8 Max (2.4T-A95B) | 78.5 | 0.7 | 380 GB | 1-bit |
| 512 GB | 2 | MiMo-V2.6-Flash (309B-A15B) | 76.7 | 0.8 | 180.1 GB | MXFP4 |
| 512 GB | 3 | Hy4 preview (770B-A49B) | 76.3 | 0.8 | 248.4 GB | 2-bit |

## Scope and limitations

- Source scores and estimated memory come from dated whatmodel.sh API snapshots. They may be incomplete or outdated and are not Stark artifact measurements.
- Source memory estimates allow an 8K token conversation; Stark beginner recommendations plan around a 32K context. Memory estimates do not establish Stark fit.
- Speed entries retain source hardware and `how` fields; speeds measured elsewhere are not Stark performance claims.
- Model repository identities come only from exact Hugging Face URLs in the pinned source catalogue. Name similarity is not identity evidence.
- This snapshot adds recommendation evidence only. It adds no conversion, backend, or native model compatibility.
- Source model catalogue SHA-256: `b1a6b655bc71b6120842a5ad4c771cdc61918953a54827d08f32a88c36b79f56`. Raw API response hashes are in the snapshot JSON.
- whatmodel.sh numerical compilation and derived scores are CC BY-SA 4.0. Source-specific grants and full notices are preserved under `tools/converter/stark_convert/_third_party/whatmodel/`.
