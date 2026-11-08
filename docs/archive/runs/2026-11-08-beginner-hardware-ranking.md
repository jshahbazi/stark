# Beginner recommendations: hardware scaling and base-model exclusion

Date: 2026-11-08  
Implementation: `1cacef3ebe420e78b00b613a1f407202f8cdb6b7`  
Base: `ea098a244e25dfe5be934809c1fb87344183ac0c`

## Reproduced behavior

A metadata-only Qwen2 catalog scan reproduced the reported Coding + Balanced
choices in order: Qwen2.5-Coder-1.5B, its 1.5B Instruct variant, then 3B Instruct.
The host inputs were 128 GiB physical memory, 107,520,000,000 bytes of Metal
budget and 63,720,000,000 bytes free in both setup destinations. Available memory
was a synthetic 96 GiB; this was a recommendation replay, not a live Mac resource
measurement. The regression fixture preserves exact source revisions, parameter
counts, attention geometry and byte estimates without weights or user content.

Balanced previously used parameters divided by squared runtime memory. Once
candidates passed the same comfortable-fit tier, the score did not vary with
host capacity. A supported chat/tool template also admitted the official base
Coder checkpoint despite its pretrained completion role.

## Change

Balanced now uses `parameters / (1 + (runtime / target)^2)`, where `target` is
half the smaller of physical memory minus 2 GiB and 85% of the Metal budget.
Comfortable-fit and task-tag precedence, installation admission, exact source
bindings and deterministic tie-breaks are preserved. This is a capacity/memory
heuristic; no answer-quality or throughput measurement informed the score.

The six official Qwen2.5-Coder base repositories are excluded from the guide.
Their Instruct variants remain eligible. Source cards for
[0.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B),
[1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B),
[3B](https://huggingface.co/Qwen/Qwen2.5-Coder-3B),
[7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B),
[14B](https://huggingface.co/Qwen/Qwen2.5-Coder-14B) and
[32B](https://huggingface.co/Qwen/Qwen2.5-Coder-32B) were checked on this date;
all identify pretraining and advise against conversational use without
post-training. This exact-source exclusion does not classify every other source
or prevent Advanced installation.

## Validation

- Repository policy, converter Ruff lint/format and whitespace checks passed.
- Guided recommendation and discovery suites: 103 passed.
- Full offline converter suite: 6,703 passed, 96 skipped in 264.55 seconds.
- Native template helper built with locked Rust 1.98.1 dependencies for the
  isolated checkout before the full converter suite.
- Independent read-only review found no actionable regression.
- Live discovery checked 17 families, returned 227 metadata candidates and
  reported zero failed families. The guide admitted 34 planned candidates,
  including its two pinned starters. With the host inputs above, the resulting
  choices were 14B Instruct, 7B Instruct and Phi-4-mini-instruct.
- The selected 14B revision was
  `aedcc2d42b622764e023cf882b6652e646b95671`, with estimated runtime memory of
  38 GiB and setup space of 59,762,911,819 bytes. These are planning estimates.
- Fixture replay selected 1.5B Instruct for a 16 GiB / 12 GiB Metal-budget host,
  and 14B Instruct for the reported large host. It preserved order when input
  order reversed and retained storage limits. With ample disk, More model
  capacity selected 32B Instruct; Lighter model selected 0.5B Instruct.

## Limits

No weights were downloaded, no model was loaded, and no generation or coding
quality was evaluated. The app was not rebuilt or installed during this work.
Native hybrid/expert candidates still need their own guide memory planners;
they remain outside this recommendation pool. The source exclusions cover the
named official base repositories and do not certify all remaining candidates as
instruction-tuned assistants. Discovery remains bounded and may omit models.
