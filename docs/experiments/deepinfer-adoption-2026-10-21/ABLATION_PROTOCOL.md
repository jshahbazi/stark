# Follow-up ablation protocol

The first combined-change campaign completed with exact output parity but only two accepted matched load pairs (0 and2). Its paired geometric mean was1.24% slower; the two pair-level results were-2.44% and-0.02%. It provides no demonstrated throughput benefit. Preserve it separately.

Before deciding whether to retain either runtime change, compare three builds: unchanged reference, argument-table recycling only, and direct token handoff only. The two ablation builds are exact source-file compositions of reference `8be4b52d` and candidate `f2a5a9885fd8c5b8c2ad01de1cdad558e1a8acaf`; record the file maps, patches and executable hashes. No new algorithm is introduced.

The argument-table build includes the complete command-metadata bundle: table retention/recycling, cache publication before waking the waiter, and failed-submission exclusion. It does not isolate table allocation from reference-count operations or completion-callback scheduling. The direct-token build retains the reference command-buffer implementation.

Use the same model, fixtures, timing boundaries and strict host guards as the first protocol, including zero system paging and hidden Codex windows. Four independent loads per engine and three measured repeats per performance fixture per load. The existing rotate/reverse order is pairwise balanced across these four blocks for all three engine pairs. Each load retains four one-repeat sanity cases. Warm all ten fixtures before fresh quiet admission.

Preserve every load, failure and exclusion. Analyze this campaign separately; do not pool its timings with the first campaign. Require exact output hash parity on all ten cases. A small or inconsistent accepted-subset difference is not a speed qualification. Do not retain runtime complexity merely because an operation count decreases; retain a performance candidate only with a credible positive measured result. If neither isolated change demonstrates benefit, restore production runtime code to the reference and retain the experiments, tests and findings as evidence.
