# Grouped model choices, search and sorting, 2026-10-27

This change starts from `31119dd21e9dedac4456ba91e72c904ac146db5d` on
`codex/qwen35-vision-artifact-contract`. It groups family download choices into
model cards and adds live search and persistent sorting to model management.

## Delivered behavior

A family model card contains individual rows for its sizes and quantizations.
Each row retains its own download action, source revision, installed marker and
optional text/image choice. Publisher and named Base, Instruct, Thinking,
fine-tune and retained context/version suffixes distinguish model headings.
Curated MLX exports can share their registered base model's heading across
publishers; arbitrary publishers remain separate.

The optional `model_group` descriptor contains an opaque presentation ID, name
and publisher. It never changes the immutable catalogue entry ID used for
installation. Normalization removes a matching terminal quantization suffix and
the first standalone M/B/T model-size token. K context suffixes and later size
tokens stay in the name. Empty normalization falls back to the original model
name. Missing, malformed or conflicting presentation descriptors preserve
separate choices in the native client.

Search narrows visible rows as the user types, with separate transient queries
for downloaded models, the family overview and the selected family. It matches
names, publishers, sizes, quantization and version metadata, supports case and
diacritic folding, and does not repeat discovery while typing. A quantization
query filters individual rows within a model card. Empty search results have a
clear-search action.

Sort choices persist independently through sheet reopening and app relaunch:
downloaded models support names and visible versions' total disk size; the
family overview supports names; a selected family supports names and known
model sizes. Numeric sorts keep unknown values last in both directions. Rows
keep exact private model IDs and catalogue entry IDs after filtering/reordering.

## Completed checks

| Evidence | Result |
| --- | --- |
| Full locked converter suite, Hub offline, final source | 4,953 passed; 88 skipped. |
| Exact-source Swift controller suite | 81 passed. New cases cover grouping, live row search, numeric/name order, unknown values, conflicting metadata, exact filtered download/removal targets and isolated persistent preferences. |
| Builder suite | 23 passed; 15 subtests passed. |
| Swift, Ruff, scoped formatting, repository policy and diff checks | Passed; final app compiled and signed. |
| Independent review | Verified grouping boundaries, row identities, query isolation and preference lifetime. Numeric overflow, context-token normalization, cross-publisher bridge admission and empty-name fallbacks corrected before final qualification. |
| Actual installed GUI | Qwen3 sizes and 4-bit/8-bit/BF16 rows share headings; Base choices stay separate; Qwen3.5 registered MLX rows share the Qwen heading and image pickers fit. The existing Qwen3 version remains Downloaded. |
| Actual live filtering | Downloaded-name search, family-name search and a `4 bit` catalogue query narrow immediately. The latter leaves only six matching Qwen3 quantization rows. Empty-result and clear controls inspected. |
| Actual sorting and persistence | Downloaded disk-size order, family reverse-name order and catalogue reverse-size order verified. All three choices survived a confirmed quit/relaunch with a distinct process. Queries reopened empty. Default sorts restored after QA. |
| Existing user data | Two models retained; registry byte-identical; selected model and mode unchanged. Filtered removal confirmation named the correct imported model and was canceled. No model download or removal was performed for this change. |

The signed local build was installed at `~/Applications/Stark.app` using a
verified rollback copy, exact candidate/tree checks and deep strict signature
verification. All 58 packaged converter source files and the compiled Swift
source match the qualified checkout. The installed app is running. It is
Developer ID signed with hardened runtime and a secure timestamp, and is not
notarized. Native workers and numerical behavior are unchanged; no new inference
or parity claim is made. Private bundles, preferences and raw validation logs
remain in ignored `runtime/model-browser-grouping-validation-2026-10-27/`.

## Final identities

| Item | SHA-256 |
| --- | --- |
| Installed application tree | `8d77f5625f45d0b8cfb61aafe1adb9f9118316f7ec58133a559a416e7b470f8e` |
| Signed Swift executable | `974b232cb399a769893fa7b25935be643321d5f4c00cc271be9e7dfd6568dabc` |
| Application manifest | `f00233533d405404c50aa6c365ca40673d58f5886ae1d30996f874ccbb798415` |
| Swift source | `2b5d904ee351964ecfff1128228c2282e53f872a98a914aa8cf2a50947d36861` |
| Catalogue source | `3b55f89d6e6c69900c0200aedcd842c3f0433cd3d71d50b775350220edbe60ee` |
| Installed-state and GUI aggregate | `0afb4e86c33309f7dc38a2c285eb57ef4493218dc3b29f00a5c1f6f9c9e279b7` |
| Full converter-suite log | `af60a54f2f3b4f65ac922d624d9122527430fb73d211b746964bdbe31b5386ee` |
| Controller log | `bcbcaf49fe46a00ec2dc6b16d7b23d48e63d68fd8eab710279528bb1b9b99e51` |
| Builder log | `99d39b31de9600b1cace5387ac3f59bace2041acd69a349ccfa7ba9aa2d15b45` |
