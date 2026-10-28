# Running-version labels, 2026-10-27

Starting from `1f1e71f42f39172cfe7c7af46ae7ee4f37784c06`, the menu and
Playground show the running model's quantization and stored weight-file size
beside its saved name. Both use a shared one-line label. Details come from the
loaded private ID's recorded metadata; missing values are omitted. The name and
long details can truncate to preserve adjacent controls.

All 99 production-controller checks passed in the independent final rerun.
The new regression distinguishes duplicate public names with different private
IDs, quantizations and sizes while another version is selected, and verifies
that stopped or missing models do not retain another version's details. Full
Swift compilation, Ruff, scoped formatting, repository policy and diff checks
passed. Independent review confirmed the identity binding; its layout concern
was addressed with one-line truncation and a bounded detail width.

Actual GUI checks used both existing saved models in the installed signed app.
Both the menu and Playground showed the correct name, quantization and size;
switching updated the labels, and Stop cleared the details in both views. Text
and controls fit without clipping. Manage Models retained its Run/Stop and
pinned-running-row behavior. No prompt was submitted, and no models were
downloaded or removed. QA ended with no model running and management open.

The Developer ID signed app at `~/Applications/Stark.app` matches the candidate's
tree, deep strict signature, Swift source and 58 packaged converter source files.
Its rollback copy was verified. The two-model registry is byte-identical and
selected-model, mode and sort preferences are preserved. Backend/engine source
is unchanged. The local build is not notarized. Private evidence remains in
ignored `runtime/running-version-label-validation-2026-10-27/`.

| Item | SHA-256 |
| --- | --- |
| Installed application tree | `7a6423125f47a1a5bd75b50a5897036093ed20e06a1435b828023a308f16d61c` |
| Signed Swift executable | `87e6afc4646b9b32eefd3c9699f035a82ec2439f9eafc1f53f5bf423ba02cf64` |
| Application manifest | `fb0fbefe91c76a2de92308574c1b520325ce766aa73e4b20877f125f0163faec` |
| Swift source | `5c26fa814a8635bef68ceae16dde990fb5a5604c89ed50ecf6075699eba76ed7` |
| Controller test source | `c76ba59527e367df34d0ea1c72f8f33da574ca840bd28d42746c4f4badafa016` |
| Installed-state and GUI aggregate | `86b5dd38859d42f00db7eadaaaa858ce8b76fb537304621f130eb2ac3969b412` |
