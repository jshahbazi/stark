# 20261113-PIGUI-01–04: Pi Connections delivery

Registered retrospectively on 2026-11-13, owner Codex. Governing sources: user's
Advanced Connections placement and build/install request,
[integration plan](../AGENT_CLIENT_INTEGRATION_PLAN.md), and repository artifact
qualification rules. Full source and delivered identities are in the
[installed run record](../archive/runs/2026-11-13-pi-connections.md).

| Attempt | Recipe / exact result | Disposition |
| --- | --- | --- |
| 20261113-PIGUI-01 | Stark source `43ea570ed80d289b05d01de67790a0e96f687593`, build `20261113132858753344`; bundled integration `4455cbfe60f8a7b88e78d82cc0238c3dfec70ddd`; eight Qwen3-4B backend/client checks passed. Installed GUI failed to detect the user's nvm Node path. | GUI candidate superseded; model result retained as a bounded smoke, no complete GUI claim. |
| 20261113-PIGUI-02 | Stark source `d5557bce3c637480f3c8cd2a34ce43f6b88348d7`, build `20261113133359056674`; same integration; all eight fresh installed-bundle checks passed, actual GUI detected Pi through nvm, created profile and opened Terminal for a selected folder. User confirmed Pi selected Stark and supplied successful-generation evidence, with an `fd` lookup warning. | Signed build installed and manually accepted; launcher warning retained for correction in PIGUI-03. |
| 20261113-PIGUI-03 | Stark `f9a247e6c87f1f52901a5a869cdcdbf2ce93aff7`, build `20261113134611603911`, integration `349f9ceb0b2a65f883d35f4a8bb83a63356e5cce`: include Pi's lexical/resolved bin directories in isolated PATH; real PTY sibling-fd and self-consistent-old-profile refusal; 51 integration checks and eight fresh installed-model checks passed. | Final signed development build installed; fresh GUI profile saved; actual bundled environment resolves fd 10.4.2; Qwen3-4B Ready for user testing. |
| 20261113-PIGUI-04 | Stark `e00401f24f367094dd4c1f01fbd8f996ede14708`, candidate build `20261113143108361356`, integration `11e91d0a23c8a21dee5e6e0ba9b9833bacd17458`; default snapshot build, runtime rediscovery/availability, normal-quit clipboard cleanup and port-80 recovery; 54 client tests and 8/8 fresh native/client stages passed. | Signed candidate validated; prior user installation preserved. [Pre-PR review](../archive/runs/2026-11-13-pi-pr-review.md) records all four detected issues/corrections, receipt and scopes. |

Exact artifact: `mlx-community/Qwen3-4B-4bit` revision
`4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`; MLX affine Q4/group 64, weights
`7c2af9a45e5b0d196f06a5cbfd6fc5cc95adf29df79885d0540f2206fb3d40d8`.
Config/tokenizer/conversion and template hashes were rechecked against the prior
[PI-01–08 record](2026-11-13-pi-manual-qualification.md); no artifact drift.
Parser `qwen3-hermes-v1`, context 40,960, Pi reply ceiling 1,024, MTP/thinking off.
Text-only source; image checks N/A. No new activation/cache/weight precision path
or native kernel change was introduced. Signature removal on disposable copies
confirmed reused worker code; new delivered signature hashes are recorded.

Method: installed Pi 1.5.2, Node 26.13.0, real packaged Rust/Metal worker, isolated
backend/registry and ephemeral manual key through the private host pipe. A fresh
Stark-owned desktop profile contains only public code/configuration. Actual
read/edit/test/diff and failed-tool recovery are checked; prompts, outputs, tool
payloads and credentials stay in RAM. Private live fixtures were removed after
each recipe. The separately created user GUI profile remains for testing.

The final PIGUI-03 receipt SHA-256 is
`3a31aba2ad9d1b4c2a8b643533e17f158b2d66f1c2f5a01268a9a9134e32e2cb`;
harness SHA-256 `77ea14d22155fa6f64e8bdc712f919a25c545712063f7b9b4c92b359967293c1`.
Source review corrections and initial harness/environment failures are retained
in the run record. The user confirmed manual key submission opened Pi with Stark
and supplied successful-generation evidence. Computer-use tools deny Terminal
access; this is human acceptance evidence, without an automated TUI claim. The
image and its transcript are not published.

Earlier aggregate receipt hashes are retained: PIGUI-01
`c5658533c80ee0f562a1213724d543f5c2eebad34e1c92be9102e3b7343ed70c`;
PIGUI-02 `5f3365355b32e7e951c1d7d491d2b1be8c49fb66afb6fe1a1ad41e89fe125ecc`.

Correctness is bounded client/backend task evidence. No broad coding score,
fresh/reused numerical equivalence campaign, speedup, peak memory, full-window
measurement, image execution or managed-auth qualification follows. The earlier
21-case lifecycle matrix keeps its original source identities; it is not relabeled
as this build's test result. Other models/quantizations/client versions are outside
these attempts.
