# Pi automatic working-folder validation

The required working-folder picker made the launch flow look like installation
discovery. Open Stark Pi now launches directly in a private automatic workspace.
Change Working Folder is optional and applies to new sessions.

- Stark code revision: `d580348dce03ec6361719426c0b0e2164ed24cc3` (clean build).
- Integration source: `2a1a89b8e78787c92bfecf88f046edeb651096a0`; all seven
  distribution files verified against the owning Git blobs.
- Installed app: `/Applications/Stark.app`, build `20261113164222116283`.
- Embedded manifest SHA-256:
  `7204120da58a53ff64ed716d9dd37d4764d3a054f027fb43a27cca3c9338916a`.
- All 21 Swift-source and 18 resource hashes matched; installed and candidate
  manifests matched; deep/strict code-signature verification passed.

The client suite passed all 56 tests with zero skipped. Native controller/window
checks passed 15 tests; packaging/distribution/build checks passed 84 tests.
Repository policy and whitespace checks passed. A separate reviewer found no
blockers and independently exercised workspace creation/reuse, explicit-folder
quoting, unsafe redirected-workspace refusal and the native controller.

On the installed app, the Qwen3 4B model reached Ready. Physical mouse clicks
completed Set Up Pi, refreshed Check Installation, opened the optional attached
folder sheet and cancelled it without changing the automatic selection.
Open Stark Pi then returned the successful Terminal-open status and hidden-key
instructions directly, with no required folder picker. Terminal input and a new
inference turn were not automated in this UI check. Earlier source-bound native
model qualification remains in its original records.

The helper creates or reuses `~/Library/Application Support/Stark/Pi/Workspace`
without changing existing workspace files. Optional selection and reset are
covered by native tests. Default Pi settings remain outside this setup. Pi is
still available only in Advanced Connections.
