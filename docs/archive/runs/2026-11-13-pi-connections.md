# Pi Connections implementation and installed validation

Date: 2026-11-13. Owner: Codex. Manual setup is implemented under
**Advanced → Connections → Apps → Pi**, built, signed and installed in
`/Applications/Stark.app`. Beginner Guided Setup and Adjust Setup have no Pi
choice. This record supplements the earlier [guarded CLI matrix](2026-11-13-pi-guard-live.md).

## Delivered identities

| Item | Identity |
| --- | --- |
| Stark source | `f9a247e6c87f1f52901a5a869cdcdbf2ce93aff7`, clean at build; initial implementation `43ea570ed80d289b05d01de67790a0e96f687593`, nvm correction `d5557bce3c637480f3c8cd2a34ce43f6b88348d7` |
| Installed application | Development version `0.0.0`, build `20261113134611603911`; Developer ID team `S5BT2PRN5B`, hardened runtime and timestamp; no notarization/release claim |
| Installed manifest SHA-256 | `4dce165998f711760c77d303701c6875933553b21ce39ab867e23e341647e15a` |
| Client source | Separate integration repository `349f9ceb0b2a65f883d35f4a8bb83a63356e5cce` |
| Pi public resource lock | Seven exact Git-blob-matched files; lock SHA-256 `3126381a2ada27639b9dacdaa03c98892f29d7d55d65425302308f0e8d267e39` |
| Installed client / Node | Pi `1.5.2`; Node `26.13.0` through nvm |
| Delivered generation worker | `3a4acf235c148c8e3a76bc489c63e898c113f181cdbbbb77d5d6a2927d176774` |
| Delivered template worker | `adf63f0a8b64ce4c934f57b5f4503c1b727cf5983d5ff79fab73a3902431b912` |
| Live harness | SHA-256 `77ea14d22155fa6f64e8bdc712f919a25c545712063f7b9b4c92b359967293c1` |
| Installed live aggregate | SHA-256 `3a31aba2ad9d1b4c2a8b643533e17f158b2d66f1c2f5a01268a9a9134e32e2cb`; [immutable aggregate](https://github.com/jshahbazi/stark-agent-clients/blob/954a8ea334ccb85fbc51b7325b8db53582ec48c2/docs/runs/data/2026-11-13-pi-desktop-final-summary.json) |

Build inputs reused the previous installed workers from source
`9e33ab729581994269f56c64685e27a0bdd94c27`. Signing changes their complete-file
hashes. Disposable signature-removed copies of both the first and final candidates
matched the previous worker code for generate, score and template. No native
engine rebuild or precision change was
introduced. The builder overlays the current converter package and records its
source/dependency hashes separately.

All 21 compiled Swift source hashes and all 18 public resource hashes matched the
candidate manifest. The installed manifest matched the verified candidate.
`codesign --verify --deep --strict /Applications/Stark.app` passed after install.

## Requirement and evidence map

| Requirement / authority | Implementation and validation | Verdict |
| --- | --- | --- |
| Advanced Connections placement; user's explicit exclusion from beginner setup | Pi card in `ConnectionDetailsView`; existing beginner routing retained; switching to Beginner hides Connections without opening a new window; native mode/window checks and installed card observation | Conforms |
| Client code stays in the separate integration repository; plan ownership boundary | Public helpers/guard copied only through the immutable `PiClient.lock.json`; no runtime npm install; actual resource staging and eleven package tests | Conforms |
| Manual key remains out of generated files, arguments and logs; plan credential contract | Environment-only authenticated setup, public `${STARK_API_KEY}` reference, explicit concealed/transient clipboard copy, two-minute unchanged-item expiry; hidden Terminal input tested through a real PTY | Conforms within tested setup/launcher scope |
| Preserve default Pi configuration and unrelated settings | Fresh private Stark-owned profile per setup; isolated offline version probe; no global provider merge; bounded regular-file/hash checks and redirected-path refusals | Conforms |
| Required guard, fixed routing, stale model/version handling | Guard API marker, exact Pi version check at setup/launch, discovery refresh and metadata comparison; synthetic real-Pi tool loop and eight installed-backend checks | Conforms for the named artifact/client |
| Build/install a reviewable candidate | Signed candidate, clean source identity, full Swift/resource closure checks and exact installed manifest; installed UI successfully creates the profile and launches Terminal | Conforms |
| Human Terminal and TUI acceptance | User confirmed Pi opened with Stark and supplied a screenshot showing Pi 1.5.2, the guard loaded, the intended model and successful generation. Terminal UI automation was denied; no alternate automation was used. No transcript/image is published | User-confirmed manual acceptance; automated live requests use official RPC |

## Model-free and installed checks

- Integration suite: **51 passed, zero skipped**. Eight desktop tests include
  configuration integrity, unsupported version refusal, redirected/oversized
  files, shell quoting, actual pinned Pi tool execution, stale metadata refusal,
  credential absence after execution, and real PTY hidden-input/environment checks.
- Stark controller/build/packaging suite: **209 passed**. Native window and Pi
  controller checks: **15 passed**, with the controller check overlapping the
  209-check suite. Unique total across those invocations: **223**.
- Ruff on changed Python files, repository policy and both Git whitespace checks
  passed. Independent source review plus seven desktop/eleven package checks
  found no remaining blocker after corrections.
- Installed UI: Pi 1.5.2 detected through nvm; actual existing Qwen3-4B model
  started; **Set Up Pi** succeeded using the app's existing credential; selected
  disposable folder; **Copy Key & Open Pi…** reported Terminal opened.

The user confirmed the hidden-key submission opened Pi with Stark and provided
an image demonstrating successful generation. That image also exposed an `fd`
lookup warning: Pi's companion binary was installed beside the real launcher,
outside the isolated PATH. A follow-up includes both the lexical and resolved
Pi bin directories, without restoring inherited cloud/proxy/Node environment.
The PTY fixture exercises a symlinked Pi with a sibling `fd`; it passes. Actual
installed `fd 10.4.2` resolves in the corrected environment. The latest integration
suite has 51 passing checks, including rejection of a self-consistent old
profile whose source no longer matches the current bundle. Attempt
`20261113-PIGUI-03` is the final installed build in the identity table; all eight
fresh real-model checks passed again. Its actual GUI rejected the old profile,
detected Pi, and saved a fresh current profile. Qwen3-4B was restored to Ready for
the user. The existing Terminal window is preserved; reopening Pi from
Connections picks up the companion-tool PATH fix.

The first installed candidate passed eight model checks but its GUI did not find
nvm Node. That failure is retained as attempt `20261113-PIGUI-01`. The final
candidate adds nvm discovery and a fixture proving newest-compatible selection
without sourcing shell startup files. Earlier review also corrected root-owned
public resource handling, isolation of Pi's version probe, and Connections hiding
when switching to Beginner. A test harness initially omitted new Swift source
dependencies and used a Python environment without the converter; both were
corrected before the passing counts above. A first PTY test used an unsuitable
non-TTY wrapper; the final test exercises the generated scripts with `expect`.

## Fresh real-model checks

Attempt `20261113-PIGUI-03` ran the installed app's embedded Python/native backend,
bundled desktop setup/launcher and actual installed Pi executable in official
RPC mode. All **8/8** stages passed: backend startup, artifact registration, model
load, authenticated discovery, intended Pi route, text/follow-up, four actual
read/edit/test/diff tool rounds, and failed-command recovery with two tool rounds.
The requested test/diff commands and unchanged test-file hash were checked in RAM.
Private fixture profiles, registry, credentials and workspaces were removed;
the receipt reports `cleaned_up: true`.

Checkpoint: `mlx-community/Qwen3-4B-4bit`, revision
`4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`, MLX affine Q4/group 64. Weight SHA-256
`7c2af9a45e5b0d196f06a5cbfd6fc5cc95adf29df79885d0540f2206fb3d40d8`;
config/tokenizer identities match the prior record. Template SHA-256
`87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5`, parser
`qwen3-hermes-v1`. API context/output ceilings: 40,960/40,959; Pi reply ceiling
1,024. MTP/thinking off; text-only artifact, images N/A. Existing native precision
policy retained; no new numerical parity claim. Host remains macOS 26.7.2,
Mac17,7 arm64, 128 GiB RAM.

These checks establish bounded compatibility and installation. They do not rerun
the earlier 21-case cancellation/restart/compaction matrix, qualify another
artifact/version, measure broad coding competence, establish throughput or memory
performance, or issue managed grants. The user GUI test profile is intentionally
retained for testing; its directory contains public configuration/code. Pi's own
managed installer may perform its ordinary update housekeeping; Stark does not
edit its default settings/provider profile.

## Reproduction and disposition

Build with `scripts/build_menubar.py`, the existing standalone CLI base runtime,
explicit installed worker/helper paths, the locked `--pi-client-source` checkout
and the existing Developer ID. Verify the candidate before `install_app` replaces
the matching `/Applications/Stark.app` atomically. The [menu bar guide](../../../apps/menu-bar/README.md#pi-in-connections)
documents the GUI and build options.

For the bounded live recipe, run the integration repository's
`scripts/qualify-pi-live.mjs --app /Applications/Stark.app --model ABSOLUTE_MODEL
--pi ABSOLUTE_PI --phase basic --desktop-resources
/Applications/Stark.app/Contents/Resources/PiClient --output PRIVATE_SUMMARY`.
Supply exact absolute paths; no real key is needed for its isolated backend.

Both feature branches are pushed. No PR, merge, release or hosted CI success is
claimed. The installed development build is ready for user testing.
