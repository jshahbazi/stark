# Stark menu bar app

A native macOS menu bar application for installing, running and chatting with converted models. Stark starts in the menu bar without restoring windows. Reopening the app shows its main menu when no window is visible; **Connection Details** and **Open Playground** open only through their named controls. Click the cat icon and **Manage Models** to run or stop a saved model. **Open Playground** opens a window with a **Chat** mode and a **Decision** mode. Status and the last request's input and output token rates appear in the menu and the playground window.

The main menu and playground show **GPU memory** as a percentage and compact usage bar beside the status in both modes. Green indicates usage below 75% of Metal's recommended memory budget, orange starts at 75%, and red starts at 90%. Whole percentages round down so the label agrees with the color thresholds. Hover for the allocated bytes and budget. It refreshes about once a second, including during model loading and requests. Apple silicon uses shared memory; the percentage measures the model worker's Metal resource allocation against the device's recommended working-set size, excluding CPU-only allocations and other applications. This budget is a performance guideline. The bar fills at 100%; percentages above the budget remain visible. If the budget is unavailable, the readout shows allocated bytes with an empty neutral bar. A dash means no measurement is available. Stop clears the readout.

## Build and install

For normal use, install the [signed release](../../docs/INSTALLATION.md). The commands below rebuild the app from source during development on an Apple silicon Mac with macOS 26.4 or newer, Xcode's command line tools, Rust 1.98.1, uv, and an existing [standalone CLI installation](../../docs/INSTALLATION.md#install-from-a-source-checkout) whose runtime the app copies. The builder refreshes native dependency and adaptation notices for the current workers using the pinned Rust toolchain and Cargo lockfile.

```sh
PATH="$HOME/.cargo/bin:$PATH" cargo +1.98.1 build --release --locked -p stark-generate
PATH="$HOME/.cargo/bin:$PATH" cargo +1.98.1 build --release --locked -p stark-score --features optimized
uv run --no-project --python 3.13.13 python scripts/build_menubar.py --install --open
```

The app is compiled directly with `swiftc` from `Sources/*.swift`. It installs to `~/Applications/Stark.app` and runs without the checkout, system Python, Rust or uv. The builder stages and verifies the new app, quits the running copy (force-killing that exact executable if it stalls), swaps the bundles atomically, and refuses to replace an unrelated application. Options: `--install-dir DIR` changes the parent folder; omitting `--install` leaves the app in the ignored `artifacts/menu-bar/` directory; `--runtime`, `--worker` and `--score-worker` select a different CLI bundle, generation binary or scoring binary (default `target/release/stark-score`).

For release DMGs containing **Install Stark**, open the image and choose that launcher. It verifies and stages the complete app before asking the running copy to quit. It prefers the current running Stark location, including `~/Applications`, then an existing `/Applications` copy, then an existing `~/Applications` copy, or a writable Applications folder. It quits, replaces and launches the exact selected destination, and shows either success or the specific failure. Models remain installed; active chats stop. The Keychain API credential and saved endpoint are reused by subsequent launches. Older DMGs use drag-and-drop; after copying `Stark.app`, quit the running old app before opening the installed app. Drag-copying alone does not stop an older running copy.

Local builds are ad-hoc signed; they are not notarized and cannot be installed by the updater. Build provenance records are described in [PROVENANCE.md](PROVENANCE.md).

## App updates

Stark checks stable release metadata automatically at launch and about once an hour while it runs. Opening the menu also checks when the last metadata check was more than 15 minutes ago. These background checks stay quiet when the network is unavailable. A newer release is shown with one **Upgrade to VERSION** button; choosing it downloads, verifies and stages the DMG, then quits Stark, stops active requests, clears the in-memory conversation, swaps the bundles and relaunches. No release DMG is downloaded or installed before that button is chosen. Models, preferences and the Keychain API credential are kept, while the active chat ends. The CLI is not updated by the app.

Before anything is installed, the updater requires: the release's versioned Apple silicon DMG with a SHA-256 digest published by GitHub, a `codesign` verification of the disk image and the complete app against Stark's Developer ID team (`S5BT2PRN5B`), a passing Gatekeeper assessment, and a matching bundle identifier, version, architecture and minimum macOS. A private helper runs from the verified candidate's bundled Python, locks the destination, waits for the old app to exit, re-verifies the staged copy, exchanges the bundles atomically and relaunches. Detected failures restore the old app. Background check failures remain quiet; errors from a requested upgrade appear in the dropdown with a retry action.

The destination must be a writable `Stark.app` outside a mounted disk image. CLI-managed installations keep their ownership records synchronized, including custom prefixes; an older custom-prefix installation needs one CLI upgrade before app-only updates are available, and the updater refuses it before quitting. DMG drag-and-drop and local builds have no CLI ownership record and update independently.

Model-free updater checks:

```sh
uv run --no-project --python 3.13.13 python -m unittest scripts.test_app_update scripts.test_install_app
```

## Models and requests

**Manage Models** contains three tabs: **Installed**, **Model Catalog** and
**Add from Source**. Installed lists saved versions; Model Catalog browses model
families; Add from Source accepts a local folder or a Hugging Face repository name or URL
such as `Qwen/Qwen3-0.6B`. A folder that already contains a converted Stark model
is registered in place; a supported local checkpoint is converted offline; a
Hugging Face checkpoint is downloaded and converted. Supported models are
governed by the [converter](../../tools/converter/README.md). The third tab shows
Download and Conversion progress; **Cancel Installation** cancels. Command-I
opens that tab in the same window.

The source tabs start with **Hugging Face**, selected by default, followed by
**Local Folder**.

**Manage Models** in the menu and Playground opens the shared management window.
These views show the running model's name and use management to choose
and run a model. Its quantization and stored weight-file size appear beside the
name when recorded, so saved versions can be distinguished. Installation uses
the third tab. Entered source and image options remain in memory when switching
tabs or reopening the window. Switching tabs preserves an active installation;
its progress and cancellation remain accessible. Completion stays in management.
Search and sort settings are separate for
**Installed**, the Model Catalog family overview and a family's detail view. Search filters
the visible rows immediately; family search uses the current catalogue and does
not fetch from Hugging Face again while you type. Search text stays within the
current browsing session. Sort choices are remembered between app launches.
The family overview uses compact, full-width rows in a single column.

**Installed** shows compact rows with the saved name, quantization, file size,
repo link, **Run** or **Stop**, and removal button on one line. **Run** selects
that exact saved version and loads it in the current Chat or Decision mode while
keeping management open. The active model's button becomes **Stop**; starting a
different model moves Stop to that row. Loading and stopping progress appears
in management. Run is unavailable while another app operation is busy. Stop can
cancel a model load or request; it cannot cancel an unrelated installation or
removal from a model row. The active model stays at the top in a separate
**Running** section; the remaining models follow your chosen sort order. Search
filters both sections. Multiple
saved variants keep a shared model heading. Hover a row for its family, source
version and image support when
available. Sort by model name A–Z or Z–A, or total disk
size ascending or descending. Saved versions with a recorded source repository
also have a **link** beside the file size; local imports can use their conversion
receipt to identify that source. Models without a known repository omit the link.
In **Model Catalog**, model cards group size and
quantization choices into compact rows. Choose a family, filter its model cards,
then choose a row and **Download**.
Inside a family, the tabs at the top switch between **All models** and each
available quantization or precision, such as **4-bit**, **8-bit** and **BF16**.
Search narrows the selected tab and sorting applies to its matching rows.
Entering another family starts at **All models**.
Each variant row has a **link** beside its download size that opens its exact
Hugging Face repository, including the publisher of a quantized export.
When text and image artifacts are both offered, choose which to install. Sort
the family overview by name A–Z or Z–A; family detail sorts by name in either
direction or model size ascending or descending. Unknown numeric values stay last
in either direction. Download and preparation progress
stay in the browser. A completed download appears in **Installed** without
selecting it; choose it and **Run** when ready. Some image installations stop
the running model for their required local checks.

The catalogue combines reference models with bounded live Hugging Face search
and shows representative compatible choices rather than every repository.
Compatible quantizations from one publisher can share a card across model sizes;
Base, Instruct, Thinking, context-length variants and fine-tunes remain distinct.
Selected registered quantized exports may join their registered base model
across publishers. Size and download bytes appear when source metadata provides
them. Browsing reads metadata only. Stark checks the selected model again before
downloading weights and uses that row's exact source version. **Back** or
**Done** leaves the browser; **Stop** cancels an active installation.

Use a model's trash button in **Installed** to review its removal. **Move to
Trash** removes an owned model folder from Stark and puts its files in macOS
Trash. Imported folders and paths Stark cannot establish as exclusively owned
offer **Remove from Stark**, which leaves their files in place. The confirmation
identifies the model and warns when its running worker will stop. Removing
another model keeps the current worker running.

Raw-checkpoint installation first checks bounded
metadata for the selected use before weight download. A conditional path
continues automatically; unsupported or inconclusive results stop inline with a
specific explanation and available alternatives. The actual download uses the
checked immutable primary source revision. Changing the source or requested image
support clears the previous installation warning. External catalog metadata is resolved
and checked again by conversion. Family coverage does not establish that a
complete model has loaded, executed or produced calibrated decisions. See the
[support states and preflight contract](../../docs/MODEL_SUPPORT.md).

**Run** loads the selected model into one worker, and requests are serialized through it. Chat mode uses the generation worker; Decision mode uses the scoring worker and shows the distribution over the supplied answer choices at temperature 1, which is a comparative readout rather than calibrated confidence. **Stop** terminates the worker, including an active request. Changing modes replaces the worker and clears the previous session.

Chat mode shows the conversation; send with the button or Command-Return, Return inserts a newline. **New Chat** clears the conversation and the worker's context while keeping the model loaded. Failed or cancelled messages return to the composer. Replies appear when generation completes; the app does not stream. Decision mode takes a context, a question and 2 to 16 comma-separated answers, and requires a template that exposes a direct-answer position; otherwise it reports a compatibility error before scoring.

**Enable image input when converting** opts into the supported Qwen3.5 vision path. It accepts the pinned reference checkpoint and compatible full merged BF16 dense models/fine-tunes. The version-2 family derives supported decoder and tower dimensions from validated configuration within fixed operation, indexing and storage bounds; previously converted version-1 0.8B artifacts remain readable. Converted vision folders are detected automatically. For a schema-3 family installation, Stark stops the running model and checks text/image execution, vocabulary accounting, recovery and cleanup before adding the model to the library. Run repeats native admission and uses a matching private receipt or checks again. Available-memory preflight includes the model's geometry-dependent check state and scratch allowance. Unsupported graphs, LoRA-only repositories, quantized vision weights and unrecognized metadata fail admission. See the [family contract](../../docs/VISION_FAMILY_CONTRACT.md).

Passing host evidence enables **Attach Image** in Chat mode. The composer and Connection Details distinguish **locally checked** models from **reference verified** models; a local pass does not prove visual accuracy or parity with the official model. Choose, drop, or paste one PNG/JPEG up to 4 MiB, preview it, or remove it before sending. PNG/JPEG paste representations are supported; TIFF-only clipboard images are rejected. The original bytes feed the common native image pipeline. One image stays in memory and is included with every turn; choose **New Chat** to replace it. Image loading can be cancelled; Cancel during generation stops the model. Stop, model/mode changes, advanced-option changes and quit clear the image. Recoverable errors restore the draft image.

Images briefly use opaque files in per-process owner-only temporary directories; claimed files are unlinked before decoding and normal completion/Stop/quit cleans unused files. Image bytes, paths and filenames are never included in the control messages or logs. Hard-killing both processes before a file is claimed can leave a private file.

**Advanced Options** in the playground window sets the system prompt, context size, output limit and thinking mode. Context defaults to the loaded model's configured maximum, including contexts above 16,384 tokens, and follows that maximum when switching models. Output defaults to up to 512 tokens and thinking is off; the context must leave room for the whole output limit. A custom context remains in memory for the app session and is bounded by each loaded model; returning to a larger model restores that custom value. **Apply and New Chat** starts a fresh conversation with the model loaded; **Restore Defaults** fills in the model's maximum context and the other defaults for review. Applying that maximum returns context to following the model. Settings carry across New Chat and Stop and Run. When the backend declares no finite maximum, Advanced Options identifies the value as a protocol conversation budget. Connection Details shows the model context as not declared; text requests allocate only their input plus requested output.

Input throughput is a first-token wall-clock proxy that includes native setup; output throughput counts sampling after the first token. Both describe the last request; short replies may not produce an output rate. See the [timing contract](../../docs/GENERATION.md#generate).

## Connect another application

Select **Connection Details…** in the menu or playground window. Choose an OpenAI-compatible provider in your application and copy the **Base URL**, **API key** and **Model ID** individually, or use **Copy All as JSON** to share formatted settings with an agent or LLM. Model IDs use the Hugging Face repository ID (`owner/model`) for Hugging Face installs and the model name for local installs. Existing installations expose these names automatically.

The modal's model picker chooses which model ID to inspect and include in the JSON. Choose **Chat** mode and use **Run** in Stark to serve that model. The JSON includes the selected model, the running model, whether they match, running-model limits, endpoints, supported capabilities and usage notes. It includes the API key even when the key is hidden in the modal. Treat that JSON as a credential; paste it only into trusted client settings. **Copy curl Example** copies a streaming request that prompts for the key with terminal echo disabled and passes it to curl over standard input. The example contains no key and is also included in the JSON when a model is running.

### Connect OpenWork without copying a key

For the official OpenWork macOS app, use **Connect to OpenWork** in the OpenWork section of Connection Details. Open OpenWork with a local workspace first. Stark installs its bundled connector into the native OpenCode global configuration directory identified from OpenWork's running engine. The configuration stores a local connector path, owner-only socket path, expected loopback Base URL and an ownership marker. It contains no API key or per-request capability. If Stark changes the configuration, reopen OpenWork and select **Stark** in its model picker.

If OpenWork settings already define a manual provider with ID `stark`, remove only that provider entry before connecting and keep your other provider entries. Stark does not migrate existing client credentials.

Stark approves the official OpenWork app identity (bundle ID `com.differentai.openwork`, Team ID `F5DJWB4CCV`) and its signed `opencode2` engine sidecar. The API master key stays in Stark's Keychain. A separate Keychain approval record controls whether OpenWork can receive a renewable 120-second bearer capability over Stark's private local socket. Stark associates the grant with the verified sidecar's process identity and current API session, and renews the same token when requested after 30 seconds. The connector keeps it in memory and places it in the outgoing Authorization header. The bearer is transferable if exposed; process identity gates issuance in Stark but does not bind later HTTP use to that process. OpenWork plugins share the authorized app process and may inspect request data or the bearer in memory, so only install plugins you trust.

Choose **Disconnect** to disable the approval and revoke current-session capabilities, cancelling requests already admitted with those grants. An expired 120-second lease does not cancel a request that was already admitted. To reconnect, select **Connect to OpenWork** again. If the Stark API port changes, choose **Reconnect** so the connector's expected Base URL is updated, then reopen OpenWork. Regenerating the master key affects clients configured with the copied key; managed OpenWork capabilities refresh through Stark. The separately copied API key remains available for other native clients; OpenWork's managed connector does not require copying it.

The stock helperless connection has narrow desktop validation with official OpenWork 0.18.63 / OpenCode 2.0.26 and the pinned Qwen3-4B model. Chat, follow-up and lifecycle recovery behaved as expected, while two bounded read-tool tasks did not fully meet their requested outputs. This is connection and recovery evidence, not a general agent-competence claim. See the [managed-connection validation record](../../docs/archive/runs/2026-10-31-openwork-managed-connection.md).

The server starts with Stark and binds only to `127.0.0.1`. **Saved port** defaults to 8000 and persists across launches; Stark never silently chooses another port. If the port is occupied, Connection Details explains the error. Close the conflicting app and choose **Retry API**, or enter another port and choose **Apply Port**, then update connected apps. Changing to an occupied port keeps an existing listener running at the displayed Base URL; the saved port remains the requested one for the next retry or launch. API failures leave the playground usable.

The Swift app reads its 256-bit bearer credential from macOS Keychain and sends it to the backend only over the private pipe. The key is created only when Keychain conclusively reports it missing and is saved before use. Unlock a locked Keychain or grant Stark access in Keychain Access, then choose **Retry API**. Corrupt entries require explicit **Regenerate API Key…** recovery. Regeneration explains that apps configured with the copied API key need the replacement; managed OpenWork capabilities refresh through Stark. Storage failure preserves the working key. **Show** and **Copy** never rotate it.

The stable Keychain service is `com.starkengine.menubar.local-api`, with account `bearer-key-v1`, for the canonical bundle with a signing Team ID. Canonical ad-hoc packages and direct development executables use the separate `com.starkengine.menubar.local-api.development` service. A noncanonical test bundle adds its sanitized bundle identifier to that development service, isolating packaged validation. These namespaces retain their credential across versions; switching between development and signed builds requires configuring that build's key. Port preferences are also separated by namespace. Keychain may require access approval when a signing identity changes; Stark reports access failure and does not create a substitute key.

Keychain reads and updates run off the main thread. If macOS waits for access approval, Connection Details explains the pending operation and the rest of Stark remains usable. Only one credential operation can be pending; retry, regeneration and port controls wait for it. After approving access in Keychain Access, quit and reopen Stark if the operation remains pending. Rebuilt ad-hoc development apps may need renewed approval for the existing item.

The key is for native applications on this Mac; browser-origin requests are rejected and remote devices are unsupported. Copies of the API key and connection JSON carry concealed and transient pasteboard markers so clipboard software that honors them can avoid retaining or syncing credentials. These markers are advisory; receiving applications and clipboard software that ignores them can still read the copied data. Use the explicit **Copy** button for the key; **Show** reveals it without enabling text selection.

The API offers authenticated health, capability, model listing and chat completion endpoints with streaming, temperature, top-p, top-k, min-p, seed and up to four stop strings. Workers with active passing image evidence accept inline PNG/JPEG image data URLs; the status document reports `image_verification`. Output defaults to the smaller of 512 tokens and the model limit (at most 4,096); context follows a declared finite model maximum, reported as `context_length` in `/v1/models`. Models without a declared finite maximum report `null` and `context_capacity_known: false`. Memory availability and request byte limits still apply. Up to eight requests queue behind the active one for at most 30 seconds. Routes, parameters, error codes and limits are in the [local API contract](../../docs/LOCAL_API.md).

## Where the app keeps state

| Data | Location |
| --- | --- |
| Model registry and managed models | `~/Library/Application Support/Stark` (folders registered in place stay where they are) |
| Local image-check receipts | `~/Library/Application Support/Stark/image-checks`; private identity hashes and boolean results, reused only for an identical runtime |
| Selected model, playground mode, pending update workspace | `UserDefaults` for `com.starkengine.menubar` (`selectedModel`, `playgroundMode`, `appUpdateWorkspace`) |
| Conversations, decision results, system prompt and limits | Memory only; cleared on Stop, mode change or quit |
| API key | macOS Keychain; stable service and account described above |
| API port | `UserDefaults`, keyed by the credential namespace; defaults to 8000 |
| Downloaded update candidates | `$TMPDIR/stark-update-<UUID>`; a failed update leaves it for inspection |
| Staged app and backup during an update | `.stark-update-*` directories next to `Stark.app` |
| Installation lock | `.Stark.app.install-lock` next to `Stark.app`; remove it only when no installer is running |

Deleting the Application Support folder and the UserDefaults domain resets the app completely. Prompts, replies, token probabilities and keys are never written to disk.

## Checks

```sh
uv run --no-project --python 3.13.13 python scripts/test_build_menubar.py
uv run --no-project --python 3.13.13 python -m unittest scripts.test_app_update scripts.test_install_app
uv run --project tools/converter --locked python scripts/test_menubar_controller.py
```

The first two run in CI. The controller test needs Xcode because it compiles `StarkApp.swift`.

Check connection JSON serialization and curl escaping without loading a model:

```sh
xcrun swiftc apps/menu-bar/Sources/ConnectionInfo.swift apps/menu-bar/Tests/ConnectionInfoTests.swift -o /tmp/stark-connection-info-tests
/tmp/stark-connection-info-tests
```

Check connection clipboard content and privacy markers using a unique private pasteboard:

```sh
xcrun swiftc apps/menu-bar/Sources/ConnectionInfo.swift apps/menu-bar/Sources/ConnectionClipboard.swift apps/menu-bar/Tests/ConnectionClipboardTests.swift -o /tmp/stark-connection-clipboard-tests
/tmp/stark-connection-clipboard-tests
```
