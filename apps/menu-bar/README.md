# Stark menu bar app

Stark's native macOS controller provides the supported manual Chat and Decision playground in the menu bar. Click the cat icon to install a model, choose a model and run it, or stop the loaded workers. Run opens the model workspace; Open Chat provides the conversation view and Decision provides the structured context/question/choices readout. Status and the last request's input/output token speeds appear in the dropdown and chat window.

## Build and install

For normal installation, use the [signed release installer](../../docs/INSTALLATION.md). It installs `~/Applications/Stark.app` together with the terminal CLI. No developer tools are required. Open the app, choose **Install Model**, select a supported model, then choose **Run** to chat.

The following commands are for rebuilding the app from source during development.

Use an Apple silicon Mac with macOS 26.4 or later. Building the Swift app requires Xcode's command line tools, uv and an existing [standalone Stark CLI installation](../../docs/INSTALLATION.md). The resulting app includes that private runtime and runs without a source checkout, system Python, Rust or uv.

Build both native workers with the locked toolchain, then build and install the app:

```sh
PATH="$HOME/.cargo/bin:$PATH" cargo +1.98.1 build --release --locked -p stark-generate
PATH="$HOME/.cargo/bin:$PATH" cargo +1.98.1 build --release --locked -p stark-score --features optimized
uv run --no-project --python 3.13.13 python scripts/build_menubar.py --install --open
```

The app installs to `~/Applications/Stark.app`. Quit Stark before replacing an existing installation. The builder stages the new app, verifies its local signature and atomically replaces a matching Stark bundle; unrelated applications are refused. `--install-dir /absolute/folder` changes the parent folder. Omit `--install` to leave the app in ignored `artifacts/menu-bar/Stark.app`. `--runtime` selects a different standalone CLI bundle; `--worker` selects a different generation binary; `--score-worker` selects a different scoring binary and defaults to `target/release/stark-score`. Existing CLI installations are copied without changing them.

The local app is ad-hoc signed. This workflow adds no Developer ID signing, notarization, hosted release workflow or Gatekeeper bypass. Use the normal macOS approval flow for a downloaded build after verifying its source.

## App updates

Choose **Check for Updates…** in the dropdown to check the latest stable GitHub release. If a newer app is available, choose **Upgrade to VERSION…** to download and prepare it, then **Install and Relaunch**. The current app stays open during download and verification. Installation quits Stark, stops active requests, clears the in-memory conversation and launches the replacement. Models and preferences remain in place.

The updater uses the release's versioned Apple silicon DMG and GitHub asset SHA-256. It checks the disk image and complete app signatures against Stark's production Developer ID team (`S5BT2PRN5B`), runs Gatekeeper assessment, checks the bundle ID, version, architecture and minimum macOS version, then stages a complete replacement before quitting. It uses the existing signed/notarized DMG packaging; no separate update feed, release asset or hosted workflow is required. Releases without a matching DMG or asset digest cannot be installed through the updater. No update is downloaded or installed automatically.

The destination must be a writable `Stark.app`. Move an app out of its mounted disk image before updating. A private helper runs from the verified candidate's bundled Python, locks the destination, waits for the old app to exit, atomically exchanges the bundles and relaunches. Detected activation, ownership-record or launch failures restore the old app and records; an unsuccessful rollback preserves the private backup. Fixed failure messages appear in the dropdown, including after relaunch. Prompts, model output and exception dumps are never logged.

App updates leave the separately installed terminal CLI at its existing version. CLI-managed installations keep both ownership records synchronized, including custom prefixes created by the current installer. Legacy default-prefix records are also supported. An older custom-prefix installation needs one upgrade with its CLI installer to record its state path before app-only updates are available; the updater refuses it before quitting. CLI prefix relocation likewise requires the installer to refresh that path. DMG drag-and-drop and local app builds have no CLI ownership record.

The updater first ships in the app built from this source. Older installed releases need one normal app installation before this dropdown action becomes available. Local development builds remain ad-hoc signed; update candidates must be production-signed releases. Developer tools and a source checkout are not needed for an installed app to update itself.

Run the model-free updater checks on an Apple silicon Mac:

```sh
uv run --no-project --python 3.13.13 python -m unittest scripts.test_app_update scripts.test_install_app
```

## Models and requests

**Install Model** offers a local folder or a Hugging Face repository name or URL, such as `Qwen/Qwen3-0.6B` or `https://huggingface.co/Qwen/Qwen3-0.6B`. Both forms select the same model. A converted Stark folder is registered in place. A supported local checkpoint is converted offline; a Hugging Face checkpoint is downloaded and converted with Stark's existing converter. Conversion receipts remain with managed models. Model support is governed by the [converter's supported formats](../../tools/converter/README.md); choosing a repository does not guarantee compatibility.

The install dialog shows separate Download and Conversion progress bars. Download counts selected checkpoint bytes, including files already in the Hugging Face cache. Conversion counts bytes written to the output weights. Each bar shows a percentage and byte counts once its total is known; preparation uses an indeterminate bar. Validation and publication finish before a stage is marked complete. Local sources skip downloading, and already converted models skip conversion. Stop cancels the install; reopen **Installation Progress** from the menu to follow an active install. Command-I also opens the install dialog while Stark is active.

Run loads one model into one persistent worker. Requests are serialized. The bundle also carries the standalone scoring worker for the manual Decision mode. Decision mode accepts a context, question and candidate choices, then displays the supplied-choice distribution at temperature 1; this is a comparative readout, not calibrated confidence. Stop terminates and reaps the worker, including an active request. The app stores its model registry and managed models beneath `~/Library/Application Support/Stark`; local folders registered in place remain at their original locations. Prompts, replies, token probabilities and decision history stay in memory. Connected applications send chat text through the authenticated loopback API described below; native inference uses private process pipes.

**Open Playground** opens the manual **Chat / Decision** selector. Choose a model and select Run. Changing modes replaces the loaded worker and clears the previous session's results. Each mode retains its own unsent text for the app session.

Decision mode accepts text, a question and 2–16 comma-separated answers. It requires distinct single-token answer letters and a template that exposes a direct-answer position with supported thinking settings. A template that leaves reasoning open, or whose answer boundary cannot be established, returns a compatibility error before scoring. The model stays loaded so you can switch to Chat or choose a different model. Literal reasoning tags in your input do not affect this check.

Chat mode shows user and model messages in a scrollable conversation. Enter a message and select Send or press Command-Return. Return inserts a new line. New Chat clears the visible conversation and the worker's conversation context while keeping the model loaded. Switching models and selecting Run starts a fresh conversation; Stop also clears the session. Failed or cancelled messages return to the composer for retry. Chat history is never saved to disk. Replies appear when generation completes.

After loading a model, select **Advanced Options…** in the chat window to adjust the system prompt, context size, output token limit and thinking mode. The controls stay hidden until this button opens the panel. Use the model's default system prompt, enter a custom prompt, or disable the default and leave the text empty to omit the system message. Thinking mode takes effect when the model's chat template supports it.

**Apply and New Chat** clears the conversation while keeping the model loaded. **Cancel** discards panel edits; **Restore Defaults** fills in the original defaults for review before applying. Context and output limits are validated against the loaded worker, and the context budget must leave room for the entire output limit. Defaults are up to 8192 context tokens, up to 512 output tokens and thinking off. Settings stay in memory for the app session and carry across New Chat, Stop/Run and model changes; a model with smaller limits reduces the token settings. Restarting the app restores defaults. System prompts and settings are not written to the model registry or preferences.

Input throughput uses first-token wall time as a prefill proxy, including native setup and scheduling. Output throughput measures sampling after the first token, including a terminal stop token when sampled. These are last-request measurements; short replies may not produce an output rate. See the [generation timing contract](../../docs/GENERATION.md). Stop or a worker failure clears the running session; starting again loads a fresh worker.

## Connect another application

Select **Connection Details…** in the menu or playground window. Choose an OpenAI-compatible provider in your application and copy the **Base URL**, **API key** and **Model ID** from this window. Select Chat mode, a model and **Run** first; the model ID refers to the model currently loaded, including when a different model is selected in the menu. **Copy curl Example** copies an authenticated streaming request for that model with curl buffering disabled. The copied command includes the API key.

The server starts with Stark and binds only to `127.0.0.1`, preferring port `8000`. If that port is occupied it chooses an available port; always use the URL shown in Connection Details. The base URL includes `/v1`. Keep Stark open while connected. The address is for native applications on the same Mac; browser-origin requests, remote devices and containers are unsupported. Each app launch creates a new random bearer key held only in memory and sent to the UI through its private process pipe. Reopening Stark requires updating the key and possibly the URL in your client. No request access logs or exception dumps are written.

Supported endpoints include authenticated model listing and detail, health, capability discovery, and text Chat Completions. See the [local API contract](../../docs/LOCAL_API.md) for exact routes, message normalization, supported parameters, error codes and limits.

Requests support live SSE streaming, native temperature/top-p/top-k/min-p sampling, seed and up to four stop strings. The reply budget defaults to 512 tokens and can increase to the loaded worker limit, capped at 4096. API context is model-bounded and capped at 16,384 tokens; Connection Details and model metadata show the actual limits. The app chat retains its smaller defaults. Text is tokenized once per request; no cross-request cache is added.

Up to eight API requests wait in FIFO order behind the active operation, with a 30-second wait limit. A full or expired queue returns HTTP 429 with Retry-After. Peer disconnects and request deadlines trigger cooperative cancellation at safe native boundaries and preserve the loaded worker. Model changes fence waiting and active requests. Native failures or older workers without cancellation support may require Run to reload the model. Stop unloads the worker; Quit closes the listener.

Streaming flushes valid Unicode text during generation and sends a finish reason and [DONE] on success. Optional include_usage adds a final usage chunk. Stop strings can span chunks and are withheld from output. Validation failures retain their HTTP status before streaming starts; later failures emit a safe error and close without a success finish. Streaming timings include decoding and output backpressure. Source-bound checks cover one local model and the serving contract; broader model quality, performance and universal client compatibility remain unqualified.

## Artwork and provenance

`Resources/Stark.jpg` is the user-supplied original cat artwork. The app icon is a padded, resized conversion of that image. `Resources/MenuBarTemplate.png` was edited directly from the original in Pixlr Express on 2026-10-21: remove the gray background, remove gold eyes and white highlight while retaining pupils, then flatten the remaining cat to black. No generative changes were used for the delivered assets. The template uses macOS template rendering for light and dark menu bars.

| Source asset | SHA-256 |
| --- | --- |
| Original JPG | `bfaecfd420af69ef3bcf6b5c92ae672a54e169eee59c8fe9ad4c8d9b119383f1` |
| Pixlr template PNG | `e51bf75a34ecfd83803714b613b1c4fd5d78c0d2feaf91b66379a9c936e5cfc5` |

The app carries the base CLI runtime's licenses, notices and original source manifest. The builder packages the complete current converter wheel with its source provenance capsule, licenses and distribution metadata. It also adds Jinja2 and MarkupSafe from the converter's existing lockfile using required hashes. Building requires uv and may fetch the pinned build backend and locked dependency wheels; the installed app includes them.

`Contents/Resources/menu-bar-manifest.json` records the source revision, source hashes, whether the checkout was dirty, the base runtime manifest hash and all app overlays, including the added Python package files. Its `converter_package` record identifies the wheel, build backend, source hashes and installed converter provenance. `artifacts/menu-bar/Stark.app.manifest.json` additionally records the signed executable hash. The base CLI manifest continues to identify the original distribution; the overlay manifest identifies changed worker and Python modules. Local build checks do not extend previous model qualification to this app or binary.

`--worker-build-record /path/to/record.json` embeds an optional development-worker build record after checking its `worker_sha256` against the selected worker. This preserves the distinction between a fresh locked Cargo build and a development worker that reuses previously compiled engine libraries.

`--score-worker-build-record /path/to/record.json` does the same for the scoring worker and is recorded separately from generation provenance. The builder verifies both workers with their content-free invalid-argument protocol before packaging.

Run the model-free packaging checks with:

```sh
uv run --no-project --python 3.13.13 python scripts/test_build_menubar.py
```
