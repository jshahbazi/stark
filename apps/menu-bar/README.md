# Stark menu bar app

Stark's native macOS controller lives in the menu bar. Click the cat icon to install a model, choose a model and run it, or stop the loaded worker. Run opens a small chat window. Status and the last request's input/output token speeds appear in the dropdown and chat window.

## Build and install

Use an Apple silicon Mac with macOS 26.4 or later. Building the Swift app requires Xcode's command line tools, uv and an existing [standalone Stark CLI installation](../../docs/INSTALLATION.md). The resulting app includes that private runtime and runs without a source checkout, system Python, Rust or uv.

Build the current generation worker with the locked toolchain, then build and install the app:

```sh
PATH="$HOME/.cargo/bin:$PATH" cargo +1.98.1 build --release --locked -p stark-generate
uv run --no-project --python 3.13.13 python scripts/build_menubar.py --install --open
```

The app installs to `~/Applications/Stark.app`. After staging and verifying the new app, the builder automatically quits the copy being replaced. If normal shutdown stalls, it terminates that copy, using a force kill if necessary, and waits for it to exit before atomically replacing the bundle. Active chats and model operations stop during replacement. This also applies to a running copy in the build output folder. `--open` launches the new app after installation. Unrelated applications are refused. `--install-dir /absolute/folder` changes the parent folder. Omit `--install` to leave the app in ignored `artifacts/menu-bar/Stark.app`. `--runtime` selects a different standalone CLI bundle; `--worker` selects a different generation binary. Existing CLI installations are copied without changing them.

The local app is ad-hoc signed. This workflow adds no Developer ID signing, notarization, hosted release workflow or Gatekeeper bypass. Use the normal macOS approval flow for a downloaded build after verifying its source.

## Models and requests

**Install Model** offers a local folder or a Hugging Face repository ID. A converted Stark folder is registered in place. A supported local checkpoint is converted offline; a Hugging Face checkpoint is downloaded and converted with Stark's existing converter. Conversion receipts remain with managed models. Model support is governed by the [converter's supported formats](../../tools/converter/README.md); choosing a repository does not guarantee compatibility.

Run loads one model into one persistent worker. Requests are serialized. Stop terminates and reaps the worker, including an active request. The app stores its model registry and managed models beneath `~/Library/Application Support/Stark`; local folders registered in place remain at their original locations. Prompts, replies and conversation history stay in memory. Connected applications send text through the authenticated loopback API described below; native inference uses private process pipes.

Open Chat shows user and model messages in a scrollable conversation. Choose a model and select Run, then enter a message and select Send or press Command-Return. Return inserts a new line. New Chat clears the visible conversation and the worker's conversation context while keeping the model loaded. Switching models and selecting Run starts a fresh conversation; Stop also clears the session. Failed or cancelled messages return to the composer for retry. Chat history is never saved to disk. Replies appear when generation completes.

Input throughput uses first-token wall time as a prefill proxy, including native setup and scheduling. Output throughput measures sampling after the first token, including a terminal stop token when sampled. These are last-request measurements; short replies may not produce an output rate. See the [generation timing contract](../../docs/GENERATION.md). Stop or a worker failure clears the running session; starting again loads a fresh worker.

## Connect another application

Select **Connection Details…** in the menu or chat window. Choose an OpenAI-compatible provider in your application and copy the **Base URL**, **API key** and **Model ID** from this window. Select a model and **Run** first; the model ID refers to the model currently loaded, including when a different model is selected in the menu. **Copy curl Example** copies an authenticated streaming request for that model with curl buffering disabled. The copied command includes the API key.

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

The app carries the base CLI runtime's licenses, notices and original source manifest. The builder adds Jinja2 and MarkupSafe from the converter's existing lockfile, installs only those two wheels with required hashes, and includes their licenses and distribution metadata. This dependency step requires uv on the build Mac and may fetch the locked wheels; the installed app includes them.

`Contents/Resources/menu-bar-manifest.json` records the source revision, source hashes, whether the checkout was dirty, the base runtime manifest hash and all app overlays, including the added Python package files. `artifacts/menu-bar/Stark.app.manifest.json` additionally records the signed executable hash. The base CLI manifest continues to identify the original distribution; the overlay manifest identifies changed worker and Python modules. Local build checks do not extend previous model qualification to this app or binary.

`--worker-build-record /path/to/record.json` embeds an optional development-worker build record after checking its `worker_sha256` against the selected worker. This preserves the distinction between a fresh locked Cargo build and a development worker that reuses previously compiled engine libraries.

Run the model-free packaging checks with:

```sh
uv run --no-project --python 3.13.13 python scripts/test_build_menubar.py
```
