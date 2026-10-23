# Stark menu bar app

Stark's native macOS controller lives in the menu bar. Click the cat icon to install a model, choose a model and run it, or stop the loaded worker. Run opens a small chat window. Status and the last request's input/output token speeds appear in the dropdown and chat window.

## Build and install

Use an Apple silicon Mac with macOS 26.4 or later. Building the Swift app requires Xcode's command line tools, uv and an existing [standalone Stark CLI installation](../../docs/INSTALLATION.md). The resulting app includes that private runtime and runs without a source checkout, system Python, Rust or uv.

Build the current generation worker with the locked toolchain, then build and install the app:

```sh
PATH="$HOME/.cargo/bin:$PATH" cargo +1.98.1 build --release --locked -p stark-generate
uv run --no-project --python 3.13.13 python scripts/build_menubar.py --install --open
```

The app installs to `~/Applications/Stark.app`. Quit Stark before replacing an existing installation. The builder stages the new app, verifies its local signature and atomically replaces a matching Stark bundle; unrelated applications are refused. `--install-dir /absolute/folder` changes the parent folder. Omit `--install` to leave the app in ignored `artifacts/menu-bar/Stark.app`. `--runtime` selects a different standalone CLI bundle; `--worker` selects a different generation binary. Existing CLI installations are copied without changing them.

The local app is ad-hoc signed. This workflow adds no Developer ID signing, notarization, hosted release workflow or Gatekeeper bypass. Use the normal macOS approval flow for a downloaded build after verifying its source.

## Models and requests

**Install Model** offers a local folder or a Hugging Face repository ID. A converted Stark folder is registered in place. A supported local checkpoint is converted offline; a Hugging Face checkpoint is downloaded and converted with Stark's existing converter. Conversion receipts remain with managed models. Model support is governed by the [converter's supported formats](../../tools/converter/README.md); choosing a repository does not guarantee compatibility.

Run loads one model into one persistent worker. Requests are serialized. Stop terminates and reaps the worker, including an active request. The app stores its model registry and managed models beneath `~/Library/Application Support/Stark`; local folders registered in place remain at their original locations. Prompts, replies and conversation history stay in memory. Connected applications send text through the authenticated loopback API described below; native inference uses private process pipes.

Open Chat shows user and model messages in a scrollable conversation. Choose a model and select Run, then enter a message and select Send or press Command-Return. Return inserts a new line. New Chat clears the visible conversation and the worker's conversation context while keeping the model loaded. Switching models and selecting Run starts a fresh conversation; Stop also clears the session. Failed or cancelled messages return to the composer for retry. Chat history is never saved to disk. Replies appear when generation completes.

Input throughput uses first-token wall time as a prefill proxy, including native setup and scheduling. Output throughput measures sampling after the first token, including a terminal stop token when sampled. These are last-request measurements; short replies may not produce an output rate. See the [generation timing contract](../../docs/GENERATION.md). Stop or a worker failure clears the running session; starting again loads a fresh worker.

## Connect another application

Select **Connection Details…** in the menu or chat window. Choose an OpenAI-compatible provider in your application and copy the **Base URL**, **API key** and **Model ID** from this window. Select a model and **Run** first; the model ID refers to the model currently loaded, including when a different model is selected in the menu. **Copy curl Example** copies a complete authenticated request for that model. The copied command includes the API key.

The server starts with Stark and binds only to `127.0.0.1`, preferring port `8000`. If that port is occupied it chooses an available port; always use the URL shown in Connection Details. The base URL includes `/v1`. Keep Stark open while connected. The address is for native applications on the same Mac; browser-origin requests, remote devices and containers are unsupported. Each app launch creates a new random bearer key held only in memory and sent to the UI through its private process pipe. Reopening Stark requires updating the key and possibly the URL in your client. No request access logs or exception dumps are written.

Supported endpoints:

| Endpoint | Behavior |
| --- | --- |
| `GET /v1/models` | OpenAI-style model list containing the running model, or an empty list when stopped |
| `POST /v1/chat/completions` | Text messages with `system`, `user` and `assistant` roles; requires the running model ID |

Send `Authorization: Bearer <API key>` on both endpoints. Chat requests accept `model`, `messages`, `stream`, `max_tokens` or `max_completion_tokens`, `temperature: 0` and `n: 1`. Omit temperature to use greedy sampling. The reply budget defaults to 512 tokens, bounded by the loaded model; explicit budgets must fit that bound. Context is bounded by the model and capped at 8192 tokens, including the reserved reply budget. If a request has no system message, the model's default system prompt is prepended when available. The maximum HTTP body is 1 MiB. Unsupported fields, tools, images, structured output, nonzero temperature and the Responses API are rejected.

Both JSON completions and `stream: true` SSE responses are supported. Generation completes before SSE content is sent; there are no live token updates. Each API request supplies its full conversation and leaves the app's chat history unchanged. One worker handles both app chat and API calls; concurrent requests receive `409 engine_busy` and should retry after the current operation finishes. Stop cancels generation; subsequent requests receive `503 model_not_loaded` until Run loads a model. Quit closes the listener and terminates the worker.

This is a bounded Chat Completions interface. Applications requiring unsupported fields or a different API need further integration work. Model-free HTTP and worker-fixture tests validate the interface and lifecycle; they do not establish new model-quality or performance qualification.

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
