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

**Install Model** offers a local folder or a Hugging Face repository name or URL, such as `Qwen/Qwen3-0.6B` or `https://huggingface.co/Qwen/Qwen3-0.6B`. Both forms select the same model. A converted Stark folder is registered in place. A supported local checkpoint is converted offline; a Hugging Face checkpoint is downloaded and converted with Stark's existing converter. Conversion receipts remain with managed models. Model support is governed by the [converter's supported formats](../../tools/converter/README.md); choosing a repository does not guarantee compatibility.

The install dialog shows separate Download and Conversion progress bars. Download counts selected checkpoint bytes, including files already in the Hugging Face cache. Conversion counts bytes written to the output weights. Each bar shows a percentage and byte counts once its total is known; preparation uses an indeterminate bar. Validation and publication finish before a stage is marked complete. Local sources skip downloading, and already converted models skip conversion. Stop cancels the install; reopen **Installation Progress** from the menu to follow an active install. Command-I also opens the install dialog while Stark is active.

Run loads one model into one persistent worker. Requests are serialized. Stop terminates and reaps the worker, including an active request. The app stores its model registry and managed models beneath `~/Library/Application Support/Stark`; local folders registered in place remain at their original locations. Prompts, replies and conversation history stay in memory and private process pipes. The app exposes no HTTP listener.

Open Chat shows user and model messages in a scrollable conversation. Choose a model and select Run, then enter a message and select Send or press Command-Return. Return inserts a new line. New Chat clears the visible conversation and the worker's conversation context while keeping the model loaded. Switching models and selecting Run starts a fresh conversation; Stop also clears the session. Failed or cancelled messages return to the composer for retry. Chat history is never saved to disk. Replies appear when generation completes.

After loading a model, select **Advanced Options…** in the chat window to adjust the system prompt, context size, output token limit and thinking mode. The controls stay hidden until this button opens the panel. Use the model's default system prompt, enter a custom prompt, or disable the default and leave the text empty to omit the system message. Thinking mode takes effect when the model's chat template supports it.

**Apply and New Chat** clears the conversation while keeping the model loaded. **Cancel** discards panel edits; **Restore Defaults** fills in the original defaults for review before applying. Context and output limits are validated against the loaded worker, and the context budget must leave room for the entire output limit. Defaults are up to 8192 context tokens, up to 512 output tokens and thinking off. Settings stay in memory for the app session and carry across New Chat, Stop/Run and model changes; a model with smaller limits reduces the token settings. Restarting the app restores defaults. System prompts and settings are not written to the model registry or preferences.

Input throughput uses first-token wall time as a prefill proxy, including native setup and scheduling. Output throughput measures sampling after the first token, including a terminal stop token when sampled. These are last-request measurements; short replies may not produce an output rate. See the [generation timing contract](../../docs/GENERATION.md). Stop or a worker failure clears the running session; starting again loads a fresh worker.

## Artwork and provenance

`Resources/Stark.jpg` is the user-supplied original cat artwork. The app icon is a padded, resized conversion of that image. `Resources/MenuBarTemplate.png` was edited directly from the original in Pixlr Express on 2026-10-21: remove the gray background, remove gold eyes and white highlight while retaining pupils, then flatten the remaining cat to black. No generative changes were used for the delivered assets. The template uses macOS template rendering for light and dark menu bars.

| Source asset | SHA-256 |
| --- | --- |
| Original JPG | `bfaecfd420af69ef3bcf6b5c92ae672a54e169eee59c8fe9ad4c8d9b119383f1` |
| Pixlr template PNG | `e51bf75a34ecfd83803714b613b1c4fd5d78c0d2feaf91b66379a9c936e5cfc5` |

The app carries the base CLI runtime's licenses, notices and original source manifest. The builder packages the complete current converter wheel with its source provenance capsule, licenses and distribution metadata. It also adds Jinja2 and MarkupSafe from the converter's existing lockfile using required hashes. Building requires uv and may fetch the pinned build backend and locked dependency wheels; the installed app includes them.

`Contents/Resources/menu-bar-manifest.json` records the source revision, source hashes, whether the checkout was dirty, the base runtime manifest hash and all app overlays, including the added Python package files. Its `converter_package` record identifies the wheel, build backend, source hashes and installed converter provenance. `artifacts/menu-bar/Stark.app.manifest.json` additionally records the signed executable hash. The base CLI manifest continues to identify the original distribution; the overlay manifest identifies changed worker and Python modules. Local build checks do not extend previous model qualification to this app or binary.

`--worker-build-record /path/to/record.json` embeds an optional development-worker build record after checking its `worker_sha256` against the selected worker. This preserves the distinction between a fresh locked Cargo build and a development worker that reuses previously compiled engine libraries.

Run the model-free packaging checks with:

```sh
uv run --no-project --python 3.13.13 python scripts/test_build_menubar.py
```
