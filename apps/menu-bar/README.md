# Stark menu bar app

Converter failures report fixed recovery guidance for insufficient disk space,
file permissions, unsupported formats/configurations and invalid checkpoints.
Unknown failures retain a generic message. These categories travel over the
private installation progress pipe; exception text and file paths are not
displayed or logged. Cancelling an installation discards its pending failure.

A native macOS menu bar application for installing, running and chatting with converted models. Stark starts in the menu bar without restoring windows. Reopening the app shows its main menu when no window is visible; **Connections** and **Open Playground** open only through their named controls. Click the cat icon and **Manage Models** to run or stop a saved model. **Open Playground** opens a window with a **Chat** mode and a **Decision** mode. Status and the last request's input and output token rates appear in the menu and the playground window.

The status dot is red when idle or in error, orange during work, and green when ready.

The main menu and playground model headings show the readable model name. Hover over the running model name for its quantization and file size.

The menu offers **Guided setup** in Beginner mode or **Manage Models** in Advanced mode, plus a **Start model** action when a model can be started and a **Stop model** action while one is starting or running. Beginner setup can be reopened from the menu; **Adjust Setup** returns to its choices so you can revisit earlier steps. Guide steps explain their prerequisites, and returning to a parent window preserves the guide for later.

The guide's task choices scroll independently of the **Find a model** action, which stays in a fixed footer at the bottom right. When choosing another model, **Keep current model** remains in the same footer.

The main menu and playground show **GPU memory** as a percentage and compact usage bar beside the status in both modes. Green indicates usage below 75% of Metal's recommended memory budget, orange starts at 75%, and red starts at 90%. Whole percentages round down so the label agrees with the color thresholds. Hover for the allocated bytes and budget. It refreshes about once a second, including during model loading and requests. Apple silicon uses shared memory; the percentage measures the model worker's Metal resource allocation against the device's recommended working-set size, excluding CPU-only allocations and other applications. This budget is a performance guideline. The bar fills at 100%; percentages above the budget remain visible. If the budget is unavailable, the readout shows allocated bytes with an empty neutral bar. A dash means no measurement is available. Stop clears the readout.

## Build and install

For normal use, install the [signed release](../../docs/INSTALLATION.md). The commands below rebuild the app from source during development on an Apple silicon Mac with macOS 26.4 or newer, Xcode's command line tools, Rust 1.98.1, uv, and an existing [standalone CLI installation](../../docs/INSTALLATION.md#install-from-a-source-checkout) whose runtime the app copies. The builder refreshes native dependency and adaptation notices for the current workers using the pinned Rust toolchain and Cargo lockfile.

```sh
PATH="$HOME/.cargo/bin:$PATH" cargo +1.98.1 build --release --locked -p stark-generate
PATH="$HOME/.cargo/bin:$PATH" cargo +1.98.1 build --release --locked -p stark-score --features optimized
PATH="$HOME/.cargo/bin:$PATH" cargo +1.98.1 build --release --locked -p stark-template
uv run --no-project --python 3.13.13 python scripts/build_menubar.py --install --open
```

The app is compiled directly with `swiftc` from `Sources/*.swift`. It installs to `~/Applications/Stark.app` and runs without the checkout, system Python, Rust or uv. The builder stages and verifies the new app, quits the running copy (force-killing that exact executable if it stalls), swaps the bundles atomically, and refuses to replace an unrelated application. Options: `--install-dir DIR` changes the parent folder; omitting `--install` leaves the app in the ignored `artifacts/menu-bar/` directory; `--runtime`, `--worker`, `--score-worker` and `--template-worker` select a different CLI bundle, generation binary, scoring binary or native template evaluator. Native binaries default to their corresponding `target/release/` paths.

The builder embeds `stark-template` in the app's runtime and verifies a synthetic
render through the installed converter. Chat and Decision modes require this
helper; template execution has no Python fallback. Models and conversation
content are not needed for the build checks.

For release DMGs containing **Install Stark**, open the image and choose that launcher. It verifies and stages the complete app before asking the running copy to quit. It prefers the current running Stark location, including `~/Applications`, then an existing `/Applications` copy, then an existing `~/Applications` copy, or a writable Applications folder. It quits, replaces and launches the exact selected destination, and shows either success or the specific failure. Models remain installed; active chats stop. The Keychain API credential and saved endpoint are reused by subsequent launches. Older DMGs use drag-and-drop; after copying `Stark.app`, quit the running old app before opening the installed app. Drag-copying alone does not stop an older running copy.

Local builds use the unique available **Developer ID Application** identity by
default. The builder signs and verifies every embedded native executable, then
seals the app with its canonical bundle identifier, hardened runtime and a secure
timestamp. Rebuilds with the same signing team retain the app's Keychain identity.
The certificate and private key must already be available in your Keychain;
the builder does not import credentials. Missing or ambiguous identities and
signing failures stop the build before replacing the previous app.

Use `--sign-identity 'Developer ID Application: NAME (TEAMID)'` or a certificate
SHA-1 to choose an identity explicitly, and `--keychain PATH` to select its
Keychain. `--ad-hoc` explicitly opts into an ad-hoc build for isolated development
or unsigned distribution staging. Such rebuilds may require renewed Keychain
approval. Local builds remain development builds and are not notarized or
published by this command.

Moving from an ad-hoc build to a Developer ID signed build switches to the
installed API credential, port preferences and private OpenWork socket namespace.
You may need one access approval and to update copied API keys in connected apps.
Open **Connections → Apps**, choose **Manage** for OpenWork, then **Reconnect** to
repair the Stark-owned public connection entry. If the app is not set up, choose
**Connect** instead. Quit OpenWork completely and reopen it when instructed.
Guided Setup also offers **Reconnect** on the Open OpenWork and Ready steps
for an authorized connection, including when OpenWork reports a rejected API key.
It refreshes the managed connector settings and saves approval again. Quit
OpenWork completely and reopen it after reconnecting, even if its saved settings
already match: its running engine may still hold stale connection state. Select
Stark and the running model in OpenWork afterward.
The bundled connector installs its request authentication hook before publishing
Stark's models, including when OpenCode defers provider transforms during startup.
The first request does not need to wait for a background connection poll.
Build provenance records are described in [PROVENANCE.md](PROVENANCE.md).

Local apps display **Development build NUMBER**. The build number is a UTC
timestamp with microseconds by default; `--build-number NUMBER` supplies a unique
positive decimal identifier of up to 20 digits instead. Local apps use
`CFBundleShortVersionString=0.0.0`, store their build number in `CFBundleVersion`
and mark `StarkBuildChannel=development`. `--version` records an archive identifier
in provenance and does not assign a release version to the app. Use identifiers
such as `0.0.0-dev.COMMIT.BUILD` for local archives. Reserve major, minor and patch
versions for the [release workflow](../../docs/RELEASING.md).

## App updates

Stark checks stable release metadata automatically at launch and about once an hour while it runs. Opening the menu also checks when the last metadata check was more than 15 minutes ago. These background checks stay quiet when the network is unavailable. When it discovers a newer release, Stark automatically downloads and verifies the DMG in the background, then shows that the update is ready. Choose **Upgrade to VERSION** to install it: Stark stages the replacement, quits, stops active requests, clears the in-memory conversation, swaps the bundles and relaunches. Installation and restart always wait for this click. Models, preferences and the Keychain API credential are kept, while the active chat ends. The CLI is not updated by the app.

The verified candidate is reused for matching release metadata during the current app session. A changed or withdrawn release discards the previous candidate. A temporary metadata-check failure keeps the previously verified download available. Download or verification failures appear in the menu; choosing **Upgrade to VERSION** retries preparation before installing. Missing temporary files also trigger a fresh download. Normal quit removes an unused prepared candidate; restarting Stark downloads an available update again.

Development builds always offer the latest compatible stable release, including
when their archive identifier or old local version is numerically higher. Build
numbers do not participate in stable update ordering. Published installations
continue to offer only newer release versions. The release signing path sets the
published app's version and stable channel; signing a local app with
`--sign-only` retains its development identity.

Before anything is installed, the updater requires: the release's versioned Apple silicon DMG with a SHA-256 digest published by GitHub, a `codesign` verification of the disk image and the complete app against Stark's Developer ID team (`S5BT2PRN5B`), a passing Gatekeeper assessment, and a matching bundle identifier, version, architecture and minimum macOS. The candidate's signature and metadata are checked again when Upgrade is chosen, before executing its bundled Python. A private helper runs from the verified candidate's bundled Python, locks the destination, waits for the old app to exit, re-verifies the staged copy, exchanges the bundles atomically and relaunches. Detected failures restore the old app. Background check failures remain quiet; errors from a requested upgrade appear in the dropdown with a retry action.

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
Hugging Face checkpoint is downloaded and converted. If the source architecture
has no implemented converter, Add from Source stops before downloading weights
and explains: **This model architecture has no implemented converter.** This
means the selected conversion path is unsupported; unavailable or inconclusive
metadata has a separate explanation. Supported models are governed by the
[converter](../../tools/converter/README.md). Conversion support alone does not
guarantee that Stark can render a model's chat template or parse its function
tool calls. The third tab shows
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

**Installed** shows compact rows with a readable model name, quantization, file size,
repo link, **Run** or **Stop**, and removal button. Names may wrap. Recognized
variants such as **Lily q4** and **MLX-Serve** appear below the name.
Recognized generated revision suffixes and packaging labels are omitted from the displayed
name; known precision remains in its own column. Verified source names are used
when they match the saved name after removing packaging and revision suffixes.
Custom aliases retain their words. Hover a row for the full saved name;
search accepts both the readable and saved names. **Run** selects
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
**GLM-5.3 Flash** offers the pinned `UD-Q2_K_XL` GGUF checkpoint (108.72 GB of
weights). Its download uses native preparation, preserves quantization and checks
the prepared source before adding it to **Installed**. It supports text chat;
image input and Decision mode are unavailable. Download progress remains
indeterminate until the stored weight total is verified.
**Qwen3.8 Flash Next** includes the pinned Lily Q4-XL source through the typed
text converter and the pinned `pipenetwork` mixed 4/8-bit source through native
preparation. The mixed source preserves its original quantized storage and uses
the same preparation and source-identity checks as the native GLM download.
Inside a family, the tabs at the top switch between **All models** and each
available quantization or precision, such as **4-bit**, **8-bit** and **BF16**.
When the labels exceed the available width, a dropdown offers the same filters.
Floating-point badges describe the app's output weight precision: supported
FP16 and FP32 source checkpoints appear as **BF16** because the app converts
their weights to BF16. Qwen3.8 Flash Next currently offers the implemented
quantized import paths.
Search narrows the selected tab and sorting applies to its matching rows.
Entering another family starts at **All models**.
Catalog cards use a compact heading and a row of quantization, parameter count
and weight-file size. Unavailable metadata is omitted. Quantization labels stay
on one line, and installation controls move below the details when space is
limited. The publisher link on a single-variant card, or the source icon on each
row of a multi-variant card, opens its exact Hugging Face repository, including
the publisher of a quantized export.
Each row shows its text or text-and-image capability. Downloads automatically
include image support when available for that artifact. Sort
the family overview by name A–Z or Z–A; family detail sorts by name in either
direction or model size ascending or descending. Unknown numeric values stay last
in either direction. Download and preparation progress
stay in the browser. A completed download appears in **Installed** without
selecting it; choose it and **Run** when ready. Some image installations stop
the running model for their required local checks.

The catalogue starts with known sources, then follows the family's Hugging Face
architecture search through successive pages. Each request checks up to 30
distinct candidates; there is no fixed total result ceiling. **Load more models**
continues discovery through all remaining pages automatically, preserving rows
already loaded. Pages that add no new models are checked until a new result
appears, discovery ends, or a request fails. An initially empty catalogue uses
the same process. If a later page fails, partial results remain visible and the
failed continuation can be retried. Search and quantization filters apply to
models loaded so far. Results appear in a bordered, scrollable area so long
catalogues remain navigable.
Compatible quantizations from one publisher can share a card across model sizes;
Base, Instruct, Thinking, context-length variants and fine-tunes remain distinct.
Selected registered quantized exports may join their registered base model
across publishers. Size and download bytes appear when source metadata provides
them. Browsing reads metadata only and offers the formats implemented by Stark;
it does not establish complete tensor compatibility or successful execution.
Stark checks the selected model again before downloading weights and uses that
row's exact source version. **Back** or
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

Chat mode shows the conversation; send with the button, Return or Command-Return. Shift-Return inserts a newline, as shown below the message box. **New Chat** clears the conversation and the worker's context while keeping the model loaded. Failed or cancelled messages return to the composer. Replies appear when generation completes; the app does not stream. Decision mode takes a context, a question and 2 to 16 comma-separated answers, and requires a template that exposes a direct-answer position; otherwise it reports a compatibility error before scoring.

**Enable image input when converting** opts into the supported Qwen3.5 vision path. It accepts the pinned reference checkpoint and compatible full merged BF16 dense models/fine-tunes. The version-2 family derives supported decoder and tower dimensions from validated configuration within fixed operation, indexing and storage bounds; previously converted version-1 0.8B artifacts remain readable. Converted vision folders are detected automatically. For a schema-3 family installation, Stark stops the running model and checks text/image execution, vocabulary accounting, recovery and cleanup before adding the model to the library. Run repeats native admission and uses a matching private receipt or checks again. Available-memory preflight includes the model's geometry-dependent check state and scratch allowance. Unsupported graphs, LoRA-only repositories, quantized vision weights and unrecognized metadata fail admission. See the [family contract](../../docs/VISION_FAMILY_CONTRACT.md).

Passing host evidence enables **Attach Image** in Chat mode. The composer and Connections distinguish **locally checked** models from **reference verified** models; a local pass does not prove visual accuracy or parity with the official model. Choose, drop, or paste one PNG/JPEG up to 4 MiB, preview it, or remove it before sending. PNG/JPEG paste representations are supported; TIFF-only clipboard images are rejected. The original bytes feed the common native image pipeline. One image stays in memory and is included with every turn; choose **New Chat** to replace it. Image loading can be cancelled; Cancel during generation stops the model. Stop, model/mode changes, advanced-option changes and quit clear the image. Recoverable errors restore the draft image.

Images briefly use opaque files in per-process owner-only temporary directories; claimed files are unlinked before decoding and normal completion/Stop/quit cleans unused files. Image bytes, paths and filenames are never included in the control messages or logs. Hard-killing both processes before a file is claimed can leave a private file.

The retained Qwen3.8 Flash Next (Lily) processor accepts up to 32 million source pixels, with each side at most 16,384 pixels, and processes up to 16,777,216 pixels (65,536 patches / 16,384 image tokens). Larger source images are resized by the original model processor. Actual execution also requires enough available memory and context for the image, conversation and output. Other model families retain their own limits. This ceiling does not establish visual accuracy or latency at the maximum resolution.

**Advanced Options** in the playground window sets the system prompt, context size, output limit and thinking mode. Context defaults to the loaded model's configured maximum, including contexts above 16,384 tokens, and follows that maximum when switching models. Output defaults to automatic: after tokenization, generation may use the remaining finite context within available memory until EOS or another termination condition. An explicit output limit must fit beside the actual input. Thinking is off by default. A custom context remains in memory for the app session and is bounded by each loaded model; returning to a larger model restores that custom value. **Apply and New Chat** starts a fresh conversation with the model loaded; **Restore Defaults** fills in the model's maximum context and the other defaults for review. Applying that maximum returns context to following the model. Settings carry across New Chat and Stop and Run. When the backend declares no finite maximum, Advanced Options identifies the value as a protocol conversation budget. Auto asks the native worker for a memory-bounded execution allowance after tokenization; no additional setting is required. A custom conversation budget remains an upper bound. Connections shows the model context as not declared; text requests allocate only their input plus the chosen output budget.

For GLM-5.3, **Maximum reasoning effort** selects Low effort when off and Max effort when on; the model's reviewed template always reasons. The API also accepts an explicit Low, High or Max `reasoning_effort`. For boolean templates such as Qwen, Thinking controls `enable_thinking`. Replies retain the raw Playground display, while assistant history stores reasoning separately from the visible answer for subsequent turns.

When a reply exhausts an explicit output limit, the Playground points to **Auto** in Advanced Options. A context termination points to the current context setting and a new chat. Automatic generation can stop at the available context or execution budget, including memory-based allowances for models with a declared context maximum; the notice does not claim that clearing the conversation is always required.

The readout box below **Manage Models** and at the top of the playground shows input and output throughput, the first-token time, the prompt token count and the generation time. While a request runs, from the playground or a connected application such as OpenWork, the box shows that request; afterwards it keeps the last request's values. Input throughput is a first-token wall-clock proxy that includes native setup; output throughput counts sampling after the first token, and short replies may not produce an output rate. When the worker reports a reused prompt prefix, the prompt line also shows how many tokens were reused rather than processed again. A dash means no measurement is available. Every row stays in place, so the menu keeps its size as requests start and finish. See the [timing contract](../../docs/GENERATION.md#generate).

While a request runs, the time row shows elapsed time against an estimated total, such as `1.3 s of ~8.7 s`, with a progress bar. The estimate is the middle of a range that updates from live token timings and recent successful requests for the same model session and options. A new session first learns its speed and shows elapsed time alone, as does a request that runs past its estimate. Response length and system load can change the estimate. These timings start when the worker begins generation; loading, queueing and prompt preparation are separate. Live values contain counts and timings only, never request or response content. Estimates and calibration counts stay in memory, and replies still appear when generation finishes.

Supported models also show **Multi-token prediction (MTP)** in **Advanced
Options**. Gemma and full-BF16 Qwen3.5-0.8B default off because MTP was slower
in the measured runs. Other supported models retain their existing default on.
Turning it on uses the running model's recommended draft count.
**Apply and New Chat** saves the on/off choice separately for that model across
stops, model switches and app restarts. Existing models with no saved choice
use the model-specific default; an explicit saved choice is preserved.
**Restore Defaults** selects that model's default for review. Image requests
keep MTP off while preserving the text preference.
Connected apps inherit this setting unless they override `mtp` for a request.

## Beginner and Advanced setup

The interface picker offers **Beginner** and **Advanced**. On first launch,
Stark waits for model inventory before choosing a default: an empty library
opens Beginner, while a library with installed models opens Advanced. The
choice is saved separately from the **Chat** or **Decision** playground mode.
Changing this picker in the main menu updates the menu without opening a window.
Use **Guided setup** or **Manage Models** to open the selected interface.
The Playground omits the Beginner/Advanced picker; use the main menu to change interfaces.
Advanced retains the existing Manage Models, Model Catalog, Add from Source,
Playground, Connections and chat controls.

Beginner's **Guided Steps** flow asks what you want help with and whether to
prioritize a balance of capacity and memory, a lighter model, or more model
capacity. Recommendations show a simple model name with the specific installed
or source variant underneath, and a **View on Hugging Face** link to the source
repository. The same identity remains visible during setup and connection.
It reads the native Metal device name, unified-memory support and
recommended working-set budget, plus CPU count, physical and available memory,
and free space in the library and download cache. GPU and system memory share
the same physical memory; the algorithm never adds their budgets together.

**Explain text** covers summaries, notes and explanations. **Work with files**
is a separate task for reading or editing a folder through OpenWork, allowing
the guide to apply its tool-use benchmark references to that choice.

The guide checks a bounded window of candidates in each supported catalog
family using metadata only. Every candidate retains its exact source revision
and quantization. A new catalog choice needs a complete memory plan for its
attention or recurrent layout, at least 32,768 tokens of published context, a usable chat template and a
supported tool parser for OpenWork. These are separate capabilities: conversion
handles recognized architecture/configuration and weight layouts; chat requires
a template Stark can load and render; tool calling additionally requires a
reviewed template wire format with a matching parser. Tool capability comes from
the loaded artifact's template/parser check. Converting a model does not add a
parser or enable tools. The official Qwen2.5-Coder base checkpoints are excluded
from Guided Setup: their tokenizers include chat templates, but their
[source cards](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B) identify pretrained completion models and advise against conversation
without post-training. Their Instruct variants remain eligible. This exclusion
does not establish instruction tuning or task competence for every other source.
The catalog planner covers dense attention and Qwen3.5-family DeltaNet layouts,
including their recurrent state and declared MTP head. Fixed native profiles
cover Flint INT4, GLM Flash UD-Q2_K_XL and the three supported Qwen Flash Next
exports. They retain separate source, quantization, template and preparation
bindings. Other layouts remain available through Advanced. Catalog compatibility describes metadata
support; it does not inherit another model's native qualification.

Installed dense text artifacts with supported attention geometry and the pinned
Lily q4-xl Qwen3.8 Flash Next artifact also have local planners. They enter the
comparison even when online discovery fails or misses their source revision.

Dense memory plans include weights, both full-context BF16 KV buffers, attention
score scratch and a workspace allowance. DeltaNet also includes F32 recurrent,
convolution, prefix-snapshot and MTP verification allowances. Native profiles
use their backend's request inventory and a loading/workspace allowance. All
plans must leave 2 GiB of physical
memory and use no more than 85% of Metal's recommended working-set budget.
For the pinned Lily and MLX-Serve text preparations, the system plan
uses nonpaged payload, the full request inventory, 1 GiB active paging and an
8 GiB host reserve: 87 GiB for Lily and 88 GiB for Serve at 32K context. Other artifacts retain full-payload
accounting. The recipe-bound estimate applies before download on a fresh Mac;
installed copies additionally require exact artifact validation. A changed
local copy falls back to conservative accounting in both recommendations and
startup. The GPU estimate includes row staging rather than the entire
n-gram table. Full accounting and its bounds are documented in
[recommendation evidence](../../docs/MODEL_RECOMMENDATIONS.md). GLM and the
original mixed export use conservative full-weight GPU estimates, which can
exclude them from the guide on a 128 GiB Mac despite their Advanced load path.
Recommendations and startup share the current available-memory check; new
downloads additionally need enough space for
source and converted copies plus setup reserves. Native preparation downloads
into the library volume and publishes by rename, so it needs one weight copy
plus metadata and the preparation helper's 4 GiB reserve. It does not require a
second weight-sized copy in the Hugging Face cache. Starting rechecks available
memory after releasing the previous worker. A running model's memory is never
assumed to be reclaimable during recommendations.

Fallback ranking first prefers comfortable memory headroom. **Balanced** next
prefers source task tags when available, then scales its memory penalty to this Mac: its
target is half the smaller of physical memory minus 2 GiB and 85% of Metal's
budget. It ranks parameter capacity divided by
`1 + (estimated runtime memory / target)^2`, so unused memory on a larger Mac
reduces the penalty on larger candidates. **Lighter model** prefers lower weight
traffic; and
**More model capacity** prefers higher parameter counts. Source task tags break
otherwise equal weight/capacity comparisons for these explicit priorities. These
are documented fallback heuristics, with no measured tokens-per-second or Stark artifact quality score. Unknown
parameter counts receive no capacity credit. Installed copies break otherwise
equal ties. The guide offers up to three choices and explains each choice's
resource fit and evidence. CPU count and GPU name describe the machine; core
count alone does not establish inference speed.

Native parameter counts are source-card architecture metadata, not packed
tensor slot counts. MoE total parameter count does not describe active compute,
memory fit or task quality. Balanced also compares published task references:
LiveCodeBench for coding, BFCL for file/tool tasks, and IFEval for instruction
following in text/everyday/unsure. For Coding + Balanced, models covered by one
selected coding cohort come first after hard resource admission, ordered by
score; models without that comparable evidence follow in fallback order. One
eligible scored parent is enough, so an unscored model cannot retain first
place through the memory heuristic. Missing scores remain unknown. Other
tasks reorder scored slots within each memory-headroom tier and preserve
unknown positions. Lighter model and More model capacity preserve their explicit
resource priorities. Beginner details show estimated memory, a short benchmark
score with a published-model caveat, and a link to the evaluated model's Hugging Face page. Technical provenance,
revisions, harness settings and validation notes remain in backend reports and
evidence records. Existing models show no new setup-space requirement.
Public parent scores do not qualify Stark exports.
The [recommendation evidence policy](../../docs/MODEL_RECOMMENDATIONS.md) describes
the imported coverage, exact comparison policy and future artifact evaluation
requirements. MaxxBench remains deferred until model-specific runs exist.

Changing the task or priority invalidates the previous recommendation. A retry
updates the visible priority, and keeping the current model restores the
previous selection.

Discovery runs separately from generation, with three concurrent family lookups
and a 65-second overall process limit. Successful nonempty catalog scans are
cached only in memory for ten minutes; partial and failed family lookups retry
on the next check. A failed source lookup preserves successful candidates while
marking the search incomplete.
Resource checks are fresh on every recommendation,
installation and start. A failed search compares locally planned installed
models and recorded starter recipes that still meet the hardware checks.
Identical installed artifacts share one offer;
choosing another model excludes every registered copy of the current artifact.

The flow shows download and preparation as separate stages. After
installation, you must explicitly start the model; when another model is
running, it remains installed and available until you explicitly choose to
switch. Stark clears the current chat only after the selected model reports
ready.

The fallback starting choices are **Qwen3-4B-4bit** and **Qwen3-0.6B**, each
with source-bound native chat evidence. Both checked OpenWork recipes require
at least 16 GiB of physical memory and use a 32,768-token Playground/API
budget. The stock OpenWork initial prompt has been observed at about 17,000 to
19,000 tokens. A 16,384-token budget failed historical client runs, and the
observed prefixes also exceed a 4,096-token budget.
The guided API budget is separate from each model's trained/native context
capacity; Advanced keeps its existing raw context controls. Memory planning
estimates are 11 GiB for the 4B recipe and 9 GiB for the 0.6B recipe, not
per-Mac fit guarantees. On an 8 GiB Mac, neither fallback OpenWork starter is
eligible. The broader catalog uses model-specific memory plans rather than
that fixed minimum. When no candidate can be planned within the hardware
budget, Beginner offers the existing model browser in Advanced.
Recommendations are conservative planning estimates; task quality, speed and
general OpenWork competence still require artifact-specific validation.

After the model is running, Beginner can guide setup of the official OpenWork
app. **Download OpenWork** opens its official download page in a browser; the
user installs the app in Applications. Detection advances **Get OpenWork** to
**Connect**, which offers the existing managed connection authorization. Beginner connection setup gracefully reopens OpenWork to load the managed
connector, then uses its authenticated loopback UI-control bridge to prepare a
new empty task in the workspace OpenWork already selected (its default or last
workspace). No folder picker or workspace/model confirmation is required in Stark.
The handoff checks the workspace catalog for the exact running Stark API model,
creates a task, binds it with `session.set_model`, and verifies the saved binding
and displayed task before advancing automatically to **Ready**. It never sends a
prompt or runs tools. OpenWork retains its folder and tool permission controls.

The handoff targets the newly prepared task; it does not change OpenWork’s global
or workspace defaults for later tasks or rebind existing conversations. Switching
models through Beginner prepares another task. Readiness is scoped to the Stark
API session, verified OpenWork process, connector identity, model and context
budget. Cancellation, revocation, model changes and process restarts invalidate
stale handoff results. Transient startup failures retry up to ten times; **Try
again** retries setup without asking the user to confirm choices. A created task
is reused on retry to avoid duplicate tasks after a binding failure. The bridge
must support discovery version 2, `models.list`, `session.create_task` and
`session.set_model` (present in the inspected official OpenWork 0.18.66 release).
Older or incompatible clients remain in setup with recovery guidance.

The ready check verifies configuration and connector metadata; it does not prove
that OpenWork can perform a task or that the selected model will perform it well.

After setup, **Try a different model** is available from Beginner chat and the
ready step. Choosing or preparing an alternative keeps the current running
model, OpenWork workspace and authorization. **Cancel new setup** cancels only
that candidate operation; it does not remove an installed model. Switching is
explicit. The current chat is cleared only after the selected model reports
ready, and the previous model remains installed.

## Connect another application

Select **Connections** in the menu or playground window. The shared model header identifies the model running in Chat mode and its current availability. **Apps** groups OpenWork and OpenCode Desktop in one list, with separate connection status and controls for each app. Choose **Connect** to set one up, or **Manage** to reveal its **Reconnect** and **Disconnect** controls. A pending OpenCode engine keeps its own **Allow** and **Deny** prompt. API startup and Keychain recovery messages remain visible when setup needs attention.

For other applications, select **Connect your own app**, choose an OpenAI-compatible provider in the client and copy the **Base URL**, **API key** and **Model ID** individually. **Advanced settings** contains the saved listening port, API retry and key-regeneration controls, **Copy All as JSON**, **Copy curl Example** and API capability details. Model IDs use the Hugging Face repository ID (`owner/model`) for Hugging Face installs and the model name for local installs. Existing installations expose these names automatically.

The model picker under **Connect your own app** chooses which model ID to inspect and include in the JSON. It does not switch the running model or the model published to managed apps. Choose **Chat** mode and use **Run** in Stark to serve that model. The JSON includes the selected model, the running model, whether they match, running-model limits, endpoints, supported capabilities and usage notes. It includes the API key even when the key is hidden in the window. Treat that JSON as a credential; paste it only into trusted client settings. **Copy curl Example** copies a streaming request that prompts for the key with terminal echo disabled and passes it to curl over standard input. The example contains no key and is also included in the JSON when a model is running.

### Connect OpenWork without copying a key

For the official OpenWork macOS app, choose **Connect** on the OpenWork row under **Connections → Apps**. Open OpenWork with a local workspace first. Stark installs its bundled connector into the native OpenCode global configuration directory identified from OpenWork's running engine. The configuration stores a local connector path, owner-only socket path, expected loopback Base URL and an ownership marker. It contains no API key or per-request capability. If Stark changes the configuration, reopen OpenWork and select **Stark** in its model picker.

If OpenWork settings already define a manual provider with ID `stark`, remove only that provider entry before connecting and keep your other provider entries. Stark does not migrate existing client credentials.

Stark approves the official OpenWork app identity (bundle ID `com.differentai.openwork`, Team ID `F5DJWB4CCV`) and its signed `opencode2` engine sidecar. The API master key stays in Stark's Keychain. A separate Keychain approval record controls whether OpenWork can receive a renewable 120-second bearer capability over Stark's private local socket. Stark associates the grant with the verified sidecar's process identity and current API session, and renews the same token when requested after 30 seconds. The connector keeps it in memory and places it in the outgoing Authorization header. The bearer is transferable if exposed; process identity gates issuance in Stark but does not bind later HTTP use to that process. OpenWork plugins share the authorized app process and may inspect request data or the bearer in memory, so only install plugins you trust.

Choose **Manage → Disconnect** on the OpenWork row to disable the approval and revoke its current-session capabilities, cancelling requests already admitted with those grants. A connected OpenCode Desktop keeps its own grants. An expired 120-second lease does not cancel a request that was already admitted. To reconnect after disconnecting, select **Connect** again. If the Stark API port changes, choose **Manage → Reconnect** so the connector's expected Base URL is updated, then reopen OpenWork. Regenerating the master key affects clients configured with the copied key; managed OpenWork capabilities refresh through Stark. The separately copied API key remains available under **Connect your own app** for other native clients; OpenWork's managed connector does not require copying it.

The stock helperless connection has narrow desktop validation with official OpenWork 0.18.63 / OpenCode 2.0.26 and the pinned Qwen3-4B model. Chat, follow-up and lifecycle recovery behaved as expected, while two bounded read-tool tasks did not fully meet their requested outputs. This is connection and recovery evidence, not a general agent-competence claim. See the [managed-connection validation record](../../docs/archive/runs/2026-10-31-openwork-managed-connection.md).

### Connect OpenCode Desktop without copying a key

For the official OpenCode Desktop macOS app (2.2.5 or later), choose **Connect** on the OpenCode Desktop row under **Connections → Apps**. Stark opens OpenCode if needed, waits for its background engine and installs the same bundled connector entry used for OpenWork into the global configuration that engine reads. OpenWork and OpenCode can share that file; connecting one leaves the other's entry unchanged. Select **Stark** in OpenCode's model picker. If OpenCode settings already define a manual provider with ID `stark`, remove only that provider entry first.

OpenCode Desktop runs a signed copy of its `opencode-cli` engine from `~/Library/Application Support/ai.opencode.desktop/cli/` as a detached background service that keeps running after OpenCode quits. Stark therefore approves each engine instance once. **Connect** approves the engine it verified during setup. When a new engine starts, for example after an OpenCode update or a login, Connections shows **Allow** and **Deny** for it, and the engine gets no capability until you choose **Allow**. A verified OpenCode Desktop app (Team ID `5NZ4Q7NXJ4`) must be running for its engine to connect, and the engine must run as Desktop's background service with Desktop connected to it. The engine's signature, staging location and launch environment are checked on every socket connection.

The approved engine is shared: a terminal `opencode` session that reuses the same background service can also use Stark through it, and plugins in that engine share its process. A standalone or Homebrew `opencode` that runs its own engine is not managed; configure it with the copied API key. **Disconnect** revokes only OpenCode's capabilities and clears its approved engines; OpenWork stays connected. Stark normally appears in OpenCode's model picker right after **Connect**, because the running engine reloads its configuration. If it does not, quit OpenCode, stop its background engine with `opencode service stop`, reopen OpenCode and choose **Allow**. See [the local API reference](../../docs/LOCAL_API.md#managed-opencode-desktop-connection) for the full peer rule.

The server starts with Stark and binds only to `127.0.0.1`. Under **Connect your own app → Advanced settings**, **Saved port** defaults to 8000 and persists across launches; Stark never silently chooses another port. If the port is occupied, Connections explains the error. Close the conflicting app and choose **Retry API**, or enter another port and choose **Apply Port**, then update connected apps. Changing to an occupied port keeps an existing listener running at the displayed Base URL; the saved port remains the requested one for the next retry or launch. API failures leave the playground usable.

The Swift app reads its 256-bit bearer credential from macOS Keychain and sends it to the backend only over the private pipe. The key is created only when Keychain conclusively reports it missing and is saved before use. Unlock a locked Keychain or grant Stark access in Keychain Access, then choose **Retry API**. Corrupt entries require explicit **Regenerate API Key…** recovery. Regeneration explains that apps configured with the copied API key need the replacement; managed OpenWork and OpenCode capabilities refresh through Stark. Storage failure preserves the working key. **Show** and **Copy** never rotate it.

The stable Keychain service is `com.starkengine.menubar.local-api`, with account `bearer-key-v1`, for the canonical bundle with a signing Team ID. Canonical ad-hoc packages and direct development executables use the separate `com.starkengine.menubar.local-api.development` service. A noncanonical test bundle adds its sanitized bundle identifier to that development service, isolating packaged validation. These namespaces retain their credential across versions; switching between development and signed builds requires configuring that build's key. Port preferences are also separated by namespace. Keychain may require access approval when a signing identity changes; Stark reports access failure and does not create a substitute key.

Keychain reads and updates run off the main thread. If macOS waits for access approval, Connections explains the pending operation and the rest of Stark remains usable. Only one credential operation can be pending; retry, regeneration and port controls wait for it. After approving access in Keychain Access, quit and reopen Stark if the operation remains pending. Rebuilt ad-hoc development apps may need renewed approval for the existing item.

The key is for native applications on this Mac; browser-origin requests are rejected and remote devices are unsupported. Copies of the API key and connection JSON carry concealed and transient pasteboard markers so clipboard software that honors them can avoid retaining or syncing credentials. These markers are advisory; receiving applications and clipboard software that ignores them can still read the copied data. Use the explicit **Copy** button for the key; **Show** reveals it without enabling text selection.

The API offers authenticated health, capability, model listing and chat completion endpoints with streaming, temperature, top-p, top-k, min-p, seed and up to four stop strings. Workers with active passing image evidence accept inline PNG/JPEG image data URLs; the status document reports `image_verification`. Omitting an output limit uses the remaining finite context after tokenization, stopping on EOS or another termination condition. Explicit limits are honored; there is no separate 4,096-token ceiling or 512-token default. Context follows a declared finite model maximum, reported as `context_length` in `/v1/models`. Models without a declared finite maximum report `null` and `context_capacity_known: false`; automatic output uses a finite allowance selected from the loaded model’s state requirements and current memory headroom.

The shared resource budget automatically selects logical content admission from available RAM at startup with a 16 MiB floor and no fixed content ceiling. Encoded text JSON permits six times the content budget plus 2 MiB for framing. Image requests have 6 MiB of additional encoded-body allowance; compressed images remain limited to 4 MiB. Tool results use the aggregate content budget without a separate 256 KiB result cap. Normal app use requires no budget configuration. A developer can override the process-start budget with `STARK_REQUEST_BUDGET_MIB` for controlled runtime testing; the [budget contract](../../docs/WORKER_PROTOCOLS.md#resource-budget) describes that override and the framing, pending-body and sandbox allocation limits. Available memory, deadlines, sandbox protections and actual model context still apply; byte admission does not guarantee every encoding of a full token window or qualify full-window execution. Up to eight requests queue behind the active one for at most 30 seconds. Routes, parameters, error codes and limits are in the [local API contract](../../docs/LOCAL_API.md).

## Where the app keeps state

| Data | Location |
| --- | --- |
| Model registry, per-model MTP preferences and managed models | `~/Library/Application Support/Stark` (folders registered in place stay where they are) |
| Local image-check receipts | `~/Library/Application Support/Stark/image-checks`; private identity hashes and boolean results, reused only for an identical runtime |
| Selected model, playground mode, Beginner/Advanced interface, pending update workspace | `UserDefaults` for `com.starkengine.menubar` (`selectedModel`, `playgroundMode`, `starkInterfaceMode`, `appUpdateWorkspace`) |
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
