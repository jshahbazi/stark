# Installed Stark user functional test — 2026-11-10

## Build and method

- Tested the installed macOS app through its native interface and OpenWork, rather than substituting CLI inference for the user flows.
- Stark: `~/Applications/Stark.app`, version 0.19.3, build `20261110142800359125`, source `cb4cea9f18e59960b8f29bf0331d8a75ac86fee0`, manifest reports a clean source tree.
- OpenWork: version 0.18.65. The active app was an existing copy under a prior local E2E worktree. An official 0.18.65 copy was also installed into `~/Applications` during setup. Its downloaded DMG SHA-256 matched the release asset: `08c9f51da435af21b77cada5f14b58006386e929a83f9f30ebb4cbddae79e2b1`.
- An older Stark 0.17.0 remains in `/Applications`. It was not the app used for the model tests.
- Model execution was local. Test files and a synthetic image were disposable local fixtures. No conversation transcripts, decision probabilities, token vectors, credentials, or exception dumps are included in this report.
- These are functional observations of specific artifacts. They do not establish general model quality, numerical parity, sustained performance, or exhaustive coverage of every input and failure condition.

## Installed artifacts at the start

| Model | Precision / displayed size | Source revision |
| --- | --- | --- |
| JevK5 | BF16 / 8.41 GB | `c4f7fdb3aeab5582336406e78d3bef11bf98833d` |
| Qwen3 4B | 4-bit / 2.26 GB | `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25` |
| Qwen3 Dense | BF16 / 1.19 GB | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Qwen3.8 Flash Next, MLX-Serve mixed | Mixed 4/8-bit / 107.3 GB | `788c07a8725642bccf13a0a3b8a21954ab84b4f2` |
| Qwen3.8 Flash Next, Lily | q4-xl / 106.91 GB | `0f94842e3682b49acace4daeb4cf2e65a6f94f83` |

## Playground observations

All five original models loaded and returned the expected result on the initial brief arithmetic check. Switching models cleared the Playground conversation. New Chat also cleared conversation content.

Additional checks:

- JevK5 and Qwen3 4B accepted follow-up conversation requests. Some extended requests were cancelled; these are not counted as completed recall tests.
- The Lily artifact accepted a PNG attachment and displayed its preview, format, size, and Remove Image control. It correctly identified the two shapes and colors in the synthetic fixture, then answered a follow-up about the image.
- Text-only models disabled Attach Image with explanatory help.
- Lily explicitly disallowed Decision mode with “Vision-language artifacts support Chat mode only.” This is a disclosed capability restriction.
- Advanced Options rejected an output limit of zero, accepted a short explicit limit, and displayed a limit termination notice when reached. Defaults were restored afterward.
- Decision mode disabled Run decision when only one answer was supplied. The static Result label appeared after execution for Qwen3 Dense, Qwen3 4B, JevK5, Flash Next mixed, and the temporary Qwen3 0.6B source artifact. For the four original text artifacts, switching models first cleared the preceding result. This verifies result rendering, not decision accuracy or calibration.
- Playground cancellation returned control and stopped the loaded worker in the observed case. Reloading a model recovered it.
- Installed-model search filtered to JevK5 correctly. Size sorting applied, and Name (A–Z) was restored.
- Beginner guided setup accepted a task selection and returned a recommendation with an explicit download/setup action. That extra download was not started; Advanced was restored.

## OpenWork observations

Each original model was selected from Stark's provider in OpenWork and exercised with a local file-tool task. File creation was checked independently on disk for exact content; an assistant's claim alone was not treated as proof.

| Model | Exact file creation | Completion / read-back |
| --- | --- | --- |
| JevK5 | Passed | Task finished; read activity observed |
| Qwen3 Dense | Passed | Task was manually stopped before complete read-back qualification |
| Qwen3 4B | Passed in an isolated fresh session | Task finished |
| Flash Next mixed | Passed | Remained active during repeated checks; manually stopped before completion |
| Flash Next Lily | Passed | Task finished; file-read activity observed |

Early overlapping Playground and OpenWork requests were cancelled and excluded from isolated tool-task conclusions. OpenWork Stop returned control. Stark's Disconnect and Reconnect controls both completed, and the connection was subsequently usable without copying an API key.

## Installation tests

### Gemma 4 with images

1. Selected `google/gemma-4-E4B-it` BF16, revision `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c`, from the Gemma 4 catalog. The row advertised Text and images. Download reached Complete at 16.02 GB; conversion failed with a generic message.
2. Entered the same Google repository in Add from Source and explicitly enabled image input. Download completed again; conversion failed with the same generic message.
3. Entered `mlx-community/gemma-4-e2b-it-4bit` with image input enabled. Stark rejected it before weight download because image input requires merged BF16 weights. The rejection was clear and actionable.
4. Selected the complete cached Google Gemma 4 E2B checkpoint, revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`, through Local Folder with image input enabled. Stark rejected it because the checkpoint's processing metadata was outside the implemented family.

No Gemma artifact reached Installed. Consequently Gemma generation, image understanding, OpenWork use, and removal of a successfully installed Gemma artifact could not be qualified. Final inventory and cleanup are recorded below.

### Hugging Face Add from Source

- A larger `Qwen/Qwen3-1.7B` download was cancelled to exercise cancellation and use a smaller independent source-install test.
- Entered the full URL for `mlx-community/Qwen3-0.6B-4bit`, with image input off. Download and conversion succeeded. Installed count increased from five to six, with the new artifact shown as Qwen3 0.6B, 4-bit, 335.6 MB.
- The new artifact loaded and passed the brief Playground arithmetic check. OpenWork discovered it. Further verification and cleanup are recorded below.
- Decision mode rendered a result. An isolated OpenWork file-tool task created the exact file contents, but did not finish the read-back task during the observation window and was stopped.

## Issues and limitations encountered

### 1. Advertised Gemma 4 E4B installation fails after downloading

The catalog's BF16 E4B Text and images entry and an explicit source installation both completed download but failed conversion. This blocks the requested Gemma user flow. Exact converter cause was not established; neither format support nor access failure should be inferred from the generic message.

Reproduction: Manage Models → Model Catalog → Gemma 4 → Google E4B BF16 → Download. The source-route reproduction uses Add from Source, the same repository, and image input enabled.

### 2. Conversion failure has insufficient diagnostic detail

The displayed message only asks the user to check model support, disk space, and Hugging Face access, even after download completes. It does not identify the failed conversion stage or a concrete recovery action. Read-only inspection of the installed converter launcher found child stdout/stderr discarded and nonzero/invalid output mapped to the generic installation failure. No raw exception logging is proposed; a bounded structured error category would make this diagnosable.

### 3. OpenWork model availability can require manual refresh

After a Stark mode/model transition, OpenWork showed its saved model selection as unavailable, with no current Stark model in the picker. Refresh repopulated the current model and allowed selection. Ordinary model switches also required selecting the new artifact ID. The tested recovery worked; automatic rediscovery was not reliable in that transition.

### 4. Tool-task completion needs artifact-specific qualification

Flash Next mixed wrote the requested file but remained active without completing the read-back task during repeated checks, until manually stopped. Qwen3 Dense and the temporary compact source artifact also created their files but were stopped before full task completion. Lily eventually completed. This establishes useful tool execution alongside incomplete or slow task completion; it does not isolate the cause to Stark, the connector, OpenWork, or model behavior. The large worker also occupied substantial shared memory during the mixed-model run, so this is not a controlled latency benchmark.

### 5. Simple answer quality differs between interfaces

JevK5 passed the initial Playground arithmetic check but returned an incorrect result in a fresh OpenWork arithmetic session. The newly installed compact Qwen artifact also failed the OpenWork arithmetic expectation while passing Playground. This is a user-visible quality observation, not a demonstrated engine or connector defect. System prompts, sampling, and host context were not matched.

### Environment findings

- Multiple Stark installations can make launch/version selection ambiguous.
- OpenWork launch discovery initially missed a prior E2E worktree copy. That copy was the active 0.18.65 app; copied apps with the same bundle ID caused launch targeting confusion.
- Chrome navigation to the official OpenWork download reported a client block. Aikido event inspection returned no events. The same official asset was downloaded separately with its published hash verified. This was a test-environment setup issue, not a Stark defect.

## Final verification and cleanup

- The temporary Qwen3 0.6B source model was removed using Stark's Move to Trash flow. The confirmation explicitly said files were recoverable. The entry disappeared and Installed returned to five.
- The final installed-model view contained the five original artifacts and no Gemma entry. No successfully installed Gemma model existed to delete. Existing shared Hugging Face source caches were preserved.
- Stark ended in Advanced interface, Chat mode, Qwen3 Dense selected, and Stopped. OpenWork had no active task.
- OpenWork Disconnect/Reconnect had restored authorization, and subsequent tool requests used it successfully.
- An accidental duplicate OpenWork copy in Downloads was moved to Trash. The official copy in `~/Applications` and verified DMG in Downloads remain. The original active E2E app was preserved.
- Disposable test files and the synthetic image remain under `/tmp`; OpenWork's test sessions remain in its local history. They are not committed to this repository.
- No product code changed, no PR was opened, and no production-data task was performed.

## Coverage limits

This run exercised both interfaces, all five original models, supported Decision paths, an installed vision artifact, managed connection lifecycle, source installation, catalog installation failure, cancellation, basic validation, search/sort, guided recommendation, and recoverable model removal. It did not qualify the blocked Gemma runtime, controlled cross-interface quality parity, full-context or endurance behavior, every sampling option, custom-client credential/port configuration, an app upgrade, or every malformed image/source case. Earlier overlapping requests and operator navigation mistakes are excluded from defect claims.

## Public source references

- [Compact source model](https://huggingface.co/mlx-community/Qwen3-0.6B-4bit)
- [Google Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it)
- [MLX Gemma source rejected for vision](https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit)
- [Official OpenWork 0.18.65 release](https://github.com/different-ai/openwork/releases/tag/v0.18.65)
