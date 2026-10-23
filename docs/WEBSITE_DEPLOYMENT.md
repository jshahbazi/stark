# Public website deployment

## Initial domain setup snapshot (2026-10-21, 18:29 UTC)

| Address | Observed status on 2026-10-21 |
| --- | --- |
| [starkengine.app](https://starkengine.app/) | Active routing and TLS; anonymous GET returns HTTP 200. README and canonical URL use this address. |
| `starkengine.io` | DNS and hosting configured; new `.io` registry delegation is pending. Public DNS returns NXDOMAIN and TLS is not active yet. |
| [Generated Sites address](https://stark-inference.arcanepine32.chatgpt.site/) | Public, anonymous GET returns HTTP 200. |

No ChatGPT account is required. Both requested apex domains are attached to the same public Site; no redirect or `www` hostname is configured.

## Source and publication

- The presentation and measured runtime history were merged into `main` in [`2a772ef6`](https://github.com/jshahbazi/stark/commit/2a772ef68dc927ec029f380dd31697bd92682027). Existing converter and CI changes were preserved.
- Website source lives in [`website/`](../website/README.md); build with `node website/build.mjs`.
- Project: `appgprj_6ad8f59df1f4819185403a1dff22a0fb`, retained in `website/.openai/hosting.json`.
- Published Site source: `ec4196c41e5da0fb118a2080d2975a93ec1cf507` in the separate Sites source repository.
- Saved version: `appgprj_6ad8f59df1f4819185403a1dff22a0fb~appgver_db8d4ad645e0819184425d83b261d5ee` (version 5).
- Deployment: `appgdep_6ad904777b8c819197572c1ce4252550`, succeeded at `2026-10-21T18:29:17.857398+00:00`.
- Uploaded archive content hash: `sha256:493dc0c53b55624303ee9d101fd40c6db5050e797b4f1e3b26703dbfa211a9f7`.

Publication uses an ignored checkout containing only the website source. Build output, archives and short-lived credentials remain outside tracked files. No inherited release or signing workflow was enabled.

## Cloudflare configuration

The installed `cf` CLI added these Sites-provided records to each previously empty zone. All records have TTL 300 and are DNS-only (`proxied: false`).

| Type | Name in each zone | Value |
| --- | --- | --- |
| A | Apex | `162.159.143.30` |
| A | Apex | `172.66.3.26` |
| TXT | `_openai-site-verification` | Exact per-domain verification value returned by Sites |
| TXT | `_cf-custom-hostname` | Exact per-domain verification value returned by Sites |

Sites custom-domain IDs:

- `.app`: `appgdom_6ad902f5cadc8191b1e40481c3907186`.
- `.io`: `appgdom_6ad902f3ae4c819183c27e210497606c`.

Both zones use `dom.ns.cloudflare.com` and `melissa.ns.cloudflare.com`. At verification time, WHOIS reported the `.io` registration was created at `2026-10-21T18:19:30Z` through Cloudflare, with the expected nameservers and no client/server hold. Cloudflare's authoritative server returned the correct A records, while the parent `.io` nameserver returned NXDOMAIN. This is an external delegation propagation boundary; changing the working DNS records does not resolve it. Refresh the Sites custom-domain status after public delegation appears. Escalate persistent missing delegation to Cloudflare Registrar support.

Read-only checks:

```sh
cf dns records list --zone starkengine.app --per-page 100
cf dns records list --zone starkengine.io --per-page 100
dig @1.1.1.1 starkengine.io A
dig @a0.nic.io starkengine.io NS +norecurse
curl -q -I https://starkengine.app/
curl -q -I https://starkengine.io/
```

## Verification and remaining gates

The static build, JavaScript syntax, repository policy, 28 repository-policy tests, README relative links and whitespace checks passed. Anonymous HTTP checks confirmed the `.app` page and logo load; the served page preserves the source body, all six benchmark rows and the `.app` canonical URL. The logo matches the original asset hash. Browser verification confirmed the custom-domain page renders with the expected headline and collapsed methodology notes.

The merge's [Security run](https://github.com/jshahbazi/stark/actions/runs/39652388275) reproduced the existing 12 blocking dependency advisories in Rust/Python lockfiles. [CI](https://github.com/jshahbazi/stark/actions/runs/39652388227) was still running for scorer and inherited dependency compatibility checks when this receipt was prepared; repository, TypeScript binding and both process-lifecycle jobs had passed. These checks are separate from successful website publication. Native benchmarks were not rerun for domain configuration.

## Benchmark expansion (2026-10-21, 18:44 UTC)

The public site now includes three comparison views: the Oct-20 Stark/MLX-LM follow-up, the Oct-19 multi-engine generation pilot and JevK5 BF16 scoring. The [multi-engine view](https://starkengine.app/#bench-engines) opens directly through its fragment URL. Its selector covers all six workloads for Stark, MLX-LM, vLLM Metal, llama.cpp and TensorFold with default drafting or drafting disabled.

The earlier Stark build stays paired with its original pilot results. The page identifies llama.cpp's different quantizer, MetalRT's unmeasured status and the low JevK5 public accuracy. vLLM's separate concurrency-eight diagnostic is excluded from the cross-engine chart. Methodology, excluded-run counts and quality details remain in closed disclosure controls and linked immutable reports.

- Stark website source commit: `d426e2697a96d5658e62da69cd972e159f17b6be` on `codex/site-engine-benchmarks`.
- Site source commit: `260b8347fcec4ada6f1e0ed6bfd549ef9afb2046`.
- Saved version 6: `appgprj_6ad8f59df1f4819185403a1dff22a0fb~appgver_71af78f860c48191a5b24758fd7b0082`.
- Deployment: `appgdep_6ad907f5a0b48191980f83102a5ab438`, succeeded at `2026-10-21T18:44:13.193734+00:00`.
- Uploaded archive content hash: `sha256:598c45f5267c5dfaa8380874fa48fc918cceefaa2f9c7202cb90a790055c38b4`.

Independent evidence review checked all 36 stock generation rates and eight scoring latencies against the retained report. Browser checks covered all six selector choices, tab click/ArrowRight/Home/End navigation, scoring disclosure, the existing relative-rate control and direct fragment navigation. All three panels had document width 390 at a 390-pixel viewport; desktop and mobile screenshots were inspected. Static build, JavaScript syntax, repository policy and whitespace checks passed. Anonymous GETs returned HTTP 200 for the public page and script, preserving the source body and exact script bytes. Native inference and benchmark acquisition were not rerun for this presentation update.

## Withdrawal of historical front-page comparisons (2026-10-21, 20:47 UTC)

The multi-engine expansion above was withdrawn from the marketing page because it used a pre-optimization Stark build and did not answer how the promoted build compares with the other engines. The historical reports and receipts remain unchanged. No old timing value was relabelled or replaced with a newer value from a different campaign. The public page again shows the newer Qwen3.5-2B Stark/MLX-LM comparison, with its original limitations.

- Corrective Stark source commit: `cd15f068` on `codex/site-engine-benchmarks`.
- Site source commit: `33c6727734db95828035d95b0dabc882a82624b9`.
- Saved version 7: `appgprj_6ad8f59df1f4819185403a1dff22a0fb~appgver_e67ea33e1978819182cfa02e6df8ddad`.
- Deployment: `appgdep_6ad924e201948191b7b949cd0ff1934e`, succeeded at `2026-10-21T20:47:35.494516+00:00`.
- Uploaded archive content hash: `sha256:493dc0c53b55624303ee9d101fd40c6db5050e797b4f1e3b26703dbfa211a9f7`.

An immediate rollback to saved version 5 preceded the source correction, using deployment `appgdep_6ad9242d35208191857bda11fc5d727f`. Anonymous HTTP and browser checks confirmed the older table was absent and the newer Stark rates were restored. A [separate fresh campaign protocol](experiments/current-engines-2026-10-21/PROTOCOL.md) was frozen before any new admission or measurement, to obtain contemporaneous results rather than reusing the historical pilot.

## Fresh multi-engine publication (2026-10-22 local time)

The user requested the complete fresh comparison and directed host-activity flags to be ignored for the website presentation. The public page now uses campaign 4 from the [Qwen3.5-2B comparison](QWEN35_ENGINE_COMPARISON_2026_10_22.md): five engines, six configurations and six performance workloads. No values from older campaigns were mixed into the chart. The 12.2% Stark/MLX-LM headline is the rounded observed aggregate from this same run.

The interface has an overall view, six workload selections, token-rate and MLX-relative modes, downloadable numeric results, and a compact methodology disclosure. It scopes results to Qwen3.5-2B on M5 Max at concurrency one, identifies llama.cpp's separate Q4_K_M quantizer, and links the full evidence. Internal host-guard jargon is absent from the page. The acquisition receipts and flags remain unchanged, and publication makes no clean-host or multi-model claim.

The existing signed v0.1.2 installer presentation was preserved from Site source b31a91786756c85c431a1ec5dda18d448ff1d33c (version 9), which was newer than the benchmark branch's website copy.

- Stark website source: fb6f101d on codex/current-engine-benchmark-resume.
- Public report source: b64daf27015d346761e369885ea320b55222b846.
- Site source: 3b11db0f1b8bdad3dbc44790bd5ff7b936a16f38.
- Saved version 10: appgprj_6ad8f59df1f4819185403a1dff22a0fb~appgver_96242c7c95588191ba9f06f73855212b.
- Deployment: appgdep_6adab7646cb081918dd6906ff467986b, succeeded at 2026-10-23T01:25:00.532450Z (2026-10-22 Pacific time).
- Archive hash: sha256:480bc3078739e54776279383e16587646c92923c644e5bf5483200ff3b80830d.
- Audience remains public.

The static build and JavaScript syntax check passed. All 42 exported rates match the observed summary exactly. Browser checks verified 84 displayed cells across seven workload choices and two modes; the 390-pixel layout had no document overflow. Installed command tabs remain functional. Production browser verification confirmed all six rows, the new headline and retained install section on starkengine.app.

During final domain checks, starkengine.io resolved but returned HTTP 404. Its old Sites custom-hostname record was still pending/active_redeploying. Refreshing that existing record returned active status with active TLS at 2026-10-23T01:28:27.346647Z; no DNS record was changed. This supersedes the earlier missing-delegation diagnosis for the current observation.

Anonymous HTTP rechecks after hostname refresh returned 200 for starkengine.io and its benchmark-results.json. Both public domains serve the same 42 rates, byte-for-byte equal to the tracked data file. The live comparison screenshot is retained privately in the task runtime. The owned local preview server was stopped after verification.
