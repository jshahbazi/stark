# Public website deployment

## Public addresses

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
