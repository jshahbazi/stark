# 20261110-CATALOG-01 — catalogue family admission

This is a retrospective ledger entry registered on 2026-11-10. Its canonical
validation record is the
[family admission run](../archive/runs/2026-11-10-catalog-family-admission.md).

## Scope and identity

The compatibility work covers catalogue discovery and admission for the existing
generic converter families, including Gemma 4 and Qwen3.8, plus visibility of
GLM-5.3 Flash and Flint variants. Baseline is `fc4c3153`, implementation is
`e527dd5c`, and the review correction is recorded on
`codex/catalog-family-admission`. Exact checkpoint revisions, smoke artifact
hashes and the installed worker hash are bound in the canonical run record.
The corrected converter source hash is
`62a6a77c6fd1646a55a7d11338625b83a346f3989fa689158e707cc4c9e2200e`.

## Problem, controls and history

Gemma 4 quantized K=V exports passed catalogue metadata admission and failed
conversion. Alias ordering delayed other export types, unsupported variants
disappeared, and generic tensor layouts were checked after bulk downloads.
The change interleaves searches, retains reasons, and feeds pinned remote headers
to the production tensor planners. GLM and Flint retain exact native preparation
identities. Guided discovery continues through unavailable pages.

The prior full offline converter suite passed with 7,084 tests and 94 skips.
The model-bound Qwen3-0.6B conversion/load/generation smoke passed and retained
aggregate hashes. Live header checks admitted the Qwen3.8 4.7 bpw export and
rejected the 3.3 bpw expert packing with a specific reason.

Pre-PR review reproduced a composition failure in which missing or malformed
vision tensors caused a text-only download fallback. Two new cases failed and
the valid control passed before the correction. All three then passed, along
with 302 focused catalogue, planner, HTTP, native, desktop and guided tests.
The correction disables the whole row if any offered mode fails its tensor
plan and preserves its image mode and default selection.

## Decision and limits

The compatibility correction is adopted on the branch for PR review. Header
admission checks source storage. Duplicate tied-tensor byte equality is deferred
to full conversion. The source-backed planners retain that mandatory check.
Network deadlines are cooperative inside discovery, with hard cancellation
owned by the desktop child-process supervisor.

The changes affect catalogue discovery and shared storage planning. Existing
importers own precision and state policies. Cache, image-loaded text performance
and MTP numerical qualification are outside this catalogue-only run. Performance
was not measured. Full execution of the 4.7 bpw artifact is unverified. GLM and
Flint alternative artifacts require supported import paths before downloading.

## Revision binding follow-up

Follow-up `20261110-CATALOG-01-R2` reproduced a live E2B row using a newer
upstream revision than its supported image profile, causing the image assertion
to fail. Known Gemma image choices now bind their profiles' immutable revisions.
The canonical run records both revisions and the failed check. The final 303
focused tests and repeated live E2B/E4B image-mode checks passed. The follow-up
converter source hash is
`73200a85f5051a2e14d39a186c0a4797575d89e0c746c47011777c9f615ee244`.
