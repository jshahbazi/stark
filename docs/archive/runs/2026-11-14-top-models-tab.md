# Top Models tab validation

Date: 2026-11-14. Base: `f579f099`.

Added Top Models between Installed and Model Catalog, using the ten approved
family aggregates from the llmbench shared-benchmark experiment. Each row
shows rank, family, relative index and scored-model count, and opens the
existing family browser. Search retains rank and is independent of catalog
search. Catalog browsing retains its existing sort controls and Back returns
to the selected tab. The initial tab remains Installed.

Checks completed on macOS:

- Repository policy check passed before and after the change:
  `uv run --locked --python 3.12.13 --script scripts/ci/check_repository.py`.
- Full production Swift source typecheck passed using `xcrun swiftc -typecheck`,
  targeting `arm64-apple-macos14.0` with AppKit, SwiftUI and Metal.
- Controller harness passed all 141 checks:
  `uv run --project tools/converter --locked --python 3.13.13 python scripts/test_menubar_controller.py`.
  Existing navigation assertions now cover four-tab order and the title;
  family-inventory assertions cover ranked IDs against backend family IDs,
  filtered rank preservation, empty search and no matches.
- Native hidden-window harness passed all 8 checks:
  `uv run --project tools/converter --locked --python 3.13.13 python scripts/test_menubar_windows.py`.
- Changed task files pass `git diff --check`. An existing unrelated website
  change has trailing whitespace and was left untouched.

The app was not installed or relaunched. The window harness uses hidden test
windows and synthetic data; it does not establish a visible screenshot review
or a live download. No inference execution path changed. Ranking methodology,
source hashes and attribution are retained separately in
[the benchmark snapshot](../../benchmarks/2026-11-14-catalog-coding-ranking.md).
