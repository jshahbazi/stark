"""Search tracked Stark research Markdown or check the research index."""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
INLINE_LINK = re.compile(
    r"\]\(\s*(?:<(?P<angle>[^>\n]+)>|(?P<plain>(?:[^\s()]|\([^()]*\))+))"
    r'(?:\s+["\'][^\n]*?["\'])?\s*\)'
)
DATED = re.compile(r"^\d{4}-\d{2}-\d{2}-.*\.md$")
EXPERIMENTS = Path("docs/model-experiments")
SOURCE_INDEX = EXPERIMENTS / "source-index.md"
CHECK_PAGES = {
    Path("docs/MODEL_RESEARCH.md"),
    SOURCE_INDEX,
    EXPERIMENTS / "branch-research.md",
    EXPERIMENTS / "branch-sources.md",
}


def git(*args, cwd=None):
    return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.PIPE)


def research_document(path):
    return path == Path("provenance/UPSTREAM.md") or (
        path.suffix == ".md"
        and path.is_relative_to("docs")
        and not path.is_relative_to("docs/brand")
    )


def match_document(path, lines, terms):
    title = path.stem
    for line in lines:
        heading = HEADING.match(line)
        if heading and heading[1] == "#":
            title = heading[2]
            break

    name = f"{title} {path}".casefold()
    name_hits = {term for term in terms if term in name}
    section_hits = set()
    body_hits = set()
    best_score = (0, 0)
    section = best_section = ""
    best_line = 0
    for number, line in enumerate(lines, 1):
        heading = HEADING.match(line)
        hits = {term for term in terms if term in line.casefold()}
        if heading and heading[1] != "#":
            section = heading[2]
            section_hits.update(hits)
            score = (2, len(hits))
        else:
            if heading:
                section = ""
            body_hits.update(hits)
            score = (1, len(hits))
        if hits - name_hits and score > best_score:
            best_score = score
            best_line = number
            best_section = section

    if name_hits | section_hits | body_hits == terms:
        return (
            -len(name_hits),
            -len(section_hits),
            path,
            title,
            best_line,
            best_section,
        )
    return None


def search(root, paths, terms, limit):
    matches = []
    for path in paths:
        candidate = root / path
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or not candidate.resolve().is_relative_to(root)
        ):
            continue
        match = match_document(
            path, candidate.read_text(encoding="utf-8").splitlines(), terms
        )
        if match:
            matches.append(match)
    for _, _, path, title, line, section in sorted(matches)[:limit]:
        line_ref = f":{line}" if line else ""
        suffix = f" ({section})" if section else ""
        print(f"{title}{suffix}  {path}{line_ref}")
    if not matches:
        print("No tracked Markdown documents match all search terms.")
        return 1
    if len(matches) > limit:
        print(
            f"Showing {limit} of {len(matches)} matching documents. Use --limit to see more."
        )
    return 0


def check(root, tracked):
    targets = set(tracked)
    for path in tracked:
        targets.update(path.parents)
    pages = CHECK_PAGES | {
        path
        for path in tracked
        if path.parent == EXPERIMENTS and path.match("speed-*.md")
    }
    expected = {
        root / path
        for path in tracked
        if (
            path.is_relative_to("docs/benchmarks")
            and path.suffix == ".md"
            and path != Path("docs/benchmarks/README.md")
        )
        or (path.is_relative_to(EXPERIMENTS) and DATED.fullmatch(path.name))
    }
    indexed = set()
    errors = []
    for page in sorted(pages):
        source = root / page
        if page not in tracked or not source.is_file():
            errors.append(f"{page}: missing tracked Markdown page")
            continue
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), 1
        ):
            for link in INLINE_LINK.finditer(line):
                url = urlsplit(link["angle"] or link["plain"])
                if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
                    continue
                target = (source.parent / unquote(url.path)).resolve()
                if (
                    not target.is_relative_to(root)
                    or target.relative_to(root) not in targets
                    or not target.exists()
                ):
                    errors.append(
                        f"{page}:{line_number}: missing tracked link target {url.path}"
                    )
                if page == SOURCE_INDEX:
                    indexed.add(target)
    if SOURCE_INDEX in tracked and (root / SOURCE_INDEX).is_file():
        for path in sorted(expected - indexed):
            errors.append(
                f"{SOURCE_INDEX}: missing direct link to {path.relative_to(root)}"
            )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        f"Checked {len(pages)} research pages and {len(expected)} benchmark/attempt links."
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "terms", nargs="*", help="case-insensitive literal terms, all required"
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="maximum search results (default 20)"
    )
    parser.add_argument(
        "--check", action="store_true", help="check report coverage and relative links"
    )
    args = parser.parse_args()
    if args.check and args.terms:
        parser.error("--check does not take search terms")
    if not args.check and (
        not args.terms or any(not term.strip() for term in args.terms)
    ):
        parser.error("provide search terms or --check")
    if args.limit < 1:
        parser.error("--limit must be positive")

    try:
        root = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
        tracked = {
            Path(p.decode()) for p in git("ls-files", "-z", cwd=root).split(b"\0") if p
        }
        if args.check:
            return check(root, tracked)
        paths = (path for path in tracked if research_document(path))
        return search(root, paths, {term.casefold() for term in args.terms}, args.limit)
    except (subprocess.CalledProcessError, OSError, UnicodeError) as error:
        print(f"Could not read the Stark worktree: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
