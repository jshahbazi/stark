"""Check generated docs links, fragments and referenced local assets."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(path.read_text())

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for key in ("href", "src"):
            if attrs.get(key):
                self.links.append(attrs[key])


root = Path(__file__).resolve().parents[1] / "dist"
pages = {p: Page(p) for p in root.rglob("*.html")}
errors = []
checked = 0
for path, page in pages.items():
    if root / "docs" not in path.parents:
        continue
    base = "/" + str(path.relative_to(root))
    for link in page.links:
        url = urlsplit(urljoin(base, link))
        if url.scheme or url.netloc:
            continue
        target = root / unquote(url.path).lstrip("/")
        if target.is_dir():
            target /= "index.html"
        checked += 1
        if not target.is_file():
            errors.append(f"{path.relative_to(root)}: missing {link}")
        elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
            errors.append(f"{path.relative_to(root)}: missing fragment {link}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"Checked {checked} documentation links and assets.")
