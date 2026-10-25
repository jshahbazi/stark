import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const publicDir = fileURLToPath(new URL('./public/', import.meta.url));
const output = fileURLToPath(new URL('./dist/', import.meta.url));
const releasePath = fileURLToPath(new URL('./release.json', import.meta.url));

// release.json is the single place to update when a new release is published.
// Its values replace {{release.*}} placeholders in the pages listed below.
const release = JSON.parse(readFileSync(releasePath, 'utf8'));
const { version, date, notes_url: notesUrl } = release;
if (!/^\d+\.\d+\.\d+$/.test(version)) {
  throw new Error(`release.json: version must be MAJOR.MINOR.PATCH, got ${JSON.stringify(version)}`);
}
if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
  throw new Error(`release.json: date must be YYYY-MM-DD, got ${JSON.stringify(date)}`);
}
const expectedNotesUrl = `https://github.com/jshahbazi/stark/releases/tag/v${version}`;
if (notesUrl !== expectedNotesUrl) {
  throw new Error(`release.json: notes_url must be ${expectedNotesUrl}, got ${JSON.stringify(notesUrl)}`);
}

const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
  'August', 'September', 'October', 'November', 'December'];
const [year, month, day] = date.split('-').map(Number);
if (month < 1 || month > 12 || day < 1 || day > 31) {
  throw new Error(`release.json: date ${date} is not a calendar date`);
}
const dateDisplay = `${day} ${months[month - 1]} ${year}`;

const escapeHtml = (value) => String(value)
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
const values = {
  'release.version': escapeHtml(version),
  'release.date': escapeHtml(date),
  'release.date_display': escapeHtml(dateDisplay),
  'release.notes_url': escapeHtml(notesUrl),
};

rmSync(output, { recursive: true, force: true });
mkdirSync(output, { recursive: true });
cpSync(publicDir, output, { recursive: true });

const templatedPages = ['index.html', '404.html'];
for (const page of templatedPages) {
  const path = `${output}${page}`;
  const source = readFileSync(path, 'utf8');
  const rendered = source.replace(/\{\{([a-z_.]+)\}\}/g, (placeholder, key) => {
    if (!(key in values)) {
      throw new Error(`${page}: unknown placeholder ${placeholder}`);
    }
    return values[key];
  });
  writeFileSync(path, rendered);
}

console.log(`Built static site in dist/ for release v${version} (${dateDisplay})`);
