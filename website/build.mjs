import { cpSync, existsSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const publicDir = fileURLToPath(new URL('./public/', import.meta.url));
const output = fileURLToPath(new URL('./dist/', import.meta.url));
const root = fileURLToPath(new URL('./', import.meta.url));
const astro = fileURLToPath(new URL('./node_modules/astro/bin/astro.mjs', import.meta.url));
if (!existsSync(astro)) {
  throw new Error('Website dependencies are missing. Run npm ci before building.');
}
rmSync(output, { recursive: true, force: true });
mkdirSync(output, { recursive: true });
cpSync(publicDir, output, { recursive: true });

const result = spawnSync(process.execPath, [astro, 'build', '--root', 'docs-site'], {
  cwd: root,
  stdio: 'inherit',
  env: { ...process.env, ASTRO_TELEMETRY_DISABLED: '1' },
});
if (result.error || result.status !== 0) {
  throw new Error('Documentation build failed; do not publish dist/.', { cause: result.error });
}
for (const entry of readdirSync(new URL('./docs-site/src/content/docs/', import.meta.url))) {
  if (!entry.endsWith('.md')) continue;
  const slug = entry.slice(0, -3);
  const route = slug === 'index' ? 'index.html' : `${slug}/index.html`;
  if (!existsSync(new URL(`./dist/docs/${route}`, import.meta.url))) {
    throw new Error(`Documentation output is missing: ${route}`);
  }
}
console.log('Built complete website and documentation in dist/');
