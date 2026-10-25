import { cpSync, mkdirSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const publicDir = fileURLToPath(new URL('./public/', import.meta.url));
const output = fileURLToPath(new URL('./dist/', import.meta.url));
rmSync(output, { recursive: true, force: true });
mkdirSync(output, { recursive: true });
cpSync(publicDir, output, { recursive: true });

console.log("Built static site in dist/");
