const assert = require('node:assert/strict');
const addon = require(process.argv[2]);
let assertions = 0;
function equal(actual, expected) { assert.equal(actual, expected); assertions++; }
function rejects(fn) { assert.throws(fn); assertions++; }

const readers = [addon.readExternal, addon.readExternalRef, addon.readJsExternal,
  addon.readJsExternalRef, addon.readCompatExternal];

for (const make of [() => addon.makeExternal('ordinary'), () => addon.makeExternalRef('ordinary'),
  ...[0, 1024, -1024].map(hint => () => addon.makeCompatExternal('ordinary', hint))]) {
  const value = make();
  equal(addon.transportExternal(value), value);
  for (const read of readers) equal(read(value), 'ordinary');
  equal(addon.mutateExternal(value), 'ordinary!');
  for (const read of readers) equal(read(value), 'ordinary!');
}

for (const value of [addon.makeWrongExternal(), ...[0, 1, 2].map(addon.makeForeignExternal)]) {
  // Generic transport remains valid; only typed payload extraction is restricted.
  equal(addon.transportExternal(value), value);
  for (const consume of [...readers, addon.mutateExternal]) rejects(() => consume(value));
}

async function collect() {
  global.gc();
  await new Promise(resolve => setImmediate(resolve));
}

async function main() {
  assert.equal(typeof global.gc, 'function', 'run with --expose-gc');
  const before = addon.externalDropCount();
  let value = addon.makeTrackedExternal('retained');
  addon.holdExternal(value);
  value = null;
  for (let i = 0; i < 5; i++) await collect();
  equal(addon.externalDropCount(), before);
  equal(addon.heldExternalValue(), 'retained');
  addon.releaseExternal();
  for (let i = 0; i < 50 && addon.externalDropCount() === before; i++) await collect();
  equal(addon.externalDropCount(), before + 1);
  for (let i = 0; i < 5; i++) await collect();
  equal(addon.externalDropCount(), before + 1);
  console.log(`N-API external identity: ${assertions} assertions passed`);
}
main().catch(() => { console.error('N-API external identity regression failed'); process.exitCode = 1; });
