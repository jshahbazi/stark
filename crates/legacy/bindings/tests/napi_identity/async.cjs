const assert = require('node:assert/strict');
const addon = require(process.argv[2]);
(async () => {
  const value = await addon.Dynamic.asyncFactory('async');
  assert.equal(addon.readDynamic(value), 'async');
  assert.throws(() => addon.readOwned(value));
  console.log('N-API async factory: 2 assertions passed');
})().catch(() => {
  process.stderr.write('N-API async factory test failed\n');
  process.exitCode = 1;
});
