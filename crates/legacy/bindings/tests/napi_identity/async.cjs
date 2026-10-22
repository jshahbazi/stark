const assert = require('node:assert/strict');
const addon = require(process.argv[2]);
(async () => {
  const value = await addon.Dynamic.asyncFactory('async');
  assert.equal(addon.readDynamic(value), 'async');
  assert.throws(() => addon.readOwned(value));
  const returned = addon.genericWrap({});
  addon.genericRemove(returned);
  function ReturnRemoved() { return returned; }
  for (let attempt = 0; attempt < 2; attempt++) {
    await assert.rejects(addon.Dynamic.asyncFactory.call(ReturnRemoved, 'rejected'));
    assert.equal(addon.genericUnwrap(addon.genericWrap(returned)), 'wrapped');
    addon.genericRemove(returned);
  }
  function ReturnFresh() { return { marker: 'fresh' }; }
  const custom = await addon.Dynamic.asyncFactory.call(ReturnFresh, 'custom');
  assert.equal(addon.readDynamic(custom), 'custom');
  assert.equal(custom.marker, 'fresh');
  function Custom() {}
  assert.equal(addon.readDynamic(await addon.Dynamic.asyncFactory.call(Custom, 'constructor')), 'constructor');
  function ReturnExisting() { return value; }
  await assert.rejects(addon.Dynamic.asyncFactory.call(ReturnExisting, 'rejected'));
  assert.equal(addon.readDynamic(value), 'async');
  console.log('N-API async factory: 11 assertions passed');
})().catch(() => {
  process.stderr.write('N-API async factory test failed\n');
  process.exitCode = 1;
});
