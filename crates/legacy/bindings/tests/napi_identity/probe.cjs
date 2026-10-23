// A same-layout synthetic wrong class demonstrates acceptance on the original
// locked runtime without depending on a crash or incompatible String layout.
const assert = require('node:assert/strict');
const addon = require(process.argv[2]);
const fixed = process.argv[3] === 'fixed';
const foreign = new addon.ForeignPayload('foreign');
let cases = 0;
for (const consume of [addon.readOwned, addon.readRef, addon.readMut, addon.readInstance, addon.readReference]) {
  if (fixed) assert.throws(() => consume(foreign));
  else assert.equal(consume(foreign), 'foreign');
  cases++;
}
Object.setPrototypeOf(foreign, addon.ContentText.prototype);
if (fixed) assert.throws(() => addon.readEnum(foreign));
else assert.equal(addon.readEnum(foreign), 'foreign');
cases++;
console.log(`N-API ${fixed ? 'patched rejection' : 'baseline wrong-class acceptance'}: ${cases} cases passed`);
