const assert = require('node:assert/strict');
const addon = require(process.argv[2]);
let assertions = 0;
function equal(actual, expected) { assert.equal(actual, expected); assertions++; }
function rejects(fn) { assert.throws(fn); assertions++; }

const payload = new addon.Payload('ordinary');
const seed = new addon.Seed();
const consumers = [addon.readOwned, addon.readRef, addon.readMut, addon.readInstance, addon.readReference];
for (const consume of consumers) {
  equal(consume(payload), 'ordinary');
  for (const foreign of [seed, new addon.ForeignPayload('foreign'), {}, Object.create(addon.Payload.prototype), null, 42, 'x']) {
    rejects(() => consume(foreign));
  }
}
equal(payload.read(), 'ordinary');
equal(payload.withReference(), 'ordinary');
rejects(() => addon.Payload.prototype.withReference.call(seed));
equal(addon.readInstances([payload, payload]), 'ordinaryordinary');
rejects(() => addon.readInstances([payload, seed]));
rejects(() => addon.readOwned(Object.create(payload)));
rejects(() => addon.readOwned(new Proxy(payload, {})));
class Subclass extends addon.Payload {}
equal(addon.readOwned(new Subclass('subclass')), 'subclass');
equal(payload.value, 'ordinary');
payload.value = 'updated';
equal(payload.read(), 'updated');

const descriptor = Object.getOwnPropertyDescriptor(addon.Payload.prototype, 'value');
rejects(() => addon.Payload.prototype.read.call(seed));
rejects(() => descriptor.get.call(seed));
rejects(() => descriptor.set.call(seed, 'forged'));

const text = new addon.ContentText('text');
equal(addon.readEnum(text), 'text');
equal(addon.readEnum(new addon.ContentEmpty()), 'empty');
rejects(() => addon.readEnum(seed));

// Forging instanceof must not alter the native allocation identity.
Object.setPrototypeOf(seed, addon.Payload.prototype);
equal(seed instanceof addon.Payload, true);
for (const consume of consumers) rejects(() => consume(seed));
rejects(() => addon.Payload.prototype.read.call(seed));
rejects(() => descriptor.get.call(seed));
rejects(() => descriptor.set.call(seed, 'forged'));
Object.setPrototypeOf(seed, addon.ContentText.prototype);
equal(seed instanceof addon.ContentText, true);
rejects(() => addon.readEnum(seed));

// Mutating hasInstance is independent of the hidden Node-API type tag.
Object.defineProperty(addon.Payload, Symbol.hasInstance, { value: () => true });
Object.defineProperty(addon.ContentText, Symbol.hasInstance, { value: () => true });
for (const consume of consumers) rejects(() => consume(new addon.Seed()));
rejects(() => addon.readEnum(new addon.Seed()));

const dynamic = addon.makeDynamic();
equal(dynamic.read(), 'dynamic');
equal(addon.Dynamic.factory('factory').read(), 'factory');
rejects(() => addon.Dynamic.factory.call(addon.Seed, 'forged'));
equal(addon.readDynamic(dynamic), 'dynamic');
rejects(() => addon.readDynamic(payload));
rejects(() => addon.readInstance(dynamic));
rejects(() => addon.Dynamic.prototype.read.call(payload));

// Generic object wrappers use a distinct tagged envelope, preserving rewrap.
const wrapped = addon.genericWrap({});
equal(addon.genericUnwrap(wrapped), 'wrapped');
rejects(() => addon.genericUnwrap(payload));
rejects(() => addon.genericWrap(payload));
rejects(() => addon.genericRemove(payload));
rejects(() => addon.readOwned(wrapped));
addon.genericRemove(wrapped);
equal(addon.genericUnwrap(addon.genericWrap(wrapped)), 'wrapped');

// Reusing another class as an AbortSignal may not reinterpret its payload.
const controller = new AbortController();
addon.signal(controller.signal);
addon.signal(controller.signal);
controller.abort();
rejects(() => addon.signal(payload));
rejects(() => controller.signal.onabort.call(payload));
equal(addon.readOwned(payload), 'updated');
console.log(`N-API native identity: ${assertions} assertions passed`);
