# Pi client distribution snapshot

`PiClient/` contains the seven public source/license files pinned in
`PiClient.lock.json`. Their owning source is the separate
[stark-agent-clients repository](https://github.com/jshahbazi/stark-agent-clients).
This is a distribution input for clean local/hosted builds, not an independently
maintained client implementation. Do not edit these files here.

Make client changes and tests in the owning repository, commit them, then copy
each allowlisted file byte-for-byte from that immutable Git revision. Verify the
copied bytes against `git show REVISION:PATH`; update the lock revision and every
SHA-256 together. Keep its license unchanged. The builder and CI check the actual
snapshot against that lock before staging public resources into the app.

The optional `--pi-client-source` build argument verifies a separate checkout
against the same pin. Default builds require no sibling checkout, package
installation, GitHub token or network access for Pi resources. The distribution
recipe includes the staging helper and this snapshot in its source identity.
