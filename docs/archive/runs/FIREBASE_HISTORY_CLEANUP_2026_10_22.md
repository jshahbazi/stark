# Firebase configuration history cleanup (2026-10-22)

The inherited playground Firebase configuration was removed from the history of all 34 published branches and all 3 tags. The rewrite was published atomically with leases on every ref. All 37 branch and tag tip trees remained identical. A fresh fetch of those refs contained 30,996 Git objects; an in-memory comparison against the flagged key found zero matches. No key value was logged or included in validation artifacts.

The validated `main` revision after the rewrite was `04a5fd1a896362849cc4219540e5b6afa63a5ace`. Commit and tag IDs changed, and the rewrite stripped original commit and tag signatures. Checked-in upstream revision references and provenance text continue to identify their original sources. The original branch-protection settings were restored and verified after publication.

GitHub still retained 22 read-only PR head refs at verification time, and the secret-scanning alert remained open. Those refs and cached historical views require separate cleanup through [GitHub Support](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository#fully-removing-the-data-from-github). This validation covers the published branches and tags; it does not establish removal from PR refs, caches, existing local clones, or upstream Uzu.

Existing clones must migrate unpublished work onto the rewritten history before merging or pushing; merging old ancestry can restore the removed configuration. The playground ignore rule excludes its local Firebase configuration from ordinary `git add` operations.
