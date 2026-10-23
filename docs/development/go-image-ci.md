# Go image verification

`Go workspace image` starts on every PR and reports one stable result with that
name. Builds run only when these toolchain/build inputs change (including renames):

- `humux/Dockerfile`: base images, apt packages, wacli version, Go toolchain,
  Node, uv, optional model/browser installation and Tailwind CLI version.
- `humux/.dockerignore`: effective build context.
- `humux/pyproject.toml` and `humux/uv.lock`: Python installation inputs copied
  before application code. There are no local Go manifests: wacli is installed
  from its versioned upstream module, specified in the Dockerfile.
- `humux/tests/smoke_go_toolchain.sh` and `humux/tests/smoke_go_image.sh`.
- `.github/workflows/go-toolchain.yml` and `.github/workflows/release.yml`.

The Dockerfile also copies application directories (`core`, `channels`, `schema`,
`skills`, `tools`, `voice`, `api`); Tailwind reads `api/static/input.css` and
`api/templates`. These are intentionally not toolchain triggers. Application,
CSS, template and documentation changes use normal application tests on PRs;
the standard image is built and smoke-tested on every release. Optional model
prefetch code is not exercised by the default image. Keep the trigger list in
sync if new build dependency files or smoke helpers are introduced.

Relevant PRs build the standard image with its default settings, then
`INSTALL_GO=false`, using one BuildKit builder. Variant-specific runtime layers
come after the common layers, so apt, Node, Python dependencies and wacli can be
reused. Separate `humux-image-go-true`/`humux-image-go-false` GHA cache scopes
prevent the variants from replacing each other's cache. Both scopes are imported
on PRs; releases read/write the standard scope, seeding the default-branch cache.
Superseded runs on the same PR are cancelled.

On release, build and load the standard image once, run the same smoke helper,
then tag and push that loaded image. No build follows the smoke test. Release
publishing preserves the existing metadata labels and tags. The Go-disabled
image is tested on relevant PRs rather than adding another full release build.

The smoke helper runs without networking as the image's default non-root user.
The standard check compiles/runs CGO code, tests and vets it, verifies Go's cache
paths under `/app/data`, and checks writable persisted caches from a second
container. Both variants run `wacli version`; the opt-out check rejects any `go`
executable. The temporary Docker volume is removed even when smoke testing fails.

If branch protection is enabled, require the final **Go workspace image** job,
not the conditional build job or the retired `smoke (true)`/`smoke (false)` names.
No workflow-level path filter is used, so unrelated PRs get a successful result
rather than leaving a required workflow pending. Failure of path detection or
image testing fails this final job. No branch protection or rulesets were enabled
when this change was prepared.

## Measured baseline

From Actions runs on 2026-10-23:

| Run | Total elapsed, including queueing | Standard build | Disabled build |
| --- | --- | --- | --- |
| 40019782633 | 13m05s | — | — |
| 40021060966 | 10m19s | 5m12s | 6m47s |
| 40022086667 | 12m43s | 9m01s | 2m23s |

In run 40021060966, the disabled build spent about 78s on system apt packages,
69s on image export/load and 104s exporting GHA cache. Its context transfer was
2.3 MB in under a second. In run 40022086667, standard-image cache export alone
took 201s; its actual Go smoke test took 7s. This is primarily build/cache work,
not expensive Go tests or excessive context transfer. Both variants and release
publishing previously wrote the default `buildkit` cache scope, and changing
`INSTALL_GO` near the start invalidated subsequent dependency layers.

Expected savings: application/docs-only PRs run no image builds; relevant PRs
avoid duplicated common stage work and cache eviction between variants. Warm
runs still pay image load/export and smoke-test costs. The first run after the
Dockerfile/cache-scope change must populate new caches, so it is not a warm-run
benchmark. The existing small, filtered context does not warrant a new context
packaging system. Cache mounts alone would not preserve downloads across the
ephemeral BuildKit builders; unchanged expensive steps use exported layer cache.
