# Go image verification

PRs do not build Humux images for Go toolchain verification. The dedicated
`Go workspace image` workflow has been removed. Application lint and tests still
run on PRs. No branch protection or rulesets required the removed checks when
this change was prepared; do not add their old names as required status checks.

On release, build and load the standard image once, smoke-test that image, then
tag and push it without rebuilding. The build uses the dedicated
`humux-image-go-true` BuildKit GHA cache scope so other component images cannot
replace its cache. Release publishing preserves existing metadata labels and
tags. Go-disabled images are not built automatically.

The smoke helper runs without networking as the image's default non-root user.
It compiles/runs CGO code, tests and vets it, verifies Go's cache paths under
`/app/data`, and checks writable persisted caches from a second container.
It also runs `wacli version`. The temporary volume is removed on failure too.

To check a locally built image manually:

```sh
sh humux/tests/smoke_go_image.sh humux:local
# For an image built with INSTALL_GO=false:
sh humux/tests/smoke_go_image.sh humux:lean disabled
```

The disabled check rejects any Go executable and verifies wacli still works.
This is now a manual check: opting out of Go no longer has automatic PR coverage.
The standard release image retains automatic coverage before publication.

## Measured cost of the removed PR workflow

From Actions runs on 2026-10-23:

| Run | Total elapsed, including queueing | Standard build | Disabled build |
| --- | --- | --- | --- |
| 40019782633 | 13m05s | — | — |
| 40021060966 | 10m19s | 5m12s | 6m47s |
| 40022086667 | 12m43s | 9m01s | 2m23s |

In run 40021060966, the disabled build spent about 78s on system apt packages,
69s on image export/load and 104s exporting GHA cache. Its context transfer was
2.3 MB in under a second. In run 40022086667, standard-image cache export alone
took 201s; its actual Go smoke test took 7s. Both variants and release publishing
previously wrote the default `buildkit` cache scope. Building and moving images,
not the tiny Go test or context transfer, dominated the delay.

Removing the PR workflow eliminates this entire 10–13 minute gate on affected
PRs. Release smoke testing adds no additional image build; loading the image for
testing still has a cost. The small existing build context needs no new packaging
system. Application code, dependencies, optional model prefetching and Tailwind
remain build inputs in the existing Dockerfile; the image is built at release.
