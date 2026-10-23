# Components

Humux (`humux/`) and Whistle (`whistle/`) are independent components. Whistle is
an OpenAI-compatible speech-to-text service, not a Humux sidecar. Keep its source,
dependencies, tests, fixture attribution, Dockerfile, and release tooling within
`whistle/`, so it can move to its own repository. It must not import Humux code.
Humux integrates through HTTP and may provide optional Compose/Make conveniences.

## Whistle

- Python 3.12; CPU-only `cactus-needle` runtime; no GPU or training dependencies.
- `POST /v1/audio/transcriptions` accepts `whistle` and `whisper-1`; `GET /health`
  reports readiness. Keep inference serialized: the native engine is not thread-safe.
- Preserve offline operation, the small runtime image, and build-time cache cleanup.
- Run `make -C whistle test`. The dedicated workflow also builds the image and
  transcribes bundled human speech with Docker networking disabled.
- Whistle's image version matches its pinned **cactus-needle package** version,
  not the Whistle weights version. Keep `whistle/VERSION` and the requirement aligned.
- `make release-whistle name=v3.2.9` (root or `humux/`) invokes the component's
  release script. It validates a published runtime version and a clean default
  branch, updates the pin/version, commits/pushes, and publishes `whistle-v3.2.9`.
  Implementing or testing this target does not authorize running a real release.
- Publishing a `whistle-v*` GitHub release triggers only Whistle verification and
  publishing to `ghcr.io/<repository-owner>/whistle:<version>` and `:latest`.
  Whistle releases must not publish Humux images or deploy Humux documentation.
- Humux consumes a pinned Whistle image version. Updating that consumer pin is
  separate from releasing Whistle. Do not couple the two release cycles.
