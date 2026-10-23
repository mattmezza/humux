# Whistle

A small CPU-only speech-to-text service with an OpenAI-compatible HTTP API.
This independent service wraps Cactus Compute's Whistle model using
`cactus-needle`. It has no dependency on Humux and can be used by any compatible
client. The runtime image bundles the native engine and weights; transcription
works offline without a GPU.

## Run

After the first Whistle release has successfully published:

```sh
docker run -d --name whistle --restart unless-stopped \
  -p 127.0.0.1:8000:8000 ghcr.io/mattmezza/whistle:3.2.8
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@recording.ogg -F model=whistle
```

The API returns `{"text":"..."}`. `whisper-1` is also accepted as a model alias.
`GET /health` returns `{"status":"ok","model":"whistle"}` once the model loads.
The service has no authentication; expose it only to trusted clients.

Audio formats supported by ffmpeg are converted to 16 kHz mono PCM. The service
accepts at most 25 MiB / 10 minutes per request and serializes native inference.
Long recordings use the runtime's built-in windowing. Optional `language` accepts
`en`, `de`, `fr`, `es`, `it`, `nl`, `pl`; omit it for automatic detection.
Only `response_format=json` is supported.

## Develop

From this directory:

```sh
make test
make build
docker run --rm -p 127.0.0.1:8000:8000 whistle:local
```

Tests use a stand-in model; ffmpeg tests run when ffmpeg is installed. The
container smoke check uses the real engine and bundled speech fixture, offline.
Python 3.12 and uv are required for development. The Docker build context is
this directory, independent of the parent repository. `tests/fixtures/README.md`
contains the audio attribution.

## Independent releases

Image versions match the pinned `cactus-needle` Python package version. This is
the inference runtime version (currently `3.2.8`), not the upstream Whistle
weights version (`2.0.0`). Updating the runtime must preserve the API contract.

After merging changes to the repository's default branch, with a clean checkout:

```sh
# From the parent repository root or humux/:
make release-whistle name=v3.2.9
# Or from this directory:
make release name=v3.2.9
```

The requested version must already exist on PyPI. The release tool updates
`VERSION` and the runtime pin, commits and pushes the change, and creates a
GitHub release tagged `whistle-v3.2.9`. For the initial release, use `v3.2.8`.
The GitHub CLI must be authenticated with repository release access.

Publishing a `whistle-v*` GitHub release triggers `.github/workflows/whistle.yml`:

1. Verify the release tag, `VERSION`, and runtime dependency match.
2. Lint/test the service and build the image on the self-hosted runner's architecture.
3. Start that image with networking disabled and transcribe the real speech fixture.
4. Push the exact tested image to `ghcr.io/<repository-owner>/whistle:<version>`;
   stable releases also update `:latest`.

PRs and manual workflow runs verify only; they never publish images. No Humux
release is created or required. Humux's Compose/Make integration pins an image
version separately, so deploying a Whistle upgrade is an explicit consumer update.
The first package publication may require setting its visibility to public in
GitHub Packages for unauthenticated pulls. There is no published image until
that first release workflow succeeds.

## License

This service's code is MIT; see `LICENSE`. Cactus Compute's engine and model have
their own licenses. The bundled test audio is CC BY-SA 4.0; see its attribution.
