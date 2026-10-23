#!/bin/sh
# Host-side smoke runner shared by PR verification and release publishing.
set -eu
image=${1:?Usage: smoke_go_image.sh IMAGE [disabled]}
mode=${2:-enabled}
script=$(dirname "$0")/smoke_go_toolchain.sh
if [ "$mode" = disabled ]; then
    docker run --rm -i --network none --entrypoint sh "$image" -s -- disabled < "$script"
    exit 0
fi
volume=$(docker volume create)
trap 'docker volume rm "$volume" >/dev/null' EXIT
# Use the image's default user and a fresh mounted data directory. A second
# container verifies the caches survived the first container's removal.
docker run --rm -i --network none --entrypoint sh \
    -v "$volume:/app/data" "$image" -s < "$script"
docker run --rm --network none --entrypoint sh \
    -v "$volume:/app/data" "$image" -c \
    'test -d /app/data/go/pkg/mod && test -w /app/data/go/pkg/mod && test -w /app/data/go-build-cache && test -n "$(ls -A /app/data/go-build-cache)"'
