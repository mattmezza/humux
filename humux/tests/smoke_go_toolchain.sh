#!/bin/sh
# Run inside the built image as its default non-root user; see deployment docs.
set -eu
[ "$(id -u)" != 0 ]
if [ "${1:-enabled}" = disabled ]; then
    if command -v go >/dev/null 2>&1; then
        echo 'Go must be absent from an INSTALL_GO=false image' >&2
        exit 1
    fi
    wacli version
    exit 0
fi

wacli version
go version
case "$(go env GOVERSION)" in
    go1.25.*) ;;
    *) echo 'Expected the Go 1.25 toolchain used by the wacli builder' >&2; exit 1 ;;
esac
[ "$(go env GOPATH)" = /app/data/go ]
[ "$(go env GOMODCACHE)" = /app/data/go/pkg/mod ]
[ "$(go env GOCACHE)" = /app/data/go-build-cache ]
[ "$(go env GOTOOLCHAIN)" = local ]
[ "$(go env CGO_ENABLED)" = 1 ]
mkdir -p "$(go env GOMODCACHE)" "$(go env GOCACHE)"
[ -w "$(go env GOMODCACHE)" ]
[ -w "$(go env GOCACHE)" ]
workspace=$(mktemp -d)
trap 'rm -r "$workspace"' EXIT
cd "$workspace"
cat > go.mod <<'EOF'
module humux.example/smoke

go 1.25
EOF
cat > main.go <<'EOF'
package main

/*
int add(int a, int b) { return a + b; }
*/
import "C"

func add(a, b int) int { return int(C.add(C.int(a), C.int(b))) }
func main() { if add(2, 3) != 5 { panic("CGO failed") } }
EOF
cat > main_test.go <<'EOF'
package main
import "testing"
func TestAdd(t *testing.T) { if add(2, 3) != 5 { t.Fatal("CGO failed") } }
EOF
go build -o smoke .
./smoke
go test ./...
go vet ./...
