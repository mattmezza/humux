.PHONY: help dev build watch deploy publish release

TAILWIND := ./tailwindcss
CSS_IN := input.css
CSS_OUT := assets/css/style.css
WWW_DIR := .
BUILD_DIR := dist

help:
	@echo ""
	@echo "humux website — Human Multiplexer marketing site"
	@echo ""
	@echo "  make dev       Start local dev server + Tailwind CSS watcher"
	@echo "  make build     Build production CSS (minified)"
	@echo "  make watch     Watch Tailwind CSS for changes"
	@echo "  make deploy    Deploy to Cloudflare Pages (via wrangler)"
	@echo "  make publish   Build + deploy in one step"
	@echo ""
	@echo "  Release:"
	@echo "    make release name=www-v0.1   Tag, push, trigger GHA deploy"
	@echo ""

PORT ?= 9091

# Local development — run both in separate terminals or use &
dev:
	@echo "Starting dev server at http://localhost:$(PORT)"
	@echo "Run 'make watch' in another terminal for CSS hot-reload"
	@cd $(WWW_DIR) && python3 -m http.server $(PORT)

# Watch Tailwind CSS for changes
watch:
	$(TAILWIND) --watch --input $(WWW_DIR)/$(CSS_IN) --output $(WWW_DIR)/$(CSS_OUT)

# Build production CSS
build:
	$(TAILWIND) --input $(WWW_DIR)/$(CSS_IN) --output $(WWW_DIR)/$(CSS_OUT) --minify
	@echo "Build complete: $(CSS_OUT)"

# Deploy to Cloudflare Pages (requires wrangler CLI + CLOUDFLARE_API_TOKEN)
deploy:
	@command -v wrangler >/dev/null 2>&1 || { echo "wrangler not found. Install: npm i -g wrangler"; exit 1; }
	wrangler pages deploy $(WWW_DIR) --project-name humux

# Build + deploy in one step
publish: build deploy

# Create a release tag for GHA-based deploy (CF Pages)
release:
	@test -n "$(name)" || (echo "Usage: make release name=www-v0.1" && exit 1)
	@case "$(name)" in www-*) ;; *) echo "ERROR: name must start with 'www-' (got: $(name))"; exit 1;; esac
	git add -A
	git commit -m "www: $(name)" || true
	git tag "$(name)"
	git push origin "$(name)"
	@echo "Tag $(name) pushed. GHA will build and deploy to Cloudflare Pages."
