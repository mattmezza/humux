.PHONY: release-whistle

# Whistle has its own version, release tag and image, independently of Humux.
release-whistle:
	$(MAKE) -C whistle release name="$(name)"
