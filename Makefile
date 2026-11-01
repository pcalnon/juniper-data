# Minimal Makefile — coverage convenience target.
# The logic lives in util/run_coverage.bash (single source of truth); this is a thin wrapper.
.PHONY: coverage
coverage:  ## CI's unit lane + both coverage gates locally (the per-file gate needs juniper-ci-tools)
	@bash util/run_coverage.bash
