# Makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
DEMOSRC       = ../demos

# Copy demo markdown + images into docs/demos/ before building
.PHONY: copy-demos
copy-demos:
	@mkdir -p _demos
	@for demo in lorenz gravity_waves simple_spectral_solver; do \
		mkdir -p demos/$$demo; \
		cp -u $(DEMOSRC)/$$demo/*.md _demos/$$demo/ 2>/dev/null || true; \
		cp -u $(DEMOSRC)/$$demo/*.gif _demos/$$demo/ 2>/dev/null || true; \
		cp -u $(DEMOSRC)/$$demo/*.png _demos/$$demo/ 2>/dev/null || true; \
	done

.PHONY: html
html: copy-demos
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo "Build finished. HTML pages are in $(BUILDDIR)/html."

.PHONY: clean
clean:
	rm -rf $(BUILDDIR) _demos/ _autosummary/

.PHONY: help
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
