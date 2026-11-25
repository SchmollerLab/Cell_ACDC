#!/bin/sh
cd "$(dirname "$0")" || exit 1

SPHINXBUILD="${SPHINXBUILD:-sphinx-build}"
SOURCEDIR="source"
BUILDDIR="_build"

if ! command -v "$SPHINXBUILD" >/dev/null 2>&1; then
    echo "The 'sphinx-build' command was not found. Install Sphinx or set SPHINXBUILD to its path."
    exit 1
fi

if [ -z "$1" ]; then
    set -- help
fi

exec "$SPHINXBUILD" -M "$1" "$SOURCEDIR" "$BUILDDIR" ${SPHINXOPTS:-} ${SPHINXOPTS_EXTRA:-}