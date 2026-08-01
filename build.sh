#!/bin/bash

# Full build: generate the figure from code/, verify checksums, then compile the paper.
# A fresh clone runs this script; compile.sh remains the LaTeX-only step.

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Step 1: figure generation ===${NC}"
if ! python3 -c "import matplotlib" 2>/dev/null; then
    echo -e "${RED}matplotlib is not available. Install the pinned environment first:${NC}"
    echo "  pip install -r code/requirements.txt"
    exit 1
fi
SOURCE_DATE_EPOCH=0 python3 code/make_figures.py

echo -e "${GREEN}=== Step 2: artefact checksums ===${NC}"
if [ -f ARTIFACT_SHA256SUMS ]; then
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 -c ARTIFACT_SHA256SUMS
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum -c ARTIFACT_SHA256SUMS
    else
        echo -e "${RED}Neither shasum nor sha256sum is available${NC}"
        exit 1
    fi
else
    echo -e "${RED}ARTIFACT_SHA256SUMS missing${NC}"
    exit 1
fi

echo -e "${GREEN}=== Step 3: LaTeX compilation ===${NC}"
bash compile.sh
