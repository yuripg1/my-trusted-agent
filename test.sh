#!/usr/bin/env bash
set -eu -o pipefail

VENV_DIR="./.venv"
CACHE_DIR="./.cache"
python3 -m venv ${VENV_DIR}
"${VENV_DIR}/bin/pip3" install -q -r ./requirements-dev.txt
PYTHONPYCACHEPREFIX="${CACHE_DIR}/pycache" COVERAGE_FILE="${CACHE_DIR}/coverage_data" "${VENV_DIR}/bin/python3" -m pytest -o "cache_dir=${CACHE_DIR}/pytest" --cov=src --cov-branch --cov-report=term-missing ./tests
