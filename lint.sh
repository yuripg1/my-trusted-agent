#!/usr/bin/env bash
set -eu -o pipefail

VENV_DIR="./.venv"
CACHE_DIR="./.cache"
python3 -m venv ${VENV_DIR}
"${VENV_DIR}/bin/pip3" install -q -r ./requirements-dev.txt
PYTHONPYCACHEPREFIX="${CACHE_DIR}/pycache" "${VENV_DIR}/bin/python3" -m mypy --cache-dir "${CACHE_DIR}/mypy" ./src
