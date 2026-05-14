#!/usr/bin/env bash
set -eu -o pipefail

VENV_DIR="./.venv"
PYCACHE_DIR="./.pycache"
python3 -m venv ${VENV_DIR}
"${VENV_DIR}/bin/pip3" install -q -r ./requirements.txt
PYTHONPYCACHEPREFIX="${PYCACHE_DIR}" "${VENV_DIR}/bin/python3" ./src/main.py
