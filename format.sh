#!/usr/bin/env bash
set -eu -o pipefail

VENV_DIR="./.venv"
python3 -m venv ${VENV_DIR}
"${VENV_DIR}/bin/pip3" install -q -r ./requirements-dev.txt
"${VENV_DIR}/bin/python3" -m ruff check --fix ./src ./tests
"${VENV_DIR}/bin/python3" -m ruff format ./src ./tests
