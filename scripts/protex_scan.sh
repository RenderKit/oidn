#!/bin/bash

## Copyright 2009-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

# Fail when individual commands fail (-e), also in intermediate steps in
# pipelines (-o pipefail).
set -euo pipefail

# Debug only: print commands before executing them (-x).
# set -x

if [ -z "${OIDN_PROTEX_USER_HOME:-}" ]; then
  echo "Error: you must set OIDN_PROTEX_USER_HOME"
  exit 1
fi

if [ -z "${OIDN_PROTEX_PROJECT_NAME:-}" ]; then
  echo "Error: you must set OIDN_PROTEX_PROJECT_NAME"
  exit 1
fi

if [ -z "${OIDN_PROTEX_BDS:-}" ]; then
  echo "Error: you must set OIDN_PROTEX_BDS"
  exit 1
fi

if [ -z "${OIDN_PROTEX_SERVER_URL:-}" ]; then
  echo "Error: you must set OIDN_PROTEX_SERVER_URL"
  exit 1
fi

# Root dir defaults to $PWD
ROOT_DIR=${OIDN_ROOT_DIR:-$PWD}

export _JAVA_OPTIONS="-Duser.home=${OIDN_PROTEX_USER_HOME}"

cd ${ROOT_DIR}

${OIDN_PROTEX_BDS} new-project --server ${OIDN_PROTEX_SERVER_URL} ${OIDN_PROTEX_PROJECT_NAME} |& tee ip_protex.log
if grep -q "command failed" ip_protex.log; then
  exit 1
fi

${OIDN_PROTEX_BDS} analyze --server ${OIDN_PROTEX_SERVER_URL} |& tee -a ip_protex.log
if grep -q "command failed" ip_protex.log; then
  exit 1
fi

if grep -E "^Files pending identification: [0-9]+$" ip_protex.log; then
  echo "Protex scan FAILED!"
  exit 1
fi

echo "Protex scan PASSED!"
exit 0



