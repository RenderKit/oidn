#!/bin/bash -x
## Copyright 2019-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

set -e
KW_SERVER_PATH=$KW_PATH/server
KW_CLIENT_PATH=$KW_PATH/client
export KLOCWORK_LTOKEN=/tmp/ltoken
echo "$KW_SERVER_IP;$KW_SERVER_PORT;$KW_USER;$KW_LTOKEN" > $KLOCWORK_LTOKEN
mkdir -p $CI_PROJECT_DIR/klocwork
log_file=$CI_PROJECT_DIR/klocwork/build.log

# Build
scripts/build.py --wrapper "$KW_CLIENT_PATH/bin/kwinject -w -o buildspec.txt" | tee -a $log_file
cd build
$KW_SERVER_PATH/bin/kwbuildproject --classic --force --url http://$KW_SERVER_IP:$KW_SERVER_PORT/oidn buildspec.txt --tables-directory mytables | tee -a $log_file
$KW_SERVER_PATH/bin/kwadmin --url http://$KW_SERVER_IP:$KW_SERVER_PORT load --force --name build-$CI_JOB_ID $KW_PROJECT_NAME mytables | tee -a $log_file

# Store kw build name for check status later
echo "build-$CI_JOB_ID" > $CI_PROJECT_DIR/klocwork/build_name

