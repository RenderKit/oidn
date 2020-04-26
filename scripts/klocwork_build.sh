#!/bin/bash
 
## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

# function repeat the command $1 times, when the last line output is equal ERROR_MSG
function retry_cmd()
{
    set +e
    MAX_RETRY=$1
    CMD="${@:2}"
    ERROR_MSG="License check failed ... Exiting"
 
    RETRY_COUNTER="0"
    while [ $RETRY_COUNTER -lt $MAX_RETRY ]; do
        CMD_OUTPUT=$($CMD)
        CMD_RETURN_CODE=$?
        echo "$CMD_OUTPUT"
        CMD_OUTPUT=$(echo $CMD_OUTPUT | tail -1)
        if [ $CMD_RETURN_CODE == 0 ] && [[ $CMD_OUTPUT != $ERROR_MSG ]]; then
            break
        elif [ $CMD_RETURN_CODE != 1 ]; then
            set -e
            echo "Unknown script error code = [$CMD_RETURN_CODE]"
            return $CMD_RETURN_CODE
        fi
        RETRY_COUNTER=$[$RETRY_COUNTER+1]
        echo "Found license check failed, [$RETRY_COUNTER/$MAX_RETRY] - retrying ... "
        sleep 10
    done
 
    set -e
    if [ $RETRY_COUNTER -ge $MAX_RETRY ]; then
        return 62
    fi
    return 0
}

if [ -z "${OIDN_KW_SERVER_IP}" ] \
|| [ -z "${OIDN_KW_SERVER_PORT}" ] \
|| [ -z "${OIDN_KW_USER}" ] \
|| [ -z "${OIDN_KW_LTOKEN}" ] \
|| [ -z "${OIDN_KW_CLIENT_PATH}" ] \
|| [ -z "${OIDN_KW_SERVER_PATH}" ];  then
  echo "You must set OIDN_KW_SERVER_IP, OIDN_KW_SERVER_PORT, OIDN_KW_USER, OIDN_KW_LTOKEN, OIDN_KW_CLIENT_PATH, and OIDN_KW_SERVER_PATH"
  exit 1
fi

KW_SERVER_IP=$OIDN_KW_SERVER_IP
KW_SERVER_PORT=$OIDN_KW_SERVER_PORT
KW_USER=$OIDN_KW_USER
KW_LTOKEN=$OIDN_KW_LTOKEN
export KLOCWORK_LTOKEN=/tmp/ltoken

KW_CLIENT_PATH=$OIDN_KW_CLIENT_PATH
KW_SERVER_PATH=$OIDN_KW_SERVER_PATH

set -e

echo "$KW_SERVER_IP;$KW_SERVER_PORT;$KW_USER;$KW_LTOKEN" > $KLOCWORK_LTOKEN

# Build
scripts/release.py build --wrapper "$KW_CLIENT_PATH/bin/kwinject -w -o buildspec.txt"
cd build_release
$KW_SERVER_PATH/bin/kwbuildproject --force --url http://$KW_SERVER_IP:$KW_SERVER_PORT/oidn buildspec.txt --tables-directory mytables
$KW_SERVER_PATH/bin/kwadmin --url http://$KW_SERVER_IP:$KW_SERVER_PORT load --force --name build-$CI_PIPELINE_ID oidn mytables | tee project_load.log

# Store kw build number for check status later
cat project_load.log | grep "Starting build" | cut -d":" -f2 > ./kw_build_number

