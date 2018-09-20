#!/usr/bin/env bash

# QUANTUMBLACK CONFIDENTIAL
#
# Copyright (c) 2016 - present QuantumBlack Visual Analytics Ltd. All
# Rights Reserved.
#
# NOTICE: All information contained herein is, and remains the property of
# QuantumBlack Visual Analytics Ltd. and its suppliers, if any. The
# intellectual and technical concepts contained herein are proprietary to
# QuantumBlack Visual Analytics Ltd. and its suppliers and may be covered
# by UK and Foreign Patents, patents in process, and are protected by trade
# secret or copyright law. Dissemination of this information or
# reproduction of this material is strictly forbidden unless prior written
# permission is obtained from QuantumBlack Visual Analytics Ltd.

GITHUB_USER=$1
GITHUB_REPO=$2
GITHUB_TOKEN=$3
VERSION=$4

GITHUB_ENDPOINT="https://api.github.com/repos/${GITHUB_USER}/${GITHUB_REPO}/releases"

PAYLOAD=$(cat <<-END
{
    "tag_name": "${VERSION}",
    "target_commitish": "master",
    "name": "${VERSION}",
    "body": "Release ${VERSION}",
    "draft": false,
    "prerelease": false
}
END
)

STATUS=$(curl -o /dev/null -L -s -w "%{http_code}\n" -X POST -H "Authorization: token ${GITHUB_TOKEN}" \
              -H "Content-Type: application/json" ${GITHUB_ENDPOINT} -d "${PAYLOAD}")

[ "${STATUS}" == "201" ] || [ "${STATUS}" == "422" ]
