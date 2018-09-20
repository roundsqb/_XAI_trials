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

PACKAGE_DIR=$1

LCA=$(git merge-base origin/develop origin/master)
CNT_LCA=$(git rev-list --count ${LCA}..HEAD)

LINE=$(perl -ne "print if /^__version__\s+=\s+'(\d+\.\d+(\.\d+|(rc\d+)*))'$/" \
  ${PACKAGE_DIR}/__init__.py | (head -n1 && tail -n1))

if [ ! -z "${LINE}" ] && [ ! -z "${CNT_LCA}" ]; then
    perl -pi -e 's/(__version__.*(\.|rc))(\d+)(.+)/$1.($3 + 1)."'".dev${CNT_LCA}"'".$4/ge' ${PACKAGE_DIR}/__init__.py
else
  exit 1
fi
