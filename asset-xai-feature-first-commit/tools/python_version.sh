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

LINE=$(perl -ne "print if /^__version__\s+=\s+'(\d+\.\d+(\.\d+|(rc\d+)*))'$/" \
  ${PACKAGE_DIR}/__init__.py | (head -n1 && tail -n1))

if [ -z "${LINE}" ]; then
  exit 1
else
  VERSION=$(echo ${LINE} | perl -p -e "s/__version__\s+=\s+'(\d+\.\d+(\.\d+|(rc\d+)*))'/\1/g")
  echo ${VERSION}
fi
