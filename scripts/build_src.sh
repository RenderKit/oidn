#!/bin/bash

## Copyright 2009-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 version"
  exit 1
fi

if [ -d oidn-$1 ]; then
  echo "Error: oidn-$1 directory already exists"
  exit 1
fi

# Clone the repo
git clone --recursive git@github.com:OpenImageDenoise/oidn.git oidn-$1

# Checkout the requested version
cd oidn-$1
git checkout v$1
git submodule update --recursive

# Remove .git dirs and files
find -name .git | xargs rm -rf

# Create source packages
cd ..
tar -czvf oidn-$1.src.tar.gz oidn-$1
zip -r oidn-$1.src.zip oidn-$1
