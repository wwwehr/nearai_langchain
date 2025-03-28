#!/bin/bash
BUILD_ARTIFACTS=/work/build
rm -rf "$BUILD_ARTIFACTS"
mkdir -p "$BUILD_ARTIFACTS"
pip install -r /work/requirements.txt --target "${BUILD_ARTIFACTS}"; \
   find ${BUILD_ARTIFACTS} -name '*.so' -exec strip {} \;; \
   find ${BUILD_ARTIFACTS} -path '**/*.py[c|o]' -delete; \
   find ${BUILD_ARTIFACTS} -path '**/__pycache__*' -delete; \
   find ${BUILD_ARTIFACTS} \( -type d -a -name test -o -name tests \) \
   -exec rm -rf '{}' +;
