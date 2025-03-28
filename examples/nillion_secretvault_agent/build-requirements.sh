#!/bin/bash -ex
rm -rf .requirements.zip
docker build -t nillion-agent-deps-builder -f Dockerfile.builder .

docker run --rm -ti -u $(id -u):$(id -g) --entrypoint=/bin/bash \
  -v $(pwd):/work \
  nillion-agent-deps-builder:latest /work/create-py-zip.sh

pushd build
zip -9 -r ../.requirements.zip .
popd
aws s3 cp --acl public-read .requirements.zip s3://we.public/
rm -rf build .requirements.zip
