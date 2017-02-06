#!/bin/sh
# This script is used to build Docker images.
# You needs to run this at bmlingam/scripts.

pushd ./docker/build_doc
docker build -t vb_bml_build_doc .
popd
