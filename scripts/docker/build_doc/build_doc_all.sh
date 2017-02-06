#!/bin/sh
# This script execute processes for building document.
# This script will run in a Docker container.
# It is assumed that /home/jovyan/work is mounted on the root directory of the repo.

cd doc
make html
cd ..
