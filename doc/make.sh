#!/usr/bin/env bash

# stop if anything goes wrong
set -e

# configure
export DJANGO_SETTINGS_MODULE=genui.settings.test
rm -rf ./source/api
mkdir -p ./source/api
sphinx-apidoc -o ./source/api/ ../src/genui/ ../src/genui/settings/databases/*prod* ../src/genui/settings/*prod* ../src/genui/settings/*stage*

# make
make html
