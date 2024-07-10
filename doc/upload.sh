#!/usr/bin/env bash

set -e

# Get the commit hash of current HEAD and qspred version
COMMIT_ID=$(git rev-parse --short HEAD)
PACKAGE_VERSION=$(python -c "import genui; print(genui.__version__)")
MSG="Adding docs to gh-pages for $COMMIT_ID"
REMOTE_NAME="origin"
REMOTE_URL=${REPO_URL:-$(git config --get remote.$REMOTE_NAME.url)}
git remote set-url --push $REMOTE_NAME "$REMOTE_URL"
echo "Remote push URL set to: $REMOTE_URL"

# Clone a temporary copy of the repo with just the gh-pages branch
BASE_DIR=$(pwd)
HTML_DIR="$BASE_DIR/build/html/"
TEMP_REPO_DIR="/tmp/docs/$USER/GenUI/"
rm -rf "$TEMP_REPO_DIR"
mkdir -p -m 0755 "$TEMP_REPO_DIR"
git clone --single-branch --branch gh-pages "$REMOTE_URL" "$TEMP_REPO_DIR"

# Update the web page directories
cd "$TEMP_REPO_DIR"
# link to latest if this is not a dev version or an alpha, beta or rc release
if [[ $PACKAGE_VERSION != *"dev"* ]]; then
    rm -rf "docs/"
    mkdir -p "docs/"
    cp -r "$HTML_DIR"/* "docs/"
else
    rm -rf "docs-dev/"
    mkdir -p "docs-dev/"
    cp -r "$HTML_DIR"/* "docs-dev/"
fi
touch .nojekyll

# commit and push changes
git add -A
git commit -m "$MSG"
#git remote add $REMOTE_NAME "$REMOTE_URL"
echo "Pushing to $REMOTE_NAME gh-pages. Remote URL: $(git config --get remote.$REMOTE_NAME.url)"
git push $REMOTE_NAME gh-pages