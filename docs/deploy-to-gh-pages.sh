#!/bin/bash
# See https://medium.com/@nthgergo/publishing-gh-pages-with-travis-ci-53a8270e87db

cd docs/
mkdir public

# config
git config --global user.email "nobody@nobody.org"
git config --global user.name "Travis CI"

# build the documentation (CHANGE THIS)

make html
cp -r build/html/. public/
make clean

# deploy
cd public
git init
git add .
git commit -m "Deploy to Github Pages"
git push --force --quiet "https://${GITHUB_TOKEN}@${GITHUB_REPO}" master:gh-pages
cd ..
rm -rf public/
