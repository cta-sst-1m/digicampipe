#!/usr/bin/env bash
# Push HTML files to gh-pages automatically.

# Fill this out with the correct org/repo
ORG=calispac
REPO=digicampipe
# This probably should match an email for one of your users.
EMAIL=Travis

set -e

# Clone the gh-pages branch outside of the repo and cd into it.
cd ..
git clone -b gh-pages "https://$GITHUB_API_KEY@github.com/$ORG/$REPO.git" gh-pages
cd gh-pages

# Update git configuration so I can push.
if [ "$1" != "dry" ]; then
    # Update git config.
    git config user.name "Travis Builder"
    git config user.email "$EMAIL"
fi

# Copy in the HTML.  You may want to change this with your documentation path.
cp -R ../docs/build/html/* ./

# Add and commit changes.
git add -A .
git commit -m "[ci skip] Autodoc commit for $COMMIT."
if [ "$1" != "dry" ]; then
    # -q is very important, otherwise you leak your GH_TOKEN
    git push --quiet --force https://$ORG:$GITHUB_API_KEY@github.com/$ORG/$REPO-gh-pages gh-pages
fi