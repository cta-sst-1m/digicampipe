# digicampipe [![Build Status](https://travis-ci.org/calispac/digicampipe.svg?branch=master)](https://travis-ci.org/calispac/digicampipe)
DigiCam pipeline based on ctapipe

# Documentation

The documentation can be found here: [Digicampipe documentation](https://calispac.github.io/digicampipe)

# Installation

## Anaconda

You'll need Anaconda, so if you don't have it yet, just install it now.
Follow [the anaconda installation instructions](https://conda.io/docs/user-guide/install/linux.html).
We propose to use the most recent version.

    wget https://repo.continuum.io/archive/Anaconda3-5.0.0.1-Linux-x86_64.sh
    bash ./Anaconda3-5.0.0.1-Linux-x86_64.sh

## digicampipe

We propose to have a tidy place and clone `digicampipe` into a folder `ctasoft/`

    mkdir ctasoft
    cd ctasoft
    git clone https://github.com/calispac/digicampipe

To not mix up your anaconda root environment with digicampipe, we propose
to make a so called environment, with all the dependencies in it.

    conda env create -f digicampipe/environment.yml
    source activate digicampipe

**Please Note**: When working with digicampipe, please always make sure you are really using the `digicampipe` environment. After `source activate digicampipe`
your prompt should look similar to this this:

    (digicampipe) username@host:~/ctasoft$

    pip install -r digicampipe/requirements.txt
    pip install -e digicampipe

Run the tests on your machine:

    pytest -vv digicampipe

## Build the documentation with [Sphinx](http://www.sphinx-doc.org/en/stable/) (optional)

In the `digicampipe` directory run:

    cd docs/
    make html

This should create the documentation in `digicampipe/docs/build/html`.
You can open the html files with your favorite web browser.