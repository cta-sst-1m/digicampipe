# digicampipe [![Build Status](https://travis-ci.org/cta-sst-1m/digicampipe.svg?branch=master)](https://travis-ci.org/cta-sst-1m/digicampipe)
DigiCam pipeline based on ctapipe

The documentation can be found here: [Digicampipe documentation](https://cta-sst-1m.github.io/digicampipe)

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
    git clone https://github.com/cta-sst-1m/digicampipe

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

If you want to execute only very fast tests do:

    pytest -c quick_test.ini digicampipe

## Build the documentation with [Sphinx](http://www.sphinx-doc.org/en/stable/) (optional)

In the `digicampipe` directory run:

    cd docs/
    make html

This should create the documentation in `digicampipe/docs/build/html`.
You can open the html files with your favorite web browser.
To delete the documentation us:

    make clean

## Pointing using lid CCD images

To be able to determine the pointing using stars reflections on the lid, astrometry.net is needed.
If not installed locally, the web service will be used instead.
To install astrometry.net:

    sudo apt-get install astrometry.net

You will need to download catalogs too:

    mkdir ~/astrometry-data
    cd ~/astrometry-data
    wget http://broiler.astrometry.net/~dstn/4200/wget.sh
    chmod +x wget.sh
    ./wget.sh

Grab a beer as it will take a while ... Also, indexes 4203 and lower are probably not needed.

Then add the folowing line to /etc/astrmetry.cfg:

    add_path ~/astrometry-data

That's it, you are ready to go.
