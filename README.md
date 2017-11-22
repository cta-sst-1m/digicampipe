# digicampipe
DigiCam pipeline based on ctapipe

# Installation

## Anaconda 

Follow [the anaconda installation instructions](https://conda.io/docs/user-guide/install/linux.html).
We propose to use the most recent version.

    wget https://repo.continuum.io/archive/Anaconda3-5.0.0.1-Linux-x86_64.sh
    bash ./Anaconda3-5.0.0.1-Linux-x86_64.sh
    
## Create a virtual environment

    conda create -n digicampipe python=3.5
    source activate digicampipe
    
## install necessary libraries and packages for CTA software

    conda install protobuf=3.0.0 numpy six scipy astropy ipython_genutils decorator 
    conda install matplotlib llvmlite hdf5 ipython

## Prepare a folder

    mkdir ctasoft
    cd ctasoft

## Get ProtoZFitsReader

This step involves a bit of manual work, but we are working on streamlining it.

    wget www.isdc.unige.ch/~lyard/repo/ProtoZFitsReader-0.41.Python3.5.Linux.x86_64.tar.gz
    pip install ProtoZFitsReader-0.41.Python3.5.Linux.x86_64.tar.gz
    
type

    python -m site

you'll get something similar to this:

    (digicampipe) dneise@lair:~/ctasoft$ python -m site
    sys.path = [
        '/home/dneise/ctasoft',
        '/home/dneise/anaconda36/envs/digicampipe/lib/python35.zip',
        '/home/dneise/anaconda36/envs/digicampipe/lib/python3.5',
        '/home/dneise/anaconda36/envs/digicampipe/lib/python3.5/plat-linux',
        '/home/dneise/anaconda36/envs/digicampipe/lib/python3.5/lib-dynload',
        '/home/dneise/.local/lib/python3.5/site-packages',
        '/home/dneise/anaconda36/envs/digicampipe/lib/python3.5/site-packages',    <----- your envs site-packages
    ]
    USER_BASE: '/home/dneise/.local' (exists)
    USER_SITE: '/home/dneise/.local/lib/python3.5/site-packages' (exists)
    ENABLE_USER_SITE: True
    
Note the `site-packages` folder of your environment. Add this folder to your LD_LIBRARY_PATH:

    export LD_LIBRARY_PATH=$HOME/anaconda36/envs/digicampipe/lib/python3.5/site-packages:$LD_LIBRARY_PATH
    

## Get a bunch of repos (and install in "editable" mode)

    git clone https://github.com/cta-observatory/pyhessio
    pip install -e pyhessio
   
    git clone https://github.com/calispac/ctapipe
    pip install -e ctapipe
    
    git clone https://github.com/calispac/ctapipe-extra
    pip install -e ctapipe-extra
    
    git clone https://github.com/calispac/digicamviewer
    pip install -e digicamviewer
    
    git clone https://github.com/calispac/digicampipe
    pip install -e digicampipe
    

You can only run the software, if you have access to example input data. You have to ask somebody for this.

    ipython digicampipe/pipeline_crab.py
