from setuptools import (
    setup,
    find_packages
)

import glob
import os.path

VERSION_PATH = os.path.dirname(os.path.abspath(__file__))
VERSION_PATH = os.path.join(VERSION_PATH, 'digicampipe/VERSION')

with open(VERSION_PATH) as f:
    __version__ = f.read().strip()


def make_console_scripts(glob_expr='digicampipe/scripts/*.py'):

    command_list = []

    for path in sorted(glob.glob(glob_expr)):

        if not os.path.basename(path).startswith('__'):

            command_name = os.path.basename(path)
            script_filename = command_name.split('.')[0]
            command_name = command_name.split('.')[0].replace('_', '-')

            command = 'digicam-{call:s}=digicampipe.scripts.{stub:s}:' \
                      'entry'.format(call=command_name, stub=script_filename)

            command_list.append(command)

    return command_list


setup(
    name='digicampipe',
    version=__version__,
    packages=find_packages(),
    url='https://github.com/cta-sst-1m/digicampipe',
    license='GNU GPL 3.0',
    author='Cyril Alispach',
    author_email='cyril.alispach@gmail.com',
    long_description=open('README.md').read(),
    description='A package for DigiCam pipeline',
    install_requires=[
        'ctapipe>=0.6.0',
        'numpy',
        'matplotlib',
        'scipy',
        'astropy',
        'h5py',
        'tqdm',
    ],
    tests_require=['pytest>=3.0.0'],
    setup_requires=['pytest-runner'],
    package_data={
        '': [
            'VERSION',
            'tests/resources/*',
            'tests/resources/stars_on_lid/*',
            'tests/resources/digicamtoy/*',
        ],
    },
    entry_points={
        'console_scripts': make_console_scripts(),
    }
)
