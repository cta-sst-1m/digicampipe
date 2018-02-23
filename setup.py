from glob import glob
from os.path import basename, splitext
from setuptools import setup

with open('digicampipe/VERSION') as f:
    __version__ = f.read().strip()

console_scripts_filenames = glob('digicampipe/scripts/*.py')


setup(
    name='digicampipe',
    version=__version__,
    packages=[
        'digicampipe',
        'digicampipe.io',
        'digicampipe.calib',
        'digicampipe.calib.camera',
        'digicampipe.utils',
        'digicampipe.visualization',
        'digicampipe.image',
        'digicampipe.instrument'
    ],
    url='https://github.com/calispac/digicampipe',
    license='GNU GPL 3.0',
    author='Cyril Alispach',
    author_email='cyril.alispach@gmail.com',
    long_description=open('README.md').read(),
    description='A package for DigiCam pipeline',
    install_requires=[
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
            'tests/resources/stars_on_lid/*'
        ],
    },
    entry_points={
        'console_scripts': [
            'dg_{filename} = digicampipe.scripts.{filename}:entry'.format(
                filename=splitext(basename(filename)))
            for filename in console_scripts_filenames
        ]
    },
)
