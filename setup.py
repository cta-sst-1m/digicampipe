from setuptools import (
    setup,
    find_packages
)

with open('digicampipe/VERSION') as f:
    __version__ = f.read().strip()


def make_console_scripts(glob_expr='digicampipe/scripts/*.py'):
    import glob
    import os.path

    return [
        'digicam-{call:22s}=digicampipe.scripts.{stub:22s}:entry'.format(
            call=os.path.basename(path).split('.')[0].replace('_', '-'),
            stub=os.path.basename(path).split('.')[0],
        )
        for path in sorted(glob.glob(glob_expr))
        if not os.path.basename(path).startswith('__')
    ]


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
