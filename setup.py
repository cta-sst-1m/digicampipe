from setuptools import setup


setup(
    name='digicampipe',
    version='0.1.0',
    packages=['digicampipe', 'digicampipe.io', 'digicampipe.calib', 'digicampipe.calib.camera', 'digicampipe.utils', 'digicampipe.visualization', 'digicampipe.image', 'digicampipe.instrument'],
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
    ],
    tests_require=['pytest>=3.0.0'],
    setup_requires=['pytest-runner'],
)
