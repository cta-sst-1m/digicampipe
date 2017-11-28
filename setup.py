from setuptools import setup


setup(
    name='digicampipe',
    version='0.1.1',
    packages=[
        'digicampipe',
        'digicampipe.io',
        'digicampipe.calib',
        'digicampipe.calib.camera',
        'digicampipe.utils',
        'digicampipe.visualization',
        'digicampipe.image',
        'digicampipe.instrument',
        'digicampipe.scripts',
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
    ],
    tests_require=['pytest>=3.0.0'],
    setup_requires=['pytest-runner'],
    entry_points={
        'console_scripts': [
            'digicam_crab_pipeline = digicampipe.scripts.pipeline_crab:main',
            'digicam_trigger_control = digicampipe.scripts.trigger_control:main',
            'digicam_shower_rate = digicampipe.scripts.shower_rate:main',
            'digicam_show_hillas = digicampipe.scripts.show_hillas:main',
            'digicam_plot_alpha_corrected = digicampipe.scripts.plot_alpha_corrected:main',
            'digicam_bias_curve_from_clocked_trigger = digicampipe.scripts.bias_curve_from_clocked_trigger:main',
            'digicam_bias_curve_mc = digicampipe.scripts.bias_curve_mc:main',
            'digicam_dark_evaluation = digicampipe.scripts.dark_evaluation:main',
            'digicam_load_container_test = digicampipe.scripts.load_container_test:main',
            'digicam_nsb_evalutation = digicampipe.scripts.nsb_evalutation:main',
            'digicam_read_raw_events = digicampipe.scripts.read_raw_events:main',
            'digicam_save_container_test = digicampipe.scripts.save_container_test:main',
        ],
    }
)
