from setuptools import setup

setup(
    name='bfdc',
    version='0.0.1',
    packages=['bfdc'],
    install_requires=[
        'scikit-image',
        'pillow==5.0.0',
        'numpy',
        'scipy',
        'matplotlib',
        'read-roi'
    ],
    url='https://gitlab.pasteur.fr/aaristov/Brightfield_Drift_Tracking_3D.git',
    license='BSD',
    author='Andrey Aristov',
    author_email='aaristov@pasteur.fr',
    description='Acquire bright field images along with the super resolution data and use it to track drift in 3D with nanometer precision!'
)
