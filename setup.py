from setuptools import setup

setup(
    name='bfdc',
    version='0.1.4',
    packages=['bfdc'],
    install_requires=[
        'scikit-image',
        'pillow<>5.1',
        'numpy',
        'scipy',
        'matplotlib<>2.2.3',
        'read-roi'
    ],
    url='https://github.com/imodpasteur/Brightfield_Drift_Tracking_3D.git',
    license='BSD',
    author='Andrey Aristov',
    author_email='aaristov@pasteur.fr',
    description='Acquire bright field images along with the super resolution data and use it to track drift in 3D with nanometer precision!'
)
