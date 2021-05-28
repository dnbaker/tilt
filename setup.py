from setuptools import setup, find_packages

setup(
    name='tilt',
    version='0.0.1',
    url='https://github.com/dnbaker/tilt.git',
    author='Daniel Baker',
    author_email='dnb@cs.jhu.edu',
    description='Tilt has data loaders and tuple miners for biased (and/)or biological datasets',
    packages=find_packages(),    
    install_requires=['numpy >= 1.11.1', 'torch'],
)
