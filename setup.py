from setuptools import setup

# Pull in the README.md
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(
    name='amor-hyp-gp',
    version='0.1.1',
    description='Task-Agnostic Amortized Inference of Gaussian Process Hyperparameters (AHGP)',
    url='https://github.com/PrincetonLIPS/AHGP',
    author='Sulin Liu',
    author_email='sulinl@princeton.edu',
    license='MIT',
    packages=['ahgp','ahgp.nn','ahgp.gp','ahgp.inference'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: MIT License',
    ],
)