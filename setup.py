'''
Machine learning project for filename based media classification and named
entity recognition.
'''

from setuptools import setup

setup(name='src',
      packages=['src'],
      version='0.2.2',
      entry_points={'console_scripts': ['mc=src.cli:main']})
