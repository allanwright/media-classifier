from setuptools import setup

setup(name='src',
      packages=['src'],
      version='0.0.1dev1',
      entry_points={
            'console_scripts': ['mc-cli=src.cli:main']
      }
)