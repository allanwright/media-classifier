from setuptools import setup

setup(name='src',
    packages=['src'],
    version='0.2.0',
    entry_points={
        'console_scripts': ['mc=src.cli:main']
    }
)