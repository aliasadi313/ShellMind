from setuptools import setup, find_packages

setup(
    name="shellmind",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'shellmind=shellmind.cli:main',
        ],
    },
    install_requires=[
        'openai',
        'PyYAML'
    ],
    extras_require={
        'test': [
            'pytest>=7.0',
            'pytest-cov>=3.0',
            'mock>=4.0',
        ],
    },
)