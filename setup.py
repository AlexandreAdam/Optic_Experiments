from setuptools import setup, find_packages

setup(
        name = "PHY3040",
        version = "1.0",
        packages = find_packages(exclude=['contrib', 'docs', 'tests']),
        install_requires=[
            "numpy==1.22.0"
            ]
        )
        
