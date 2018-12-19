from setuptools import setup, find_packages

name = 'find_phone'

packages = find_packages()

setup(
    name=name,
    version='0.0.1',
    packages=packages,
    include_package_data=True,
    install_requires=[],
)
