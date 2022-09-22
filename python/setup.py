from gettext import install

from setuptools import find_packages, setup

from JSparse import __version__

with open("../README.md", "r") as file:
    description = file.read()

print(find_packages())
setup(
    name='JSparse',
    version=__version__,
    description=description,
    packages=find_packages(),
    install_requires=["jittor"]
)