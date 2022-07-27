from gettext import install

from setuptools import find_packages, setup

with open("../README.md", "r") as file:
    description = file.read()

print(find_packages())
setup(
    name='JSparse',
    version='0.1',
    description=description,
    packages=find_packages(),
    install_requires=["jittor", "type_check"]
)