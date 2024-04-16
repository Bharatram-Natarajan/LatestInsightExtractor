from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="ConversationalInformationExtractor",
    version="1.0.0",
    description="Python Package to extract important insights from many conversations.",
    long_description=long_description,
    author="Bharatram Natarajan",
    author_email="bharatram.natarjan@freshworks.com",
    packages=find_packages(),
    install_requires=required
)
