from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent
long_description = ""
readme = here / "README.md"
if readme.exists():
    long_description = readme.read_text(encoding="utf-8")

setup(
    name="tecio",
    version="0.0.0",
    description="Utilities for reading/writing TecIO/PLT/SZL files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests", "test*")),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=["numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
