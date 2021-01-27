from os import path

from setuptools import find_packages, setup

# this_directory = path.abspath(path.dirname(__file__))
# with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
#     long_description = f.read()

# extract version
path = path.realpath("microutil/_version.py")
version_ns = {}
with open(path, encoding="utf8") as f:
    exec(f.read(), {}, version_ns)
version = version_ns["__version__"]

name = "microutil"

setup_args = dict(
    name=name,
    version=version,
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "matplotlib>=3.3",
        "numpy",
        "pandas",
        "dask",
        "xarray",
        "tifffile",
        "read_roi",
    ],
    author="Ian Hunt-Isaak, John Russell",
    author_email="ianhuntisaak@g.harvard.edu, johnrussell@g.harvard.edu",
    license="BSD 3-Clause",
    platforms="Linux, Mac OS X, Windows",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    url="https://github.com/Hekstra-Lab/microutil",
    extras_require={
        "test": [
            "black",
        ],
    },
)

if __name__ == "__main__":
    setup(**setup_args)
