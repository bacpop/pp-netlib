import re
from setuptools import setup

version = "0.0.1" #re.search(
#     '^__version__\s*=\s*"(.*)"',
#     open('__init__.py').read(),
#     re.M
#     ).group(1)

with open("README.md", "rb") as readme:
    full_description = readme.read().decode("utf-8")

setup(name = "poppunk_network_utils",
        packages = ["pp_netlib"],
        version = version, description = "PopPUNK network utilities.",
        long_description = full_description)