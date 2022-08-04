from importlib.metadata import entry_points
import re
from setuptools import setup

version = re.search(
    '^__version__\s*=\s*"(.*)"',
    open('__init__.py').read(),
    re.M
    ).group(1)

with open("README.md", "rb") as readme:
    full_description = readme.read().decode("utf-8")

setup(name = "poppunk_network_utils", packages = ["pop_net_utils"], entry_points = {"console_scripts": ["run_network = pop_net_utils.__main__:main"]}, version = version, description = "PopPUNK network utilities.", long_description = full_description)