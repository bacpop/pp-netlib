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

setup(name = "poppunk_network_utils", packages = ["main_logic"], entry_points = {"console_scripts": ["greet = main_logic.greet:main", "Hey = main_logic.other_greet:other_greeting"]}, version = version, description = "Barebones python package.", long_description = full_description, author = "Bruhad Dave")