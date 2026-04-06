#!/usr/bin/env python3
"""Setup script for matching-module (PEP 621 metadata lives in pyproject.toml)."""

from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="matching-module",
    version="0.1.0",
    description="Image matching module using Roma model",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.9,<3.12",
    include_package_data=True,
    package_data={
        "*": ["*.py", "*.pth", "*.ckpt"],
    },
    entry_points={
        "console_scripts": [
            "roma-match=core.matchers.roma:main",
        ],
    },
)
