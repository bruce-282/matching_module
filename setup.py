#!/usr/bin/env python3
"""
Setup script for matching-module with dynamic TEASER++ path resolution
"""

import os
from pathlib import Path
from setuptools import setup, find_packages

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
TEASERPP_PATH = PROJECT_ROOT / "third_party" / "TEASER-plusplus"

# Read requirements
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Dynamic optional dependencies
optional_dependencies = {
    "teaserpp": [
        f"teaserpp-python @ file://{TEASERPP_PATH}",
    ]
}

setup(
    name="matching-module",
    version="0.1.0",
    description="Image matching module using Roma model",
    packages=find_packages(),
    install_requires=read_requirements(),
    extras_require=optional_dependencies,
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
