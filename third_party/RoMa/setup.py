from pathlib import Path

from setuptools import find_packages, setup


def _req_lines():
    path = Path(__file__).with_name("requirements.txt")
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


setup(
    name="romatch",
    packages=find_packages(include=("romatch*",)),
    version="0.0.2",
    author="Johan Edstedt",
    install_requires=_req_lines(),
)
