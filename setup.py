#!/usr/bin/env python3
"""Setup script for matching-module (PEP 621 metadata lives in pyproject.toml).

install_requires 는 여기서 읽지 않는다. open() 기본 인코딩이 Windows(cp949)일 때
requirements.txt(UTF-8)에서 UnicodeDecodeError 가 나기 때문이다.
의존성은 pyproject.toml 의 [tool.setuptools.dynamic] dependencies = { file = [...] } 만 사용.
"""

from setuptools import find_packages, setup


def _discover_packages():
    """core/datasets + vendored RoMa(romatch). third_party 루트 패키지는 제외."""
    base = find_packages(
        include=("core*", "datasets*"),
        exclude=("third_party",),
    )
    roma = find_packages(where="third_party/RoMa", include=("romatch*",))
    return list(dict.fromkeys(base + roma))


setup(
    name="matching-module",
    version="0.1.0",
    description="Image matching module using Roma model",
    packages=_discover_packages(),
    python_requires=">=3.9,<3.12",
    include_package_data=True,
    package_data={
        "*": ["*.py", "*.pth", "*.ckpt"],
        # romatch 패키지 루트가 아닌 하위 weights/ 는 명시해야 휠에 포함됨
        "romatch": [
            "weights/*.pth",
            "weights/*.pt",
            "weights/*.ckpt",
            "weights/*.txt",
            "weights/*.sh",
        ],
    },
    entry_points={
        "console_scripts": [
            "roma-match=core.matchers.roma:main",
        ],
    },
)
