"""Autochecker matching module error handler.

reconstruction_module 의 ErrorCode 범주 패턴 (0/100/200/300/400/500) 그대로 따라 사내
일관성 유지.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


# 사내 모듈 번호 (3 자리). error code 의 MMMSEEE 인코딩에서 prefix 부분에 들어간다.
MODULE_NAME = "000"


class ErrorCode(Enum):
    """Error codes for autochecker matching operations."""

    # General errors (0-99)
    UNKNOWN = (0, "Unknown cause of error")
    NOT_INITIALIZED = (1, "Module not initialized")
    INVALID_PARAM = (5, "Invalid parameter")
    NOT_IMPLEMENTED = (8, "Not implemented")

    # System errors (100-199)
    SYSTEM_RESET_FAILED = (101, "Module reset failed")
    ENGINE_INIT_FAILED = (102, "Matcher engine initialization failed")

    # File I/O errors (200-299)
    FILE_NOT_FOUND = (201, "File not found")
    FILE_LOAD_FAILED = (202, "Failed to load file")
    SAVE_FAILED = (203, "Failed to save output")

    # Template errors (300-399)
    TEMPLATE_NOT_LOADED = (301, "Template not loaded — call req_load_config first")
    TEMPLATE_INVALID = (302, "Template parameters invalid")

    # Matching errors (400-499)
    MATCH_FAILED = (401, "Feature matching failed")
    INSUFFICIENT_MATCHES = (402, "Not enough matches found")
    HOMOGRAPHY_FAILED = (403, "Homography estimation failed")

    # Pose / depth errors (500-599)
    POSE_ESTIMATION_FAILED = (501, "Object pose estimation failed")
    DEPTH_CALCULATION_FAILED = (502, "Anchor depth calculation failed")
    DEPTH_OUT_OF_RANGE = (503, "Anchor depth outside stable range")
    SAFE_ZONE_VIOLATION = (504, "Anchor outside safe zone")

    def __init__(self, num: int, default_msg: str):
        self.num = num
        self.default_msg = default_msg

    def code_str(self, severity: str) -> int:
        """Generate numeric error code: MMMSEEE (reconstruction_module 패턴 그대로)."""
        mod_raw = str(globals().get("MODULE_NAME", "000"))
        mod_digits = "".join(ch for ch in mod_raw if ch.isdigit()) or "0"
        mod3 = mod_digits[-3:].zfill(3)
        mod_val = int(mod3)

        try:
            sev = int(severity)
        except (ValueError, TypeError):
            sev = 0
        sev = abs(sev) % 10

        try:
            num = int(self.num)
        except (ValueError, TypeError):
            num = 0
        num = abs(num) % 10000

        code = mod_val * 100000 + sev * 10000 + num
        return max(0, min(code, 0xFFFFFFFF))

    @property
    def default_message(self) -> str:
        return self.default_msg


class MatchingError(Exception):
    """Exception for matching operations."""

    def __init__(
        self,
        error_code: ErrorCode,
        severity: str,
        message: Optional[str] = None,
    ):
        self.error_code = error_code
        self.severity = severity
        self.message = message or error_code.default_message
        self.code = error_code.code_str(severity)
        super().__init__(self.message)

    def to_msg(self) -> str:
        return f"[{self.code}] {self.message}"

    def __str__(self) -> str:
        return self.to_msg()

    def __repr__(self) -> str:
        return (
            f"MatchingError({self.error_code.name}, "
            f"severity={self.severity}, message={self.message!r})"
        )
