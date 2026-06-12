"""Matcher 파이프라인 결과 객체.

``run_pipeline`` 은 예외를 던지지 않고 **항상 ``MatchResult`` 를 반환**한다.
상위(in-process) 호출 측은 try/except 없이 ``success`` 플래그로 분기하고, 실패 시
사내 표준 ``error_code`` (``ErrorCode``) 로 원인을 구분한다.

예시::

    from core.matchers.results import MatchResult
    from core.matchers.error_handler import ErrorCode

    res = matcher.run_pipeline(...)
    if not res.success:
        if res.error_code == ErrorCode.SAFE_ZONE_VIOLATION:
            bad = res.details["point"]      # "L" 또는 "R"
            pos = res.details["position"]   # [x, y, z]
            ...  # 매칭 실패로 처리 (로봇 이동 금지 등)
        return
    use(res.point_l, res.point_r, res.point_u, res.plane_normal)

crp_core 응답/로깅에서 사내 표준 예외가 필요하면 ``res.to_error()`` 로 ``MatchingError``
를, 숫자 코드는 ``res.code`` (MMMSEEE) 로 얻는다.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .error_handler import ErrorCode, MatchingError

# 실패 결과의 기본 severity. code_str(MMMSEEE) 의 S 자리에 들어간다.
# TODO: reconstruction_module 의 severity 규약에 맞춰 확정/세분화.
DEFAULT_SEVERITY = "2"


@dataclass
class MatchResult:
    """매칭 파이프라인의 단일 결과.

    Attributes:
        success: 성공 여부. True 면 point_* / plane_normal 이 채워지고,
            False 면 error_code / error_message / details 가 채워진다.
        point_l, point_r, point_u: anchor 3D 좌표 [x, y, z] (성공 시)
        plane_normal: 평면 법선 (성공 시)
        error_code: 사내 표준 실패 코드 (실패 시, ``ErrorCode``)
        error_message: 사람이 읽는 실패 설명 (실패 시)
        severity: 실패 심각도 (``MatchingError``/숫자 코드 생성용)
        details: 추가 정보 (예: safe zone 위반 시 {"point", "position"})
    """

    success: bool
    point_l: Optional[np.ndarray] = None
    point_r: Optional[np.ndarray] = None
    point_u: Optional[np.ndarray] = None
    plane_normal: Optional[np.ndarray] = None
    error_code: Optional[ErrorCode] = None
    error_message: Optional[str] = None
    severity: str = DEFAULT_SEVERITY
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        point_l: np.ndarray,
        point_r: np.ndarray,
        point_u: np.ndarray,
        plane_normal: np.ndarray,
    ) -> "MatchResult":
        """성공 결과 생성."""
        return cls(
            success=True,
            point_l=point_l,
            point_r=point_r,
            point_u=point_u,
            plane_normal=plane_normal,
        )

    @classmethod
    def fail(
        cls,
        error_code: ErrorCode,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = DEFAULT_SEVERITY,
    ) -> "MatchResult":
        """실패 결과 생성. message 가 없으면 error_code 기본 메시지를 쓴다."""
        return cls(
            success=False,
            error_code=error_code,
            error_message=message or error_code.default_message,
            severity=severity,
            details=details or {},
        )

    @property
    def code(self) -> Optional[int]:
        """사내 숫자 에러 코드(MMMSEEE). 실패 시에만 의미가 있다."""
        if self.error_code is None:
            return None
        return self.error_code.code_str(self.severity)

    def to_error(self) -> MatchingError:
        """사내 표준 예외(``MatchingError``)로 변환. (crp_core 응답/로깅용)

        성공 결과에 호출하면 안 된다(error_code 가 None).
        """
        if self.error_code is None:
            raise ValueError("Cannot convert a successful MatchResult to MatchingError")
        return MatchingError(self.error_code, self.severity, self.error_message)
