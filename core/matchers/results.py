"""Matcher 파이프라인 결과 객체.

``run_pipeline`` 은 예외를 던지지 않고 **항상 ``MatchResult`` 를 반환**한다.
상위(in-process) 호출 측은 try/except 없이 ``success`` 플래그로 분기하고, 실패 시
``error_code`` 로 원인을 구분한다.

예시::

    from core.matchers.results import MatchResult
    from core.matchers.errors import MatcherErrorCode

    res = matcher.run_pipeline(...)
    if not res.success:
        if res.error_code == MatcherErrorCode.SAFE_ZONE_VIOLATION:
            bad = res.details["point"]      # "L" 또는 "R"
            pos = res.details["position"]   # [x, y, z]
            ...  # 매칭 실패로 처리 (로봇 이동 금지 등)
        return
    use(res.point_l, res.point_r, res.point_u, res.plane_normal)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .errors import MatcherErrorCode


@dataclass
class MatchResult:
    """매칭 파이프라인의 단일 결과.

    Attributes:
        success: 성공 여부. True 면 point_* / plane_normal 이 채워지고,
            False 면 error_code / error_message / details 가 채워진다.
        point_l, point_r, point_u: anchor 3D 좌표 [x, y, z] (성공 시)
        plane_normal: 평면 법선 (성공 시)
        error_code: 실패 유형 코드 (실패 시, ``MatcherErrorCode``)
        error_message: 사람이 읽는 실패 설명 (실패 시)
        details: 추가 정보 (예: safe zone 위반 시 {"point", "position"})
    """

    success: bool
    point_l: Optional[np.ndarray] = None
    point_r: Optional[np.ndarray] = None
    point_u: Optional[np.ndarray] = None
    plane_normal: Optional[np.ndarray] = None
    error_code: Optional[MatcherErrorCode] = None
    error_message: Optional[str] = None
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
        code: MatcherErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> "MatchResult":
        """실패 결과 생성."""
        return cls(
            success=False,
            error_code=code,
            error_message=message,
            details=details or {},
        )
