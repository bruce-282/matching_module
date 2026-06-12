"""Matcher 도메인 에러 코드 및 (내부) 예외.

외부(호출 측)에서 실패 유형을 **코드로 구분**할 수 있도록 ``MatcherErrorCode`` 를 둔다.

공개 계약(권장): ``run_pipeline`` 은 예외를 던지지 않고 항상 ``MatchResult``
(``core.matchers.results``) 를 반환한다. 호출 측은 try/except 없이 ``success`` 로
분기하고, 실패 시 ``error_code`` 로 원인을 구분한다::

    from core.matchers.results import MatchResult
    from core.matchers.errors import MatcherErrorCode

    res = matcher.run_pipeline(...)
    if not res.success:
        if res.error_code == MatcherErrorCode.SAFE_ZONE_VIOLATION:
            # res.details == {"point": "L", "position": [x, y, z]}
            ...  # 매칭 실패로 처리 (로봇 이동 금지 등)

``MatcherError`` 는 파이프라인 **내부**에서 실패를 코드와 함께 전달하기 위한 예외이며,
``run_pipeline`` 경계에서 잡혀 ``MatchResult`` 로 변환된다. (외부로 새지 않는다.)
"""

from enum import Enum
from typing import Any, Dict, Optional


class MatcherErrorCode(str, Enum):
    """매처 실패 유형 코드. (str 기반이라 로그/JSON 직렬화에 그대로 사용 가능)"""

    SAFE_ZONE_VIOLATION = "SAFE_ZONE_VIOLATION"  # anchor 가 safe zone 을 벗어남
    MATCHING_FAILED = "MATCHING_FAILED"  # 그 외 매칭/깊이 계산 실패 (일반 실패)


class MatcherError(Exception):
    """코드를 가진 매처 도메인 예외.

    Attributes:
        code: 실패 유형 코드 (``MatcherErrorCode``)
        message: 사람이 읽는 설명
        details: 추가 정보 (예: 어떤 포인트, 좌표 등). 외부 처리/로깅에 사용.
    """

    def __init__(
        self,
        code: "MatcherErrorCode",
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"[{code.value}] {message}")
