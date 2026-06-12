"""Matcher 도메인 예외 및 에러 코드.

외부(호출 측)에서 실패 유형을 **코드로 구분**해 처리할 수 있도록, 매처의 실패는
``MatcherError`` 로 raise 되며 구조화된 ``code``(``MatcherErrorCode``) 와 ``details``
를 가진다.

예시::

    from core.matchers.errors import MatcherError, MatcherErrorCode

    try:
        result = matcher.run_pipeline(...)
    except MatcherError as e:
        if e.code == MatcherErrorCode.SAFE_ZONE_VIOLATION:
            # e.details == {"point": "L", "position": [x, y, z]}
            ...  # 매칭 실패로 처리 (로봇 이동 금지 등)
"""

from enum import Enum
from typing import Any, Dict, Optional


class MatcherErrorCode(str, Enum):
    """매처 실패 유형 코드. (str 기반이라 로그/JSON 직렬화에 그대로 사용 가능)"""

    SAFE_ZONE_VIOLATION = "SAFE_ZONE_VIOLATION"  # anchor 가 safe zone 을 벗어남


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
