"""Matcher 내부 파이프라인 신호용 예외.

``run_pipeline`` *내부*에서 실패를 사내 표준 ``ErrorCode`` (``error_handler``) 및
부가정보(``details``)와 함께 전달하기 위한 **내부 전용** 예외다. ``run_pipeline``
경계에서 잡혀 ``MatchResult`` 로 변환되며, 외부로 새지 않는다.

공개 계약은 ``MatchResult`` (``results``) 이고, 사내 표준 에러 코드/예외는
``error_handler`` (``ErrorCode``, ``MatchingError``) 를 사용한다.
"""

from typing import Any, Dict, Optional

from .error_handler import ErrorCode


class MatcherError(Exception):
    """파이프라인 내부 실패 신호 (ErrorCode + details 보유).

    Attributes:
        error_code: 사내 표준 실패 코드 (``ErrorCode``)
        message: 사람이 읽는 설명 (없으면 ``error_code`` 기본 메시지)
        details: 추가 정보 (예: safe zone 위반 시 {"point", "position"}).
            외부 처리/로깅에 사용. (사내 ``MatchingError`` 에는 details 가 없으므로
            ``MatchResult.details`` 로만 전달된다.)
    """

    def __init__(
        self,
        error_code: ErrorCode,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.error_code = error_code
        self.message = message or error_code.default_message
        self.details = details or {}
        super().__init__(f"[{error_code.name}] {self.message}")
