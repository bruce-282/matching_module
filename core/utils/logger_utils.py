"""
로거 유틸리티 함수들
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = __name__,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    date_format: str = '%H:%M:%S',
    include_file_info: bool = True
) -> logging.Logger:
    """
    로거를 설정하고 반환합니다.
    
    Args:
        name: 로거 이름 (기본값: __name__)
        level: 로그 레벨 (기본값: logging.INFO)
        format_string: 커스텀 포맷 문자열 (기본값: None)
        date_format: 날짜 포맷 (기본값: '%H:%M:%S')
        include_file_info: 파일명과 라인 번호 포함 여부 (기본값: True)
    
    Returns:
        설정된 로거 객체
    """
    # 기본 포맷 설정
    if format_string is None:
        if include_file_info:
            format_string = '[%(asctime)s] %(levelname)-6s %(filename)s:%(lineno)d: %(message)s'
        else:
            format_string = '[%(asctime)s] %(levelname)-6s: %(message)s'
    
    # 포맷터 생성
    formatter = logging.Formatter(
        fmt=format_string,
        datefmt=date_format
    )
    
    # 핸들러 생성 및 설정
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # 로거 생성 및 설정
    logger = logging.getLogger(name)
    logger.handlers.clear()  # 기존 핸들러 제거
    logger.addHandler(handler)
    logger.setLevel(level)
    
    # 상위 로거로 전파 방지
    logger.propagate = False
    
    return logger


def setup_file_logger(
    name: str = __name__,
    level: int = logging.INFO,
    log_file: str = 'app.log',
    format_string: Optional[str] = None,
    date_format: str = '%Y-%m-%d %H:%M:%S',
    include_file_info: bool = True
) -> logging.Logger:
    """
    파일 로거를 설정하고 반환합니다.
    
    Args:
        name: 로거 이름 (기본값: __name__)
        level: 로그 레벨 (기본값: logging.INFO)
        log_file: 로그 파일 경로 (기본값: 'app.log')
        format_string: 커스텀 포맷 문자열 (기본값: None)
        date_format: 날짜 포맷 (기본값: '%Y-%m-%d %H:%M:%S')
        include_file_info: 파일명과 라인 번호 포함 여부 (기본값: True)
    
    Returns:
        설정된 로거 객체
    """
    # 기본 포맷 설정
    if format_string is None:
        if include_file_info:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        else:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 포맷터 생성
    formatter = logging.Formatter(
        fmt=format_string,
        datefmt=date_format
    )
    
    # 파일 핸들러 생성 및 설정
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 로거 생성 및 설정
    logger = logging.getLogger(name)
    logger.handlers.clear()  # 기존 핸들러 제거
    logger.addHandler(file_handler)
    logger.setLevel(level)
    
    # 상위 로거로 전파 방지
    logger.propagate = False
    
    return logger


def setup_dual_logger(
    name: str = __name__,
    level: int = logging.INFO,
    log_file: str = 'app.log',
    console_format: Optional[str] = None,
    file_format: Optional[str] = None,
    console_date_format: str = '%H:%M:%S',
    file_date_format: str = '%Y-%m-%d %H:%M:%S',
    include_file_info: bool = True
) -> logging.Logger:
    """
    콘솔과 파일에 동시에 로그를 출력하는 로거를 설정합니다.
    
    Args:
        name: 로거 이름 (기본값: __name__)
        level: 로그 레벨 (기본값: logging.INFO)
        log_file: 로그 파일 경로 (기본값: 'app.log')
        console_format: 콘솔 포맷 문자열 (기본값: None)
        file_format: 파일 포맷 문자열 (기본값: None)
        console_date_format: 콘솔 날짜 포맷 (기본값: '%H:%M:%S')
        file_date_format: 파일 날짜 포맷 (기본값: '%Y-%m-%d %H:%M:%S')
        include_file_info: 파일명과 라인 번호 포함 여부 (기본값: True)
    
    Returns:
        설정된 로거 객체
    """
    # 콘솔 포맷 설정
    if console_format is None:
        if include_file_info:
            console_format = '[%(asctime)s] %(levelname)-6s %(filename)s:%(lineno)d: %(message)s'
        else:
            console_format = '[%(asctime)s] %(levelname)-6s: %(message)s'
    
    # 파일 포맷 설정
    if file_format is None:
        if include_file_info:
            file_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        else:
            file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 콘솔 포맷터
    console_formatter = logging.Formatter(
        fmt=console_format,
        datefmt=console_date_format
    )
    
    # 파일 포맷터
    file_formatter = logging.Formatter(
        fmt=file_format,
        datefmt=file_date_format
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    
    # 로거 생성 및 설정
    logger = logging.getLogger(name)
    logger.handlers.clear()  # 기존 핸들러 제거
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(level)
    
    # 상위 로거로 전파 방지
    logger.propagate = False
    
    return logger


# 편의 함수들
def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """기본 로거를 반환합니다."""
    return setup_logger(name, level)


def get_debug_logger(name: str = __name__) -> logging.Logger:
    """디버그 레벨 로거를 반환합니다."""
    return setup_logger(name, logging.DEBUG)


def get_file_logger(name: str = __name__, log_file: str = 'app.log') -> logging.Logger:
    """파일 로거를 반환합니다."""
    return setup_file_logger(name, logging.INFO, log_file)
