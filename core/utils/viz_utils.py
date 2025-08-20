"""
시각화 유틸리티
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple


def visualize_matches(
    image0_origin: np.ndarray,
    image1_origin: np.ndarray,
    keypoints0,
    keypoints1,
    confidence,
    output_path: Path,
    confidence_threshold: float = 0.5,
    circle_radius: int = 5,
    line_thickness: int = 1,
    circle_color: Tuple[int, int, int] = (0, 255, 0),  # 녹색
    line_color: Tuple[int, int, int] = (255, 0, 0),  # 파란색
):
    """매칭 결과를 시각화합니다."""
    # 이미지 로드

    if image0_origin is None or image1_origin is None:
        raise ValueError(
            f"이미지를 로드할 수 없습니다: {image0_origin}, {image1_origin}"
        )

    # BGR to RGB 변환
    img0_rgb = cv2.cvtColor(image0_origin, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(image1_origin, cv2.COLOR_BGR2RGB)

    # 이미지 크기 조정 (높이 맞춤)
    h0, w0 = img0_rgb.shape[:2]
    h1, w1 = img1_rgb.shape[:2]

    # 두 이미지의 높이를 맞춤
    max_height = max(h0, h1)
    scale0 = 1.0
    scale1 = 1.0

    if h0 != max_height:
        scale0 = max_height / h0
        new_w0 = int(w0 * scale0)
        img0_rgb = cv2.resize(img0_rgb, (new_w0, max_height))
        w0 = new_w0
    if h1 != max_height:
        scale1 = max_height / h1
        new_w1 = int(w1 * scale1)
        img1_rgb = cv2.resize(img1_rgb, (new_w1, max_height))
        w1 = new_w1

    # 이미지 연결
    combined_img = np.hstack([img0_rgb, img1_rgb])

    # 키포인트 그리기 (스케일링 적용)
    match_count = 0
    for i, (kp0, kp1, conf) in enumerate(zip(keypoints0, keypoints1, confidence)):
        if conf > confidence_threshold:
            match_count += 1
            # 첫 번째 이미지의 키포인트 (스케일링 적용)
            x0, y0 = int(kp0[0] * scale0), int(kp0[1] * scale0)
            cv2.circle(combined_img, (x0, y0), circle_radius, circle_color, -1)

            # 두 번째 이미지의 키포인트 (스케일링 적용, x 좌표에 w0 더함)
            x1, y1 = int(kp1[0] * scale1) + w0, int(kp1[1] * scale1)
            cv2.circle(combined_img, (x1, y1), circle_radius, circle_color, -1)

            # 매칭 선 그리기
            cv2.line(combined_img, (x0, y0), (x1, y1), line_color, line_thickness)

    # 결과 저장
    combined_img_bgr = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, combined_img_bgr)
    print(f"매칭 결과가 {output_path}에 저장되었습니다.")
    print(f"총 {match_count}개의 매칭이 시각화되었습니다.")

    return combined_img_bgr


def visualize_keypoints(
    image_path,
    keypoints,
    output_path="keypoints_visualization.png",
    circle_radius=5,
    color=(0, 255, 0),
    thickness=-1,
):
    """키포인트를 시각화합니다."""
    # 이미지 로드
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

    # 키포인트 그리기
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(img, (x, y), circle_radius, color, thickness)

    # 결과 저장
    cv2.imwrite(output_path, img)
    print(f"키포인트 시각화가 {output_path}에 저장되었습니다.")
    print(f"총 {len(keypoints)}개의 키포인트가 표시되었습니다.")

    return img


def create_matching_animation(
    image0_origin: np.ndarray,
    image1_origin: np.ndarray,
    keypoints0,
    keypoints1,
    confidence,
    output_path: Path,
    confidence_threshold: float = 0.5,
    fps: int = 10,
):
    """매칭 애니메이션을 생성합니다."""
    # 이 기능은 추가 라이브러리가 필요할 수 있습니다
    print("애니메이션 기능은 향후 구현 예정입니다.")
    print("현재는 정적 이미지로 시각화합니다.")

    return visualize_matches(
        image0_origin,
        image1_origin,
        keypoints0,
        keypoints1,
        confidence,
        output_path,
        confidence_threshold,
    )
