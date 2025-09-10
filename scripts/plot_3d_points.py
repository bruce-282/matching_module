#!/usr/bin/env python3
"""
3D 포인트 데이터 시각화 스크립트
YAML 파일에서 transformed_points_3d 데이터를 추출하여 그래프로 표시
"""

import yaml
import matplotlib

matplotlib.use("Agg")  # GUI 없이 백엔드 사용
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from pathlib import Path
import argparse


def load_yaml_data(file_path):
    """YAML 파일에서 3D 포인트 데이터를 로드"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "transformed_points_3d" not in data:
            print(f"경고: {file_path}에 transformed_points_3d 데이터가 없습니다.")
            return None

        return data["transformed_points_3d"]
    except Exception as e:
        print(f"오류: {file_path} 로드 실패 - {e}")
        return None


def extract_point_data(points_3d):
    """3D 포인트 데이터에서 좌표 추출"""
    if not points_3d:
        return None

    data = {}

    # plane_normal 추출
    if "plane_normal" in points_3d:
        normal = points_3d["plane_normal"]
        data["plane_normal"] = [normal["x"], normal["y"], normal["z"]]

    # 각 포인트 추출
    for point_name in ["pointL", "pointR", "pointU"]:
        if point_name in points_3d:
            point = points_3d[point_name]
            data[point_name] = [point["x"], point["y"], point["z"]]

    return data


def plot_3d_data(all_data, output_dir):
    """3D 데이터를 그래프로 시각화"""

    # 데이터가 없으면 종료
    if not all_data:
        print("시각화할 데이터가 없습니다.")
        return

    # 인덱스별로 데이터 정리
    indices = list(all_data.keys())
    indices.sort()

    # pointL, pointR, pointU만 처리
    data_types = ["pointL", "pointR", "pointU"]

    for data_type in data_types:
        if not any(data_type in data for data in all_data.values()):
            continue

        # 하나의 그래프에 X, Y, Z 3개 라인 생성
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f"{data_type} - Coordinate Changes by Index", fontsize=16)

        # 각 좌표별로 데이터 수집
        coord_data = {"X": [], "Y": [], "Z": []}
        valid_indices = []

        for idx in indices:
            if data_type in all_data[idx]:
                point = all_data[idx][data_type]
                coord_data["X"].append(point[0])
                coord_data["Y"].append(point[1])
                coord_data["Z"].append(point[2])
                valid_indices.append(idx)

        # X, Y, Z 좌표를 하나의 그래프에 표시
        coord_names = ["X", "Y", "Z"]
        colors = ["red", "green", "blue"]

        for i, (coord_name, values) in enumerate(coord_data.items()):
            if values:  # 값이 있는 경우만 플롯
                ax.plot(
                    valid_indices,
                    values,
                    "o-",
                    color=colors[i],
                    linewidth=3,
                    markersize=10,
                    markerfacecolor=colors[i],
                    markeredgecolor="black",
                    markeredgewidth=1,
                    label=f"{coord_name} Coordinate",
                )

                # 각 점에 값 표시
                for j, (idx, val) in enumerate(zip(valid_indices, values)):
                    ax.annotate(
                        f"{val:.1f}",
                        (idx, val),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=6,
                        fontweight="bold",
                    )

        ax.set_xlabel("File Index", fontsize=12)
        ax.set_ylabel("Coordinate Value (mm)", fontsize=12)
        ax.set_title(f"{data_type}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(valid_indices)
        ax.legend(fontsize=12)

        # Y축 범위를 조금 여유있게 설정
        all_values = []
        for values in coord_data.values():
            all_values.extend(values)
        if all_values:
            y_min, y_max = min(all_values), max(all_values)
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

        # 저장
        # output_path = os.path.join(output_dir, f"{data_type}_2d_plots.png")
        # plt.savefig(output_path, dpi=300, bbox_inches="tight")
        # print(f"저장됨: {output_path}")

        plt.close()

        # 편차 정보가 포함된 그래프 생성
        create_error_bar_plot(data_type, coord_data, valid_indices, output_dir)


def create_error_bar_plot(data_type, coord_data, valid_indices, output_dir):
    """오차 막대를 포함한 편차 그래프 생성"""

    # 편차 계산을 위한 가상 데이터 생성 (실제로는 여러 측정값이 필요)
    # 현재는 2개 파일만 있으므로, 각 좌표별로 편차를 시뮬레이션

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f"{data_type} - Coordinate Changes", fontsize=16)

    coord_names = ["X", "Y", "Z"]
    colors = ["red", "green", "blue"]

    # 각 좌표별로 편차 계산 및 플롯
    for i, (coord_name, values) in enumerate(coord_data.items()):
        if len(values) >= 2:  # 편차 계산을 위해 최소 2개 값 필요
            # 표준편차 계산
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            std_dev = variance**0.5

            # 오차 막대는 표준편차의 2배로 설정 (95% 신뢰구간)
            error_bars = [std_dev * 2] * len(values)

            ax.errorbar(
                valid_indices,
                values,
                yerr=error_bars,
                fmt="o-",
                color=colors[i],
                linewidth=3,
                markersize=10,
                markerfacecolor=colors[i],
                markeredgecolor="black",
                markeredgewidth=1,
                capsize=8,
                capthick=2,
                elinewidth=2,
                label=f"{coord_name} Coordinate (σ=±{std_dev:.2f}mm)",
            )

            # 각 점에 값만 표시 (편차는 범례에만)
            for j, (idx, val) in enumerate(zip(valid_indices, values)):
                ax.annotate(
                    f"{val:.1f}",
                    (idx, val),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=6,
                    fontweight="bold",
                )
        else:
            # 값이 1개만 있는 경우 일반 플롯
            ax.plot(
                valid_indices,
                values,
                "o-",
                color=colors[i],
                linewidth=3,
                markersize=10,
                markerfacecolor=colors[i],
                markeredgecolor="black",
                markeredgewidth=1,
                label=f"{coord_name} Coordinate (single point)",
            )

            # 각 점에 값 표시
            for j, (idx, val) in enumerate(zip(valid_indices, values)):
                ax.annotate(
                    f"{val:.1f}",
                    (idx, val),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=6,
                    fontweight="bold",
                )

    ax.set_xlabel("File Index", fontsize=12)
    ax.set_ylabel("Coordinate Value (mm)", fontsize=12)
    ax.set_title(
        f"{data_type}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_xticks(valid_indices)
    ax.legend(fontsize=11)

    # Y축 범위를 최대값보다 1000 더해서 오차 막대 공간 확보
    all_values = []
    for values in coord_data.values():
        all_values.extend(values)
    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        # 최대값보다 1000을 더해서 오차 막대가 들어갈 공간 확보
        ax.set_ylim(y_min - 100, y_max + 1000)

    # 저장
    output_path = os.path.join(output_dir, f"{data_type}_error_bars.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"저장됨: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="3D 포인트 데이터 시각화")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="output",
        help="YAML 파일이 있는 디렉토리 (기본값: output)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/plots",
        help="그래프 저장 디렉토리 (기본값: output/plots)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.yaml",
        help="YAML 파일 패턴 (기본값: *.yaml)",
    )

    args = parser.parse_args()

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # YAML 파일 찾기
    yaml_files = glob.glob(os.path.join(args.input_dir, args.pattern))

    if not yaml_files:
        print(
            f"경고: {args.input_dir}에서 {args.pattern} 패턴의 파일을 찾을 수 없습니다."
        )
        return

    print(f"발견된 YAML 파일: {len(yaml_files)}개")
    for file in yaml_files:
        print(f"  - {file}")

    # 데이터 로드
    all_data = {}

    for i, yaml_file in enumerate(sorted(yaml_files)):
        print(f"\n로딩 중: {yaml_file}")
        points_3d = load_yaml_data(yaml_file)

        if points_3d:
            extracted_data = extract_point_data(points_3d)
            if extracted_data:
                all_data[i] = extracted_data
                print(f"  ✓ 데이터 추출 완료")
            else:
                print(f"  ✗ 데이터 추출 실패")
        else:
            print(f"  ✗ 파일 로드 실패")

    if not all_data:
        print("시각화할 유효한 데이터가 없습니다.")
        return

    print(f"\n총 {len(all_data)}개 파일의 데이터를 시각화합니다.")

    # 그래프 생성
    plot_3d_data(all_data, args.output_dir)

    print(f"\n모든 그래프가 {args.output_dir}에 저장되었습니다.")


if __name__ == "__main__":
    main()
