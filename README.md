# Image Matching Module

Roma 모델을 사용한 이미지 매칭 및 RANSAC 필터링 모듈입니다.

## 설치 방법

### 1. 저장소 클론
```bash
git clone <repository-url>
cd matching_module
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows
```

### 3. PyTorch 설치 (CUDA 지원)
```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

### 4. 나머지 의존성 설치
```bash
pip install -r requirements.txt
```

## 구동 방법

### 기본 사용법
```bash
python run_matcher.py --source datasets/source.png --target datasets/target.png
```

### 모든 옵션을 사용한 예시
```bash
python run_matcher.py \
  --source datasets/source.png \
  --target datasets/target.png \
  --output_dir output \
  --max_keypoints 2000 \
  --ransac_method CV2_USAC_MAGSAC \
  --ransac_reproj_threshold 8.0 \
  --ransac_confidence 0.9999 \
  --debug \
  --offset_point1_x 0.5 \
  --offset_point1_y 0.92 \
  --offset_point2_x 1.4 \
  --offset_point2_y 0.92 \
  --point_radius 10
```

## 명령행 인자

### 필수 인자
- `--source`: 첫 번째 이미지 경로 (기본값: datasets/source.png)
- `--target`: 두 번째 이미지 경로 (기본값: datasets/target.png)

### 선택적 인자
- `--output_dir`: 결과 저장 디렉토리 (기본값: output)
- `--max_keypoints`: 최대 키포인트 수 (기본값: 2000)
- `--ransac_method`: RANSAC 메서드 (기본값: CV2_USAC_MAGSAC)
- `--ransac_reproj_threshold`: RANSAC 재투영 임계값 (기본값: 8.0)
- `--ransac_confidence`: RANSAC 신뢰도 (기본값: 0.9999)
- `--debug`: 디버그 모드 활성화 (파일 저장 및 상세 로그)
- `--offset_point1_x`: 첫 번째 포인트 X 좌표 비율 (기본값: 0.5)
- `--offset_point1_y`: 첫 번째 포인트 Y 좌표 비율 (기본값: 0.92)
- `--offset_point2_x`: 두 번째 포인트 X 좌표 비율 (기본값: 1.4)
- `--offset_point2_y`: 두 번째 포인트 Y 좌표 비율 (기본값: 0.92)
- `--point_radius`: 포인트 반지름 (기본값: 10)

## 입력 파일

### 이미지 파일
- **형식**: PNG, JPG, JPEG 등 OpenCV가 지원하는 모든 이미지 형식
- **크기**: 제한 없음 (자동으로 resize됨)
- **채널**: RGB 또는 그레이스케일 (자동 변환)

### 예시 디렉토리 구조
```
datasets/
├── source.png
└── target.png
```

## 출력 파일

### 기본 출력 (항상 생성)
- `{source_name}_result.yaml`: 변환된 포인트 위치 정보

### 디버그 모드 출력 (--debug 옵션 사용 시)
- `{source_name}_matches_original.png`: Roma 매칭 결과 시각화
- `{source_name}_matches_ransac_filtered.png`: RANSAC 필터링 후 결과 시각화
- `{source_name}_warped_overlapped.png`: 이미지 변환 및 오버레이 결과

### 출력 디렉토리 구조
```
output/
├── source_result.yaml                    # 포인트 위치 데이터
├── source_matches_original.png           # Roma 매칭 결과 (debug 모드)
├── source_matches_ransac_filtered.png    # RANSAC 필터링 결과 (debug 모드)
└── source_warped_overlapped.png          # 변환된 이미지 (debug 모드)
```

## 출력 파일 설명

### YAML 파일 ({source_name}_result.yaml)
변환된 포인트 위치 정보를 포함하는 구조화된 데이터 파일입니다.

```yaml
source_image: source.png
image_size:
  width: 1920
  height: 1080
transformed_points:
  pointL:
    x: 507
    y: 972
  pointR:
    x: 1420
    y: 972
```

### 디버그 모드 이미지 파일들

#### 1. {source_name}_matches_original.png
- Roma 모델의 원본 매칭 결과를 시각화
- 두 이미지 간의 키포인트 매칭을 선으로 표시
- 신뢰도 임계값 이상의 매칭만 표시

#### 2. {source_name}_matches_ransac_filtered.png
- RANSAC 필터링을 거친 후의 매칭 결과
- 기하학적으로 일관성 있는 매칭만 표시
- 노이즈가 제거된 깔끔한 매칭 결과

#### 3. {source_name}_warped_overlapped.png
- Homography 변환을 적용한 이미지 오버레이
- 두 번째 이미지를 첫 번째 이미지에 투영하여 합성
- 사용자가 설정한 포인트 위치에 빨간색 원으로 표시
- 오버레이 영역에 빨간색 틴트 적용

## 포인트 설정

사용자는 `--offset_point1_x`, `--offset_point1_y`, `--offset_point2_x`, `--offset_point2_y` 인자를 통해 변환된 이미지에서 표시할 포인트의 위치를 설정할 수 있습니다.

## 로그 레벨

- **INFO 레벨**: 시간 측정 결과 및 주요 진행 상황
- **DEBUG 레벨**: 상세한 처리 과정 및 중간 결과 (--debug 옵션 사용 시)

## 문제 해결

### 일반적인 오류
1. **매칭 실패**: 이미지 품질이나 특징점이 부족할 수 있습니다
2. **RANSAC 실패**: `--ransac_reproj_threshold` 값을 조정해보세요

### 성능 최적화
- GPU 사용 시 더 빠른 처리 속도
- `--max_keypoints` 값을 조정하여 속도와 정확도 균형 조절
- 디버그 모드는 파일 저장으로 인해 처리 시간이 증가할 수 있습니다
