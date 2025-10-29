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

### 기본 사용법 (YAML 설정 파일 기반)

```bash
python run_matcher.py \
  --config_path configs/DefualtModel/matcher.config.yaml \
  --template_param_path configs/DefualtModel/matcher.teaching.param.yaml
```

### 설정 파일 구조

프로그램은 두 개의 YAML 설정 파일을 필요로 합니다:

1. **설정 파일 (--config_path)**: 매칭 파라미터, 카메라 설정, RANSAC 옵션 등
2. **템플릿 파라미터 파일 (--template_param_path)**: 소스 이미지 경로 및 매칭 포인트 3D 좌표

### 예시 디렉토리 구조
```
datasets/
└── 0910/                          # 입력 디렉토리 (config에서 설정)
    ├── image1_texture.png         # Texture 이미지 (매칭용)
    ├── image1_depth.tif           # Depth 이미지 (3D 계산용)
    ├── image2_texture.png
    ├── image2_depth.tif
    └── ...

configs/
└── DefualtModel/
    ├── matcher.config.yaml        # 메인 설정 파일
    └── matcher.teaching.param.yaml # 템플릿 파라미터 파일
```

## 명령행 인자

### 필수 인자
- `--config_path`: 설정 파일 경로 (YAML 형식)
- `--template_param_path`: 템플릿 파라미터 파일 경로 (YAML 형식)

## 설정 파일 설명

### 설정 파일 (matcher.config.yaml)

주요 설정 항목:

```yaml
# 입력/출력 디렉토리
input_dir: "datasets/0910/"      # 입력 디렉토리 (여기서 *_texture.png, *_depth.tif 찾음)
output_dir: "output0910"          # 결과 저장 디렉토리

# 매칭 파라미터
max_keypoints: 5000                # 최대 키포인트 수
geometry_type: "Homography"        # 기하학적 변환 타입
ransac_method: "CV2_USAC_MAGSAC"   # RANSAC 메서드
ransac_reproj_threshold: 13.0      # RANSAC 재투영 임계값
ransac_max_iter: 30000             # RANSAC 최대 반복 횟수
ransac_confidence: 0.9999          # RANSAC 신뢰도
debug_mode: true                   # 디버그 모드 활성화

# 3D 매칭 설정
enable_3d_matching: true           # 3D 매칭 활성화
stable_depth_range: 50.0           # 안정적인 depth 범위 (±mm)
pose_estimation_method: "ransac"    # 포즈 추정 방법 ("svd" 또는 "ransac")

# 카메라 설정
camera_intrinsics:
  fx: 2344.06988494
  fy: 2344.40009342502
  cx: 989.06314625513
  cy: 807.02989528271
camera_distortions:
  k1: -0.24331290305526787
  k2: 0.13922919417642093
  p1: 0.0005252878633098153
  p2: -0.0010237886757940777
  k3: -0.01443719970450923
image_size:
  width: 2064
  height: 1544
image_undistortion: true           # 이미지 왜곡 보정 활성화

# Depth 설정
depth_max: 2500.0                  # Depth map 최대 값
```

### 템플릿 파라미터 파일 (matcher.teaching.param.yaml)

매칭에 사용되는 소스 이미지와 3D 포인트 좌표를 정의합니다:

```yaml
# 소스 이미지 경로 (매칭에 사용될 템플릿 이미지)
path_match_source: "datasets/source_0919.tif"

# 매칭 포인트의 3D 좌표 (L, R, U 세 점)
selected_points:
  L:
    x: -28.02078685152369
    y: 322.8911333002952
    z: 1312.000036239624
  R:
    x: 402.44106120018216
    y: 439.71793461742914
    z: 1562.000036239624
  U:
    x: 184.56082806030466
    y: 261.9381813145888
    z: 1486.9999885559082
```

## 입력 파일

### 이미지 파일

프로그램은 `input_dir`에서 `*_texture.png`와 `*_depth.tif` 파일 쌍을 자동으로 찾습니다.

**Texture 이미지 (`*_texture.png`):**
- **용도**: Roma 매칭에 사용되는 이미지
- **형식**: PNG, JPG, JPEG 등 OpenCV/PIL이 지원하는 모든 이미지 형식
- **특징**: 매칭 성능 향상을 위해 명확한 특징점이 있는 이미지 권장

**Depth 이미지 (`*_depth.tif`):**
- **용도**: 3D 포인트 계산에 사용되는 depth map
- **형식**: 32비트 TIFF 파일 (PIL을 통한 로드)
- **특징**: depth_max 값보다 큰 영역은 texture 이미지 값으로 보완됨

**소스 이미지 (`path_match_source`):**
- **용도**: 템플릿 파라미터 파일에서 지정된 매칭 소스 이미지
- **형식**: PNG, JPG, JPEG, TIFF 등

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

매칭에 사용되는 포인트는 템플릿 파라미터 파일(`matcher.teaching.param.yaml`)의 `selected_points` 섹션에서 정의됩니다.

- **L (Left)**: 왼쪽 포인트의 3D 좌표 (x, y, z)
- **R (Right)**: 오른쪽 포인트의 3D 좌표 (x, y, z)
- **U (Up)**: 상단 포인트의 3D 좌표 (x, y, z)

이 세 점은 매칭 대상 객체의 위치를 정의하며, 카메라 위치 및 뷰 계산에 사용됩니다. 카메라 거리는 L과 R 포인트 간의 거리를 기준으로 계산됩니다 (거리 × 3.0).

## 로그 레벨

- **INFO 레벨**: 시간 측정 결과 및 주요 진행 상황
- **DEBUG 레벨**: 상세한 처리 과정 및 중간 결과 (설정 파일에서 `debug_mode: true`로 설정 시)

## 문제 해결

### 일반적인 오류
1. **매칭 실패**: 이미지 품질이나 특징점이 부족할 수 있습니다
2. **RANSAC 실패**: 설정 파일의 `ransac_reproj_threshold` 값을 조정해보세요
3. **템플릿 파라미터 오류**: `template_param_path` 파일이 올바르게 설정되었는지 확인하세요
4. **입력 파일 없음**: `input_dir`에 `*_texture.png`와 `*_depth.tif` 파일 쌍이 있는지 확인하세요

### 성능 최적화
- GPU 사용 시 더 빠른 처리 속도
- 설정 파일의 `max_keypoints` 값을 조정하여 속도와 정확도 균형 조절
- `debug_mode: false`로 설정하면 파일 저장 없이 처리 시간이 단축됩니다
- 카메라 거리 계산이 매칭 포인트 간 거리 기반으로 변경되어 더 안정적입니다

## 주요 변경사항

### 최신 업데이트
- **설정 파일 기반 구동**: CLI 인자 대신 YAML 설정 파일 사용
- **템플릿 파라미터 분리**: 소스 이미지 및 3D 포인트 좌표를 별도 파일로 관리
- **카메라 거리 계산 개선**: 포인트 클라우드 크기 대신 매칭 포인트 간 거리 기반 계산 (노이즈에 강건)
- **유틸리티 함수 추가**:
  - `clip_pointcloud_by_depth()`: Depth 범위로 포인트 클라우드 클리핑
  - `create_transform_matrix_from_vectors()`: 벡터로부터 변환 행렬 생성
