import sys
from pathlib import Path

# 스크립트 직접 실행 시 (python core/matchers/models/roma_v2.py) 프로젝트 루트를 path에 추가
if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(_root))

import torch
import numpy as np
import argparse
import logging

from core.matchers.models.base_model import BaseModel
from core.utils.image_utils import load_image
from core.utils.viz_utils import visualize_matches

from romav2 import RoMaV2
from romav2.device import device as romav2_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RomaV2(BaseModel):
    """RoMaV2 모델 - BaseModel 인터페이스 기반 래퍼"""

    default_conf = {
        "max_keypoints": 5000,
        "setting": "turbo",  # precise, fast, turbo, base, mega1500, scannet1500, wxbs, satast
        "compile": True,
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    def _init(self, conf):
        logging.getLogger(__name__).debug("Loading RoMaV2 model...")

        # RoMaV2는 float32 matmul precision이 highest여야 함
        torch.set_float32_matmul_precision("highest")

        cfg = RoMaV2.Cfg(
            setting=conf["setting"],
            compile=conf["compile"],
        )
        self.net = RoMaV2(cfg=cfg)
        self.net.to(device)
        self.net.eval()

        logging.getLogger(__name__).debug("Load RoMaV2 model done.")

    def _forward(self, data):
        img0 = data["image0"]
        img1 = data["image1"]

        # 입력 이미지 크기 (H, W) - Matcher가 원본 좌표계로 스케일링함
        H_A, W_A = img0.shape[-2], img0.shape[-1]
        H_B, W_B = img1.shape[-2], img1.shape[-1]

        # RoMaV2.match()는 경로, PIL, numpy, tensor 모두 받음
        # tensor: (B, C, H, W) 또는 (C, H, W), [0, 1] 범위
        # _load_image는 (B,C,H,W) 또는 (C,H,W) 기대 - (C,H,W)면 batch 차원 추가
        if img0.dim() == 3:
            img0_input = img0.unsqueeze(0)
        else:
            img0_input = img0
        if img1.dim() == 3:
            img1_input = img1.unsqueeze(0)
        else:
            img1_input = img1
        img0_input = img0_input.to(romav2_device)
        img1_input = img1_input.to(romav2_device)

        # Match densely
        preds = self.net.match(img0_input, img1_input)

        # Sample matches for estimation
        matches, overlaps, precision_AB, precision_BA = self.net.sample(
            preds, self.conf["max_keypoints"]
        )

        # RoMaV2 출력 해상도 (warp 해상도)
        H_out = self.net.H_hr if self.net.H_hr is not None else self.net.H_lr
        W_out = self.net.W_hr if self.net.W_hr is not None else self.net.W_lr

        # Convert to pixel coordinates (RoMaV2 produces matches in [-1,1]x[-1,1])
        kpts0, kpts1 = self.net.to_pixel_coordinates(
            matches, H_out, W_out, H_out, W_out
        )

        # RoMaV2 내부 리사이즈 해상도 → 입력 이미지 해상도로 스케일
        scale_x_A = W_A / W_out
        scale_y_A = H_A / H_out
        scale_x_B = W_B / W_out
        scale_y_B = H_B / H_out

        kpts0_scaled = kpts0.clone()
        kpts0_scaled[..., 0] *= scale_x_A
        kpts0_scaled[..., 1] *= scale_y_A

        kpts1_scaled = kpts1.clone()
        kpts1_scaled[..., 0] *= scale_x_B
        kpts1_scaled[..., 1] *= scale_y_B

        pred = {
            "keypoints0": kpts0_scaled,
            "keypoints1": kpts1_scaled,
            "mconf": overlaps,
        }

        return pred


def main():
    """RoMaV2 모델을 테스트하는 main 함수"""
    parser = argparse.ArgumentParser(description="RoMaV2 모델을 사용한 이미지 매칭")
    parser.add_argument(
        "--image0",
        type=str,
        default="core/matchers/models/RoMaV2/assets/toronto_A.jpg",
        help="첫 번째 이미지 경로",
    )
    parser.add_argument(
        "--image1",
        type=str,
        default="core/matchers/models/RoMaV2/assets/toronto_B.jpg",
        help="두 번째 이미지 경로",
    )
    parser.add_argument(
        "--output", type=str, default="roma_v2_matches.png", help="결과 이미지 저장 경로"
    )
    parser.add_argument(
        "--max_keypoints", type=int, default=3000, help="최대 키포인트 수"
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.5, help="신뢰도 임계값"
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="fast",
        choices=["precise", "fast", "turbo", "base"],
        help="RoMaV2 설정 (precise, fast, turbo, base)",
    )

    args = parser.parse_args()

    print(f"image0: {args.image0}")
    print(f"image1: {args.image1}")
    print(f"device: {device}")

    try:
        # Load images
        image0 = load_image(args.image0)
        image1 = load_image(args.image1)

        # RoMaV2 model initialization
        print("RoMaV2 model initialization in progress...")
        conf = RomaV2.default_conf.copy()
        conf["max_keypoints"] = args.max_keypoints
        conf["setting"] = args.setting
        roma_v2_model = RomaV2(conf)

        # Matching execution
        print("Image matching in progress...")
        data = {
            "image0": image0.unsqueeze(0),
            "image1": image1.unsqueeze(0),
        }

        result = roma_v2_model(data)

        # Result output
        keypoints0 = result["keypoints0"]
        keypoints1 = result["keypoints1"]
        confidence = result["mconf"]

        print(f"Matching completed!")
        print(f"Total matches: {len(keypoints0)}")
        print(f"Average confidence: {torch.mean(confidence).item():.3f}")
        print(f"Max confidence: {torch.max(confidence).item():.3f}")
        print(f"Min confidence: {torch.min(confidence).item():.3f}")

        # 신뢰도 임계값 이상의 매칭만 필터링
        high_conf_mask = confidence > args.confidence_threshold
        high_conf_kpts0 = keypoints0[high_conf_mask]
        high_conf_kpts1 = keypoints1[high_conf_mask]
        high_conf_scores = confidence[high_conf_mask]

        print(
            f"Confidence {args.confidence_threshold} or higher matches: {len(high_conf_kpts0)}"
        )

        # 결과 시각화 (visualize_matches는 이미지 배열을 받음)
        print("Visualizing results...")
        img0_np = (image0.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img1_np = (image1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        visualize_matches(
            img0_np,
            img1_np,
            high_conf_kpts0.cpu().numpy() if torch.is_tensor(high_conf_kpts0) else high_conf_kpts0,
            high_conf_kpts1.cpu().numpy() if torch.is_tensor(high_conf_kpts1) else high_conf_kpts1,
            high_conf_scores.cpu().numpy() if torch.is_tensor(high_conf_scores) else high_conf_scores,
            args.output,
        )

        print("Test completed successfully!")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
