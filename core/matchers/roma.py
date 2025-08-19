import sys
from pathlib import Path

import torch
from PIL import Image
import argparse
import numpy as np
import cv2

import logging

from ..utils import MODEL_REPO_ID
from ..utils.base_model import BaseModel
from ..utils.image_utils import load_image
from ..utils.viz_utils import visualize_matches

roma_path = Path(__file__).parent.parent.parent / "third_party/RoMa"
sys.path.append(str(roma_path))
from romatch.models.model_zoo import roma_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Roma(BaseModel):
    default_conf = {
        "name": "two_view_pipeline",
        "model_name": "roma_outdoor.pth",
        "model_utils_name": "dinov2_vitl14_pretrain.pth",
        "max_keypoints": 3000,
        "coarse_res": (560, 560),
        "upsample_res": (864, 1152),
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    # Initialize the line matcher
    def _init(self, conf):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        print(self.conf)
        model_path = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, self.conf["model_name"]),
        )

        dinov2_weights = self._download_model(
            repo_id=MODEL_REPO_ID,
            filename="{}/{}".format(Path(__file__).stem, self.conf["model_utils_name"]),
        )
        logger.info("Loading Roma model")
        # load the model
        weights = torch.load(model_path, map_location="cpu")
        dinov2_weights = torch.load(dinov2_weights, map_location="cpu")

        if str(device) == "cpu":
            amp_dtype = torch.float32
        else:
            amp_dtype = torch.float16
        self.net = roma_model(
            resolution=self.conf["coarse_res"],
            upsample_preds=True,
            weights=weights,
            dinov2_weights=dinov2_weights,
            device=device,
            amp_dtype=amp_dtype,
        )
        self.net.upsample_res = self.conf["upsample_res"]
        logger.info("Load Roma model done.")

    def _forward(self, data):
        img0 = data["image0"].cpu().numpy().squeeze() * 255
        img1 = data["image1"].cpu().numpy().squeeze() * 255
        img0 = img0.transpose(1, 2, 0)
        img1 = img1.transpose(1, 2, 0)
        img0 = Image.fromarray(img0.astype("uint8"))
        img1 = Image.fromarray(img1.astype("uint8"))
        W_A, H_A = img0.size
        W_B, H_B = img1.size

        # Match
        warp, certainty = self.net.match(img0, img1, device=device)
        # Sample matches for estimation
        matches, certainty = self.net.sample(
            warp, certainty, num=self.conf["max_keypoints"]
        )
        kpts1, kpts2 = self.net.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
        pred = {
            "keypoints0": kpts1,
            "keypoints1": kpts2,
            "mconf": certainty,
        }

        return pred


def main():
    """Roma 모델을 테스트하는 main 함수"""
    parser = argparse.ArgumentParser(description="Roma 모델을 사용한 이미지 매칭")
    parser.add_argument(
        "--image0",
        type=str,
        default="datasets/source.png",
        help="첫 번째 이미지 경로",
    )
    parser.add_argument(
        "--image1",
        type=str,
        default="datasets/target.png",
        help="두 번째 이미지 경로",
    )
    parser.add_argument(
        "--output", type=str, default="roma_matches.png", help="결과 이미지 저장 경로"
    )
    parser.add_argument(
        "--max_keypoints", type=int, default=1000, help="최대 키포인트 수"
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.5, help="신뢰도 임계값"
    )

    args = parser.parse_args()

    print("Roma 모델 테스트를 시작합니다...")
    print(f"이미지0: {args.image0}")
    print(f"이미지1: {args.image1}")
    print(f"디바이스: {device}")

    try:
        # 이미지 로드
        print("이미지를 로드하는 중...")
        image0 = load_image(args.image0)
        image1 = load_image(args.image1)

        # Roma 모델 초기화
        print("Roma 모델을 초기화하는 중...")
        conf = Roma.default_conf.copy()
        conf["max_keypoints"] = args.max_keypoints
        roma_model = Roma(conf)

        # 매칭 실행
        print("이미지 매칭을 실행하는 중...")
        data = {
            "image0": image0.unsqueeze(0),  # 배치 차원 추가
            "image1": image1.unsqueeze(0),
        }

        result = roma_model(data)

        # 결과 출력
        keypoints0 = result["keypoints0"]
        keypoints1 = result["keypoints1"]
        confidence = result["mconf"]

        print(f"매칭 완료!")
        print(f"총 매칭 수: {len(keypoints0)}")
        print(f"평균 신뢰도: {torch.mean(confidence).item():.3f}")
        print(f"최고 신뢰도: {torch.max(confidence).item():.3f}")
        print(f"최저 신뢰도: {torch.min(confidence).item():.3f}")

        # 신뢰도 임계값 이상의 매칭만 필터링
        high_conf_mask = confidence > args.confidence_threshold
        high_conf_kpts0 = keypoints0[high_conf_mask]
        high_conf_kpts1 = keypoints1[high_conf_mask]
        high_conf_scores = confidence[high_conf_mask]

        print(f"신뢰도 {args.confidence_threshold} 이상의 매칭: {len(high_conf_kpts0)}")

        # 결과 시각화
        print("결과를 시각화하는 중...")
        visualize_matches(
            args.image0,
            args.image1,
            high_conf_kpts0,
            high_conf_kpts1,
            high_conf_scores,
            args.output,
        )

        print("테스트가 성공적으로 완료되었습니다!")

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
