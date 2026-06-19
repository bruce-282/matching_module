import sys
import shutil
from pathlib import Path

import torch
from PIL import Image
import argparse
import numpy as np

import logging

from ...utils import MODEL_REPO_ID
from .base_model import BaseModel
from ...utils.image_utils import load_image
from ...utils.viz_utils import visualize_matches

roma_path = Path(__file__).parent.parent.parent.parent / "third_party/RoMa"
sys.path.append(str(roma_path))
from romatch.models.model_zoo import roma_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Roma(BaseModel):
    default_conf = {
        "name": "two_view_pipeline",
        "model_name": "roma_outdoor.pth",
        "model_utils_name": "dinov2_vitl14_pretrain.pth",
        "max_keypoints": 2000,
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

        # 가중치 영구 경로(레포 내). HF 캐시(~/.cache/huggingface)는 지워질 수 있어,
        # 다운로드한 파일은 여기로 복사해 보존한다 → 다음 실행부터 캐시와 무관하게 사용.
        weights_dir = (
            Path(__file__).parent.parent.parent / "third_party/RoMa/weights"
        )
        local_model_path = weights_dir / self.conf["model_name"]
        local_dinov2_path = weights_dir / self.conf["model_utils_name"]

        def _resolve_weight(local_path: Path, filename: str) -> str:
            """로컬 영구 파일이 있으면 그걸 쓰고, 없으면 HF 에서 받아 로컬로 복사해 보존."""
            if local_path.exists():
                logger.info(f"Local weight used: {local_path}")
                return str(local_path)
            logger.info(f"Local weight not found, downloading from HF: {filename}")
            downloaded = self._download_model(
                repo_id=MODEL_REPO_ID, filename=filename
            )
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(downloaded, local_path)
                logger.info(
                    f"Weight persisted to repo (cache-independent): {local_path}"
                )
                return str(local_path)
            except OSError as e:
                # 복사 실패 시에도 동작하도록 캐시 경로를 그대로 사용(이번 실행 한정).
                logger.warning(f"Failed to persist weight to {local_path}: {e}")
                return downloaded

        model_path = _resolve_weight(
            local_model_path,
            "{}/{}".format(Path(__file__).stem, self.conf["model_name"]),
        )
        dinov2_weights = _resolve_weight(
            local_dinov2_path,
            "{}/{}".format(Path(__file__).stem, self.conf["model_utils_name"]),
        )
        logger.debug("Loading Roma model")
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
        logger.debug("Load Roma model done.")

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

    print(f"image0: {args.image0}")
    print(f"image1: {args.image1}")
    print(f"device: {device}")

    try:
        # Load images
        image0 = load_image(args.image0)
        image1 = load_image(args.image1)

        # Roma model initialization
        print("Roma model initialization in progress...")
        conf = Roma.default_conf.copy()
        conf["max_keypoints"] = args.max_keypoints
        roma_model = Roma(conf)

        # Matching execution
        print("Image matching in progress...")
        data = {
            "image0": image0.unsqueeze(0),  # Add batch dimension
            "image1": image1.unsqueeze(0),
        }

        result = roma_model(data)

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

        print(f"Confidence {args.confidence_threshold} or higher matches: {len(high_conf_kpts0)}")

        # 결과 시각화
        print("Visualizing results...")
        visualize_matches(
            args.image0,
            args.image1,
            high_conf_kpts0,
            high_conf_kpts1,
            high_conf_scores,
            args.output,
        )

        print("Test completed successfully!")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
