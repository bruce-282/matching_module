"""RoMa (romatch) 및 RoMa V2 (romav2, matching-module 번들) feature matcher."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ...utils import MODEL_REPO_ID
from ...utils.image_utils import load_image
from ...utils.viz_utils import visualize_matches
from .base_model import BaseModel

# -----------------------------------------------------------------------------
# RoMa / RoMa V2 공통
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ROMAV2_TYPES = frozenset({"roma_v2", "romav2"})


def _resolve_roma_weight_file(filename: str) -> Optional[Path]:
    """휠에 포함된 romatch/weights 우선, 이전 레이아웃(third_party/RoMa/weights) 호환."""
    import romatch

    bundled = Path(romatch.__file__).resolve().parent / "weights" / filename
    if bundled.is_file():
        return bundled
    repo_root = Path(__file__).resolve().parents[3]
    legacy = repo_root / "third_party" / "RoMa" / "weights" / filename
    if legacy.is_file():
        return legacy
    return None


# -----------------------------------------------------------------------------
# RoMa (romatch)
# -----------------------------------------------------------------------------


from romatch.models.model_zoo import roma_model  # noqa: E402  # after path / deps


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

    def _init(self, conf):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        local_model_path = _resolve_roma_weight_file(self.conf["model_name"])
        local_dinov2_path = _resolve_roma_weight_file(self.conf["model_utils_name"])

        if local_model_path is not None:
            logger.info("Local model file used: %s", local_model_path)
            model_path = str(local_model_path)
        else:
            model_path = self._download_model(
                repo_id=MODEL_REPO_ID,
                filename="{}/{}".format(Path(__file__).stem, self.conf["model_name"]),
            )

        if local_dinov2_path is not None:
            logger.info("Local DINOv2 file used: %s", local_dinov2_path)
            dinov2_weights = str(local_dinov2_path)
        else:
            dinov2_weights = self._download_model(
                repo_id=MODEL_REPO_ID,
                filename="{}/{}".format(
                    Path(__file__).stem, self.conf["model_utils_name"]
                ),
            )
        logger.debug("Loading Roma model")
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

        warp, certainty = self.net.match(img0, img1, device=device)
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


# -----------------------------------------------------------------------------
# RoMa V2 (romav2)
# -----------------------------------------------------------------------------


from romav2 import RoMaV2 as _RoMaV2Net  # noqa: E402
from romav2.device import device as romav2_device  # noqa: E402


class RomaV2(BaseModel):
    """RoMaV2 — BaseModel 래퍼 (romav2 패키지)."""

    default_conf = {
        "max_keypoints": 5000,
        "setting": "turbo",
        "compile": True,
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    def _init(self, conf):
        logging.getLogger(__name__).debug("Loading RoMaV2 model...")
        torch.set_float32_matmul_precision("highest")
        cfg = _RoMaV2Net.Cfg(
            setting=conf["setting"],
            compile=conf["compile"],
        )
        self.net = _RoMaV2Net(cfg=cfg)
        self.net.to(device)
        self.net.eval()
        logging.getLogger(__name__).debug("Load RoMaV2 model done.")

    def _forward(self, data):
        img0 = data["image0"]
        img1 = data["image1"]

        H_A, W_A = img0.shape[-2], img0.shape[-1]
        H_B, W_B = img1.shape[-2], img1.shape[-1]

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

        preds = self.net.match(img0_input, img1_input)

        matches, overlaps, _precision_ab, _precision_ba = self.net.sample(
            preds, self.conf["max_keypoints"]
        )

        H_out = self.net.H_hr if self.net.H_hr is not None else self.net.H_lr
        W_out = self.net.W_hr if self.net.W_hr is not None else self.net.W_lr

        kpts0, kpts1 = self.net.to_pixel_coordinates(
            matches, H_out, W_out, H_out, W_out
        )

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

        return {
            "keypoints0": kpts0_scaled,
            "keypoints1": kpts1_scaled,
            "mconf": overlaps,
        }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_demo_args():
    p = argparse.ArgumentParser(
        description="RoMa (romatch) / RoMa V2 (romav2) 이미지 매칭 데모"
    )
    p.add_argument(
        "--roma-v2",
        action="store_true",
        help="RoMaV2(romav2)로 실행 (기본은 RoMa/romatch)",
    )
    p.add_argument("--image0", type=str, default="")
    p.add_argument("--image1", type=str, default="")
    p.add_argument("--output", type=str, default="")
    p.add_argument("--max_keypoints", type=int, default=-1)
    p.add_argument("--confidence_threshold", type=float, default=0.5)
    p.add_argument(
        "--setting",
        type=str,
        default="fast",
        choices=["precise", "fast", "turbo", "base"],
        help="--roma-v2 일 때만 사용",
    )
    return p.parse_args()


def main():
    """콘솔 스크립트 `roma-match` 및 `python -m core.matchers.models.roma` 공통 진입점."""
    args = _parse_demo_args()

    if args.roma_v2:
        image0 = args.image0 or "third_party/RoMaV2/assets/toronto_A.jpg"
        image1 = args.image1 or "third_party/RoMaV2/assets/toronto_B.jpg"
        output = args.output or "roma_v2_matches.png"
        max_k = args.max_keypoints if args.max_keypoints > 0 else 3000

        print("image0:", image0)
        print("image1:", image1)
        print("device:", device)

        try:
            image0_t = load_image(image0)
            image1_t = load_image(image1)
            print("RoMaV2 model initialization in progress...")
            conf = RomaV2.default_conf.copy()
            conf["max_keypoints"] = max_k
            conf["setting"] = args.setting
            model = RomaV2(conf)
            print("Image matching in progress...")
            result = model(
                {"image0": image0_t.unsqueeze(0), "image1": image1_t.unsqueeze(0)}
            )
            keypoints0 = result["keypoints0"]
            keypoints1 = result["keypoints1"]
            confidence = result["mconf"]
            print("Matching completed!")
            print("Total matches:", len(keypoints0))
            print("Average confidence:", torch.mean(confidence).item())
            high_conf_mask = confidence > args.confidence_threshold
            h0 = keypoints0[high_conf_mask]
            h1 = keypoints1[high_conf_mask]
            hs = confidence[high_conf_mask]
            img0_np = (image0_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img1_np = (image1_t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            visualize_matches(
                img0_np,
                img1_np,
                h0.cpu().numpy() if torch.is_tensor(h0) else h0,
                h1.cpu().numpy() if torch.is_tensor(h1) else h1,
                hs.cpu().numpy() if torch.is_tensor(hs) else hs,
                output,
            )
            print("Test completed successfully!")
        except Exception as e:
            print("Error occurred:", e)
            import traceback

            traceback.print_exc()
        return

    # RoMa (romatch) demo
    image0 = args.image0 or "datasets/source.png"
    image1 = args.image1 or "datasets/target.png"
    output = args.output or "roma_matches.png"
    max_k = args.max_keypoints if args.max_keypoints > 0 else 1000

    print("image0:", image0)
    print("image1:", image1)
    print("device:", device)

    try:
        image0_t = load_image(image0)
        image1_t = load_image(image1)
        print("Roma model initialization in progress...")
        conf = Roma.default_conf.copy()
        conf["max_keypoints"] = max_k
        roma = Roma(conf)
        print("Image matching in progress...")
        result = roma(
            {"image0": image0_t.unsqueeze(0), "image1": image1_t.unsqueeze(0)}
        )
        keypoints0 = result["keypoints0"]
        keypoints1 = result["keypoints1"]
        confidence = result["mconf"]
        print("Matching completed!")
        print("Total matches:", len(keypoints0))
        print("Average confidence:", torch.mean(confidence).item())
        high_conf_mask = confidence > args.confidence_threshold
        high_conf_kpts0 = keypoints0[high_conf_mask]
        high_conf_kpts1 = keypoints1[high_conf_mask]
        high_conf_scores = confidence[high_conf_mask]
        print(
            "Confidence %s or higher matches: %s"
            % (args.confidence_threshold, len(high_conf_kpts0))
        )
        visualize_matches(
            image0,
            image1,
            high_conf_kpts0,
            high_conf_kpts1,
            high_conf_scores,
            output,
        )
        print("Test completed successfully!")
    except Exception as e:
        print("Error occurred:", e)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # 직접 실행 시 레포 루트를 path에 넣어 core.* / romatch import 가능하게
    _root = Path(__file__).resolve().parent.parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    main()
