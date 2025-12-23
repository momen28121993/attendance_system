"""
SilentFace Anti-Spoofing integration.
Loads the open-source MiniFASNetV2 model and provides a simple liveness score.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import urllib.request
import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Lightweight MiniFASNet implementation (trimmed from the original SilentFace repo)
# -------------------------------------------------------------------------------


class L2Norm(nn.Module):
    def forward(self, input):
        return F.normalize(input)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DepthWise(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        c3,
        residual=False,
        kernel=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        groups=1,
    ):
        super().__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = ConvBlock(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = ConvBlock(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = LinearBlock(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(nn.Module):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            modules.append(
                DepthWise(
                    c1_tuple, c2_tuple, c3_tuple, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups
                )
            )
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.sigmoid(x)
        return module_input * x


class DepthWiseSE(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        c3,
        residual=False,
        kernel=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        groups=1,
        se_reduct=8,
    ):
        super().__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = ConvBlock(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = ConvBlock(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = LinearBlock(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
        self.se_module = SEModule(c3_out, se_reduct)

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            x = self.se_module(x)
            output = short_cut + x
        else:
            output = x
        return output


class ResidualSE(nn.Module):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se_reduct=4):
        super().__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            if i == num_block - 1:
                modules.append(
                    DepthWiseSE(
                        c1_tuple,
                        c2_tuple,
                        c3_tuple,
                        residual=True,
                        kernel=kernel,
                        padding=padding,
                        stride=stride,
                        groups=groups,
                        se_reduct=se_reduct,
                    )
                )
            else:
                modules.append(
                    DepthWise(
                        c1_tuple,
                        c2_tuple,
                        c3_tuple,
                        residual=True,
                        kernel=kernel,
                        padding=padding,
                        stride=stride,
                        groups=groups,
                    )
                )
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MiniFASNet(nn.Module):
    def __init__(self, keep, embedding_size, conv6_kernel=(7, 7), drop_p=0.0, num_classes=3, img_channel=3):
        super().__init__()
        self.embedding_size = embedding_size

        self.conv1 = ConvBlock(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(keep[0], keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=keep[1])

        c1 = [(keep[1], keep[2])]
        c2 = [(keep[2], keep[3])]
        c3 = [(keep[3], keep[4])]

        self.conv_23 = DepthWise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[3])

        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]

        self.conv_3 = Residual(c1, c2, c3, num_block=4, groups=keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[16], keep[17])]
        c2 = [(keep[17], keep[18])]
        c3 = [(keep[18], keep[19])]

        self.conv_34 = DepthWise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[19])

        c1 = [
            (keep[19], keep[20]),
            (keep[22], keep[23]),
            (keep[25], keep[26]),
            (keep[28], keep[29]),
            (keep[31], keep[32]),
            (keep[34], keep[35]),
        ]
        c2 = [
            (keep[20], keep[21]),
            (keep[23], keep[24]),
            (keep[26], keep[27]),
            (keep[29], keep[30]),
            (keep[32], keep[33]),
            (keep[35], keep[36]),
        ]
        c3 = [
            (keep[21], keep[22]),
            (keep[24], keep[25]),
            (keep[27], keep[28]),
            (keep[30], keep[31]),
            (keep[33], keep[34]),
            (keep[36], keep[37]),
        ]

        self.conv_4 = Residual(c1, c2, c3, num_block=6, groups=keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[37], keep[38])]
        c2 = [(keep[38], keep[39])]
        c3 = [(keep[39], keep[40])]

        self.conv_45 = DepthWise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[40])

        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]

        self.conv_5 = Residual(c1, c2, c3, num_block=2, groups=keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = ConvBlock(keep[46], keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = LinearBlock(keep[47], keep[48], groups=keep[48], kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.drop = nn.Dropout(p=drop_p)
        self.prob = nn.Linear(embedding_size, num_classes, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        if self.embedding_size != 512:
            out = self.linear(out)
        out = self.bn(out)
        out = self.drop(out)
        out = self.prob(out)
        return out


class MiniFASNetSE(MiniFASNet):
    def __init__(self, keep, embedding_size, conv6_kernel=(7, 7), drop_p=0.75, num_classes=4, img_channel=3):
        super().__init__(
            keep=keep, embedding_size=embedding_size, conv6_kernel=conv6_kernel, drop_p=drop_p, num_classes=num_classes, img_channel=img_channel
        )

        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]

        self.conv_3 = ResidualSE(c1, c2, c3, num_block=4, groups=keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [
            (keep[19], keep[20]),
            (keep[22], keep[23]),
            (keep[25], keep[26]),
            (keep[28], keep[29]),
            (keep[31], keep[32]),
            (keep[34], keep[35]),
        ]
        c2 = [
            (keep[20], keep[21]),
            (keep[23], keep[24]),
            (keep[26], keep[27]),
            (keep[29], keep[30]),
            (keep[32], keep[33]),
            (keep[35], keep[36]),
        ]
        c3 = [
            (keep[21], keep[22]),
            (keep[24], keep[25]),
            (keep[27], keep[28]),
            (keep[30], keep[31]),
            (keep[33], keep[34]),
            (keep[36], keep[37]),
        ]

        self.conv_4 = ResidualSE(c1, c2, c3, num_block=6, groups=keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        self.conv_5 = ResidualSE(c1, c2, c3, num_block=2, groups=keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1))


keep_dict = {
    "1.8M": [
        32,
        32,
        103,
        103,
        64,
        13,
        13,
        64,
        26,
        26,
        64,
        13,
        13,
        64,
        52,
        52,
        64,
        231,
        231,
        128,
        154,
        154,
        128,
        52,
        52,
        128,
        26,
        26,
        128,
        52,
        52,
        128,
        26,
        26,
        128,
        26,
        26,
        128,
        308,
        308,
        128,
        26,
        26,
        128,
        26,
        26,
        128,
        512,
        512,
    ],
    "1.8M_": [
        32,
        32,
        103,
        103,
        64,
        13,
        13,
        64,
        13,
        13,
        64,
        13,
        13,
        64,
        13,
        13,
        64,
        231,
        231,
        128,
        231,
        231,
        128,
        52,
        52,
        128,
        26,
        26,
        128,
        77,
        77,
        128,
        26,
        26,
        128,
        26,
        26,
        128,
        308,
        308,
        128,
        26,
        26,
        128,
        26,
        26,
        128,
        512,
        512,
    ],
}


def MiniFASNetV2(embedding_size=128, conv6_kernel=(7, 7), drop_p=0.2, num_classes=3, img_channel=3):
    return MiniFASNet(keep_dict["1.8M_"], embedding_size, conv6_kernel, drop_p, num_classes, img_channel)


def get_kernel(height: int, width: int) -> Tuple[int, int]:
    return ((height + 15) // 16, (width + 15) // 16)


# Anti-spoofing wrapper
# -------------------------------------------------------------------------------

DEFAULT_MODEL_URL = (
    "https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/master/resources/anti_spoof_models/"
    "2.7_80x80_MiniFASNetV2.pth"
)


class SilentFaceAntiSpoof:
    """Minimal SilentFace wrapper. Returns a liveness score in [0,1]."""

    def __init__(
        self,
        model_dir: Path,
        threshold: float = 0.5,
        device: Optional[str] = None,
        download_url: str = DEFAULT_MODEL_URL,
    ):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "2.7_80x80_MiniFASNetV2.pth"
        self.threshold = threshold
        self.download_url = download_url
        self.device = torch.device(device) if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.input_size: Tuple[int, int] = (80, 80)
        self.scale: Optional[float] = 2.7  # match default 2.7_80x80 model crop
        self.conv6_kernel = get_kernel(*self.input_size)
        self._lock = threading.Lock()
        self._load_error_reported = False
        self.disabled = False

    def _ensure_model_file(self):
        if self.model_path.exists():
            return
        self.model_dir.mkdir(parents=True, exist_ok=True)
        print(f"⬇️  Downloading SilentFace model to {self.model_path} ...")
        urllib.request.urlretrieve(self.download_url, self.model_path)
        print("✓ Downloaded SilentFace model")

    def _load_model(self):
        if self.model is not None:
            return
        self._ensure_model_file()
        # Derive input size and scale from file name if possible (e.g. 2.7_80x80_MiniFASNetV2.pth)
        stem_parts = self.model_path.stem.split("_")
        if len(stem_parts) >= 2 and "x" in stem_parts[1]:
            try:
                h, w = map(int, stem_parts[1].split("x"))
                self.input_size = (w, h)
                self.conv6_kernel = get_kernel(h, w)
            except Exception:
                pass
            try:
                scale_candidate = float(stem_parts[0])
                self.scale = scale_candidate
            except Exception:
                pass
        state_dict = torch.load(self.model_path, map_location=self.device)
        if next(iter(state_dict)).startswith("module."):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model = MiniFASNetV2(conv6_kernel=self.conv6_kernel, num_classes=3)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    def _crop_with_scale(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Crop using the same scaling strategy as the reference implementation
        (expands the box by `scale` while staying inside image bounds).
        """
        if self.scale is None:
            # Fallback: simple resize of the detected face
            x, y, w, h = bbox
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            if x2 <= x1 or y2 <= y1:
                return None
            patch = frame[y1:y2, x1:x2]
            return cv2.resize(patch, self.input_size)

        x, y, w, h = bbox
        src_h, src_w = frame.shape[:2]
        # Mirror of generate_patches.CropImage._get_new_box
        scale = min((src_h - 1) / max(h, 1), min((src_w - 1) / max(w, 1), self.scale))
        new_w = w * scale
        new_h = h * scale
        cx, cy = w / 2 + x, h / 2 + y
        x1 = cx - new_w / 2
        y1 = cy - new_h / 2
        x2 = cx + new_w / 2
        y2 = cy + new_h / 2

        # Clamp to image boundaries while preserving size as much as possible
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > src_w - 1:
            x1 -= x2 - src_w + 1
            x2 = src_w - 1
        if y2 > src_h - 1:
            y1 -= y2 - src_h + 1
            y2 = src_h - 1

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        if x2 <= x1 or y2 <= y1:
            return None
        patch = frame[y1 : y2 + 1, x1 : x2 + 1]
        if patch.size == 0:
            return None
        return cv2.resize(patch, self.input_size)

    def predict(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Dict]:
        """
        Returns dict with liveness info or None if something went wrong.
        """
        if self.disabled:
            return None

        with self._lock:
            try:
                self._load_model()
            except Exception as exc:  # pylint: disable=broad-except
                if not self._load_error_reported:
                    print(f"⚠ SilentFace model load failed: {exc}")
                    self._load_error_reported = True
                # Permanently disable to avoid blocking recognition
                self.disabled = True
                return None

        x, y, w, h = bbox
        crop = self._crop_with_scale(frame, bbox)
        if crop is None or crop.size == 0:
            return None

        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, self.input_size)
        except Exception:
            return None

        # BGR to tensor in [0,1]
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        real_prob = float(probs[1]) if probs.shape[0] > 1 else float(probs[0])
        if probs.shape[0] > 1:
            other = np.delete(probs, 1)
            spoof_prob = float(np.max(other))
        else:
            spoof_prob = float(1.0 - real_prob)

        # Require a margin over spoof prob; fail closed only on confident spoof
        margin = real_prob - spoof_prob
        is_real = (real_prob >= self.threshold and margin >= -0.1) or margin >= 0.15 or real_prob >= 0.55

        return {
            "is_real": is_real,
            "real_prob": real_prob,
            "spoof_prob": spoof_prob,
            "probs": probs,
        }
