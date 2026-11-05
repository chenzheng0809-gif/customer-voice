"""OCR 支线：客户截图 → 文本 → 接入同一条分析管线。

业务场景：很多客户不是打字，而是截一张聊天记录/订单页的图发过来。
这条支线的价值是"入口多一种"，而不是另起一套分析逻辑 ——
OCR 之后走的是和文本完全相同的模型，这点在设计上很重要。

用法：
  python src/ocr_pipeline.py --image data/ocr_samples/xxx.png
  python src/ocr_pipeline.py --demo          # 自动生成一张合成聊天截图再识别
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@dataclass
class OcrLine:
    text: str
    box: list            # 四点坐标 [[x,y],...]
    score: float


@dataclass
class OcrResult:
    image: str
    lines: list[OcrLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    @property
    def mean_score(self) -> float:
        return sum(l.score for l in self.lines) / len(self.lines) if self.lines else 0.0


_ENGINE = None


def _engine():
    """懒加载：模型加载有几百毫秒开销，只做一次。"""
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        _ENGINE = RapidOCR()
    return _ENGINE


def run_ocr(image_path: str | Path) -> OcrResult:
    eng = _engine()
    raw, _ = eng(str(image_path))
    res = OcrResult(image=str(image_path))
    for item in raw or []:
        # RapidOCR 返回 [box, text, score]
        box, text, score = item[0], item[1], float(item[2])
        res.lines.append(OcrLine(text=text.strip(), box=box, score=score))
    return res


# ------------------------------------------------------------------ 结构化
# 聊天截图里通常有"昵称：内容"的结构，或者按行从上到下就是消息顺序。
# 这里做一层轻量结构化：识别说话人、去噪。
_SPEAKER = re.compile(r"^([\u4e00-\u9fffA-Za-z0-9_]{1,12})\s*[:：]\s*(.+)$")
_NOISE = re.compile(r"^(发送|取消|输入中|正在输入|\d{1,2}:\d{2}(:\d{2})?|今天|昨天)$")

# 聊天截图顶部/底部的非消息文本。不排除的话会被当成客户发言分析，
# 比如标题"客户咨询"会被判成负面情绪，污染统计。
_UI_TEXT = re.compile(
    r"^(客户咨询|在线客服|客服中心|聊天记录|消息|群聊|"
    r"\d{1,2}:\d{2}(:\d{2})?|上午|下午|昨天|今天|星期[一二三四五六日])"
)

# 字数太少的行通常也不是有效消息（如 "1"、"+"、发送按钮）
_MIN_LEN = 4


def structure(res: OcrResult) -> list[dict]:
    """把 OCR 行整理成 [{speaker, text}]，过滤时间戳等无信息行。"""
    out: list[dict] = []
    for line in res.lines:
        t = line.text.strip()
        if not t or _NOISE.match(t) or _UI_TEXT.match(t):
            continue
        m = _SPEAKER.match(t)
        if m:
            speaker, body = m.group(1), m.group(2).strip()
        else:
            speaker, body = "unknown", t
        # 去掉 UI 文字后再判断长度，避免把空格或单个符号当消息
        if len(body) < _MIN_LEN:
            continue
        out.append({"speaker": speaker, "text": body})
    return out


def analyze_image(image_path: str | Path) -> dict:
    """OCR + 结构化 + 意图/情感分析，一条龙。

    这里复用 server 里的模型加载逻辑（同一个 baseline 模型），
    保证图片和文本两条路径的判定标准一致。
    """
    from server import api_analyze

    res = run_ocr(image_path)
    msgs = structure(res)
    results = []
    for m in msgs:
        try:
            pred = api_analyze(m["text"])
            results.append({**m, "intent": pred["intent"], "sentiment": pred["sentiment"]})
        except Exception as e:
            results.append({**m, "error": str(e)})

    return {
        "image": str(image_path),
        "ocr_mean_score": round(res.mean_score, 4),
        "n_lines": len(res.lines),
        "messages": results,
        "raw_text": res.text,
    }


# ------------------------------------------------------------------ 演示图
def make_demo_image(path: Path) -> Path:
    """用 PIL 画一张仿聊天截图，供没有真实素材时演示。"""
    from PIL import Image, ImageDraw, ImageFont

    W, H = 640, 420
    img = Image.new("RGB", (W, H), "#f2f3f5")
    d = ImageDraw.Draw(img)

    # 按候选顺序找第一个可用的中文字体。
    # 注意：PingFang.ttc 在部分 macOS 上不存在，字体加载失败会静默画成方框，
    # OCR 再把方框识别成 "000000"，所以一定要探测而不是写死路径。
    font_candidates = [
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    font = small = None
    for fp in font_candidates:
        try:
            font = ImageFont.truetype(fp, 20)
            small = ImageFont.truetype(fp, 14)
            break
        except OSError:
            continue
    if font is None:
        raise RuntimeError(
            "找不到可用的中文字体，无法生成演示图；"
            "请把真实截图放到 data/ocr_samples/ 下用 --image 指定"
        )

    d.text((20, 16), "客户咨询", font=small, fill="#666")

    def bubble(x, y, text, color, right=False):
        w = int(d.textlength(text, font=font)) + 28
        x0 = W - w - 20 if right else 20
        d.rounded_rectangle([x0, y, x0 + w, y + 40], radius=10, fill=color)
        d.text((x0 + 14, y + 9), text, font=font, fill="#1a1a1a")
        return y + 52

    y = 50
    y = bubble(20, y, "客服：您好，请问有什么可以帮您", "#ffffff")
    y = bubble(20, y, "我想问下这个什么时候发货", "#95ec69", right=True)
    y = bubble(20, y, "都等了三天了，还没动静", "#95ec69", right=True)
    y = bubble(20, y, "这是你们说的次日达吗？", "#95ec69", right=True)
    d.text((20, y + 4), "14:32", font=small, fill="#999")

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    if a.demo or not a.image:
        import json

        p = make_demo_image(ROOT / "data" / "ocr_samples" / "demo_chat.png")
        print(f"已生成演示图: {p}")
        out = analyze_image(p)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        import json

        print(json.dumps(analyze_image(a.image), ensure_ascii=False, indent=2))