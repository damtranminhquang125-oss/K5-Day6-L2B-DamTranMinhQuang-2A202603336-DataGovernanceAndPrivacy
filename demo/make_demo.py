"""Ảnh chụp demo: ghép top-B ảnh cần review (score, loại lỗi, KEEP/RECAPTURE) -> demo/top20_demo.png.

  python -m src.pipeline all            # hoặc: python -m src.main --track t02 --output out/poc --run-name poc
  python demo/make_demo.py [--run poc] [--cols 5]

Chỉ dùng Pillow; chú thích ASCII vì font mặc định của Pillow không có dấu tiếng Việt. Không đọc gt/ và không dùng nhãn.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "fixtures" / "t02_raw_qc"
TYPE_COLOR = {"blur": (60, 110, 168), "dark": (75, 79, 88), "glare": (184, 134, 11), "compression": (122, 74, 160), "none": (138, 144, 153)}
ACTION_COLOR = {"RECAPTURE": (168, 50, 31), "KEEP": (47, 111, 79)}


def build(run: str, cols: int, out: Path) -> Path:
    q = pd.read_csv(ROOT / "out" / run / "review_queue.csv")
    r = pd.read_csv(ROOT / "out" / run / "ranked_review.csv").set_index("image_id")
    tw, th, cap, head, pad = 256, 144, 34, 30, 6
    rows = -(-len(q) // cols)
    canvas = Image.new("RGB", (cols * (tw + pad) + pad, head + rows * (th + cap + pad) + pad), (233, 234, 236))
    d = ImageDraw.Draw(canvas); font = ImageFont.load_default()
    n_rec = int((q.suggested_action == "RECAPTURE").sum())
    d.text((pad, 9), f"T02 Raw QC Triage | run={run} | top-{len(q)} review queue | RECAPTURE {n_rec}/{len(q)} | proxy labels NOT used", fill=(26, 30, 35), font=font)
    for i, row in q.iterrows():
        x, y = pad + (i % cols) * (tw + pad), head + (i // cols) * (th + cap + pad)
        im = Image.open(FX / row.evidence_path).convert("RGB").resize((tw, th))
        canvas.paste(im, (x, y))
        typ = r.loc[row.item_id, "pred_defect_type"]; act = r.loc[row.item_id, "pred_action"]
        d.rectangle([x, y + th, x + tw, y + th + cap], fill=(247, 247, 248))
        d.rectangle([x, y, x + 24, y + 14], fill=(26, 30, 35)); d.text((x + 4, y + 2), f"#{row['rank']}", fill="white", font=font)
        d.text((x + 3, y + th + 3), f"{row.item_id}  score {row.score:.2f}", fill=(26, 30, 35), font=font)
        d.rectangle([x + 3, y + th + 18, x + 3 + 70, y + th + 30], fill=TYPE_COLOR.get(typ, TYPE_COLOR["none"]))
        d.text((x + 6, y + th + 19), typ, fill="white", font=font)
        d.rectangle([x + 80, y + th + 18, x + 80 + 70, y + th + 30], fill=ACTION_COLOR[act])
        d.text((x + 83, y + th + 19), act, fill="white", font=font)
        d.text((x + 156, y + th + 19), str(row.slice), fill=(70, 78, 90), font=font)
    out.parent.mkdir(parents=True, exist_ok=True); canvas.save(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="poc"); ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "demo" / "top20_demo.png")); a = ap.parse_args()
    print("wrote", build(a.run, a.cols, Path(a.out)))
