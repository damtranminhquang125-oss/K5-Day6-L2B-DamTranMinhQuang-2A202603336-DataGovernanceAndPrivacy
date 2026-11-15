"""T02 · Raw QC Triage — PIPELINE CỦA NHÓM (PoC).

Hợp đồng lab = 5 bước, chạy bởi `python -m src.main ...`:
    01 load      manifest + kiểm sha256 (strict_provenance) + bảng ảnh (image_id, path, split, camera_id, light)
    02 process   feature (Laplacian, độ sáng, ROI cháy sáng 4x4, blockiness 8px) -> điểm theo loại lỗi -> score
    03 rank      top-B (budget) trên pool ngoài split dev -> review_queue.csv + contact_sheet.html (ROI glare)
    04 review    nếu có out/<run>/review_queue_reviewed.csv: áp override + AuditLog -> final_actions.csv
    05 decide    GO / HOLD / REWORK + reason_code + owner (đọc reports/metrics_*.json nếu đã chấm)

Ba biến thể, chọn bằng `--run-name` (cùng pool / budget / seed 42):
    baseline  một ngưỡng Laplacian toàn ảnh (mốc so sánh bắt buộc của đề)
    poc       PoC v1: đa feature + chuẩn hoá theo light/camera + ROI  (file nộp chính)
    poc_v2    PoC v1 + feature "bloom" — POST-HOC (thêm sau khi thấy lỗi của v1 trên eval), không phải kết quả giữ-out sạch

Lệnh (chạy từ thư mục gốc repo):
    python -m src.pipeline fit                       # fit tham số/ngưỡng CHỈ trên split dev -> configs/t02_params.json
    python -m src.main --track t02 --output out/baseline --run-name baseline --offline --seed 42
    python -m src.main --track t02 --output out/poc      --run-name poc      --offline --seed 42
    python -m src.main --track t02 --output out/poc_v2   --run-name poc_v2   --offline --seed 42
    python -m src.evaluate --track t02 --gt <gt/labels.csv> --pred out/poc/ranked_review.csv --run-name poc --out reports/metrics_poc.json
    python -m src.pipeline ablation | failure-modes | decide
    python -m src.pipeline all [--gt ...]            # toàn bộ chuỗi trên (GT mặc định = nhãn proxy)

Quy tắc: không đọc fixtures/*/gt/ (GT guard của src.main); pipeline xác định, không có thành phần ngẫu nhiên
(seed 42 chỉ ghi nhận); không gọi network. GT mặc định ở bước chấm là NHÃN PROXY gán bằng mắt
(fixtures/t02_raw_qc/proxy_gt/labels.csv) — không phải gt/labels.csv chính thức.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label

from .common.contact_sheet import render_contact_sheet
from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.image_ops import brightness_stats, laplacian_variance, load_rgb, to_gray
from .common.io import load_config, read_json, write_json
from .common.manifest import verify_manifest
from .common.review import apply_overrides, load_review_decisions, make_review_queue, write_review_queue
from .metrics.tracks import get_track

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures" / "t02_raw_qc"
PARAMS_PATH = ROOT / "configs" / "t02_params.json"          # tham số đã freeze (kết quả của `fit`, chỉ từ dev)
LABELS_PATH = FIXTURE / "proxy_labels.csv"                   # nhãn proxy (tự gán bằng mắt) — fit đọc CHỈ các dòng dev
PROXY_GT_PATH = FIXTURE / "proxy_gt" / "labels.csv"          # cùng schema gt/labels.csv, dùng để chấm tự kiểm
OUT_DIR = ROOT / "out"
REPORTS = ROOT / "reports"

TYPES = ["blur", "dark", "glare", "compression"]
OUT_COLS = ["image_id", "score", "pred_defect_type", "pred_action"]
IMG_EXT = (".jpg", ".jpeg", ".png")
VALID_ACTIONS = {"KEEP", "RECAPTURE"}
ACTION_THRESHOLD = 1.0       # score >= 1 <=> vượt ngưỡng ít nhất một loại lỗi
BLOOM_MARGIN = 0.10          # biên tối thiểu (đặt tay) giữa max sạch và ví dụ bloom khi fit thr_bloom
PLACEHOLDER_OWNERS = ("tbd", "todo", "nhóm_điền", "nhom_dien")
GRID = (4, 4)                # lưới ROI n×m cho feature cục bộ
SAT_LEVEL = 240              # pixel >= mức này coi là cháy sáng
BLOOM_LEVEL = 235
_EIGHT = np.ones((3, 3), dtype=int)


# =============================================================================================== features
def tile_boxes(h: int, w: int, n: int = GRID[0], m: int = GRID[1]) -> list[tuple[int, int, int, int]]:
    """Danh sách (x, y, w, h) của n×m ô, theo thứ tự hàng trước."""
    th, tw = h // n, w // m
    return [(j * tw, i * th, tw, th) for i in range(n) for j in range(m)]


def blockiness8(gray: np.ndarray) -> float:
    """Tỉ lệ |Δ| tại biên khối 8px so với |Δ| trong khối (>1 ⇒ có artifact khối JPEG)."""
    dx = np.abs(np.diff(gray, axis=1)); dy = np.abs(np.diff(gray, axis=0))
    bx = dx[:, 7::8].mean(); ix = np.delete(dx, np.s_[7::8], axis=1).mean()
    by = dy[7::8, :].mean(); iy = np.delete(dy, np.s_[7::8], axis=0).mean()
    return float((bx + by) / (ix + iy + 1e-6))


def flat_step_ratio(gray: np.ndarray) -> float:
    """Tỉ lệ pixel có gradient ≈ 0 theo cả 2 trục (ảnh posterize/lượng tử hoá có nhiều vùng phẳng)."""
    dx = np.abs(np.diff(gray, axis=1))[:-1, :]; dy = np.abs(np.diff(gray, axis=0))[:, :-1]
    return float(((dx < 0.5) & (dy < 0.5)).mean())


def hottest_tile_box(gray: np.ndarray, label_text: str = "glare ROI") -> dict:
    """Ô có tỉ lệ pixel cháy sáng cao nhất -> dict box (x,y,w,h theo px ảnh gốc) để vẽ lên contact sheet."""
    best, best_frac = None, -1.0
    for (x, y, w, h) in tile_boxes(*gray.shape):
        frac = float((gray[y:y + h, x:x + w] >= SAT_LEVEL).mean())
        if frac > best_frac:
            best, best_frac = (x, y, w, h), frac
    x, y, w, h = best
    return {"x": x, "y": y, "w": w, "h": h, "label": f"{label_text} {best_frac:.0%}", "color": "red"}


def extract(path: str | Path) -> dict:
    """Feature của một ảnh — chỉ dùng pixel (+ kích thước file), KHÔNG đọc gt/."""
    path = Path(path)
    rgb = load_rgb(path)
    g = to_gray(rgb)
    boxes = tile_boxes(*g.shape)
    lv_tiles = np.array([laplacian_variance(g[y:y + h, x:x + w]) for x, y, w, h in boxes])
    sat_tiles = np.array([(g[y:y + h, x:x + w] >= SAT_LEVEL).mean() for x, y, w, h in boxes])
    mean_tiles = np.array([g[y:y + h, x:x + w].mean() for x, y, w, h in boxes])
    bs = brightness_stats(rgb)
    lab, n_lab = label(g >= BLOOM_LEVEL, structure=_EIGHT)        # vùng sáng liên thông lớn nhất – chỉ để phân tích
    big_bright = float(np.bincount(lab.ravel())[1:].max() / g.size) if n_lab else 0.0
    with Image.open(path) as im:
        q = getattr(im, "quantization", None) or {0: [0]}
    return {
        "lap_var": laplacian_variance(rgb),                      # = baseline (toàn ảnh)
        "lap_var_p75_tile": float(np.percentile(lv_tiles, 75)),
        "mean_luma": bs["mean"],
        "p99_luma": float(np.percentile(g, 99)),
        "p50_luma": float(np.percentile(g, 50)),
        "dark_frac": bs["frac_dark"],
        "sat_frac": float((g >= SAT_LEVEL).mean()),
        "sat_tile_max": float(sat_tiles.max()),                  # glare cục bộ (ROI)
        "bloom_area": big_bright,
        "tile_mean_range": float(mean_tiles.max() - mean_tiles.min()),
        "blockiness8": blockiness8(g),
        "flat_ratio": flat_step_ratio(g),
        "jpeg_q0": float(np.mean(q[0])),                         # CHỈ phân tích – lối tắt metadata, không dùng chấm điểm
        "bytes_per_px": path.stat().st_size / g.size,
    }


def derive(F: pd.DataFrame) -> pd.DataFrame:
    F = F.copy()
    F["llv"] = np.log(F.lap_var + 1e-3)
    F["lml"] = np.log(F.mean_luma + 1.0)
    F["block_excess"] = F.blockiness8 - 1.0
    return F


def compute_features(input_dir: Path, img: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([img.reset_index(drop=True), derive(pd.DataFrame([extract(input_dir / p) for p in img.path]))], axis=1)


# =============================================================================================== input table
def image_table(input_dir: Path) -> pd.DataFrame:
    """Bảng ảnh từ manifest.csv (+ camera_id/light từ images.csv nếu có; không có thì 'unknown')."""
    man = pd.read_csv(input_dir / "manifest.csv")
    miss = [c for c in ("file_id", "path", "split", "sha256") if c not in man.columns]
    if miss:
        raise ValueError(f"manifest.csv thiếu cột bắt buộc: {miss}")
    is_img = man["path"].str.lower().str.endswith(IMG_EXT)
    img = man[is_img].rename(columns={"file_id": "image_id"})[["image_id", "path", "split"]].copy()
    meta_path = input_dir / "images.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        miss = [c for c in ("image_id", "camera_id", "light") if c not in meta.columns]
        if miss:
            raise ValueError(f"images.csv thiếu cột bắt buộc: {miss}")
        img = img.merge(meta[["image_id", "camera_id", "light"]], on="image_id", how="left")
    else:
        img["camera_id"] = "unknown"; img["light"] = "unknown"
    img[["camera_id", "light"]] = img[["camera_id", "light"]].fillna("unknown")
    return img.reset_index(drop=True)


def _dev_split(cfg_or_ctx) -> str:
    if isinstance(cfg_or_ctx, RunContext):
        return str(cfg_or_ctx.param("dev_split", "dev"))
    return str((cfg_or_ctx.get("params") or {}).get("dev_split", "dev"))


# =============================================================================================== scoring
def load_params() -> dict:
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"thiếu {PARAMS_PATH.relative_to(ROOT)} — chạy `python -m src.pipeline fit` trước")
    return read_json(PARAMS_PATH)


def ref_llv(params: dict, light: str, cam: str) -> float:
    return params["a_light"].get(light, params["a_mean"]) + params["b_cam"].get(cam, 0.0)


def subscores(F: pd.DataFrame, params: dict) -> pd.DataFrame:
    S = pd.DataFrame(index=F.index)
    S["blur"] = [ref_llv(params, l, c) - v for l, c, v in zip(F.light, F.camera_id, F.llv)]
    S["dark"] = [params["ref_lml"].get(l, params["ref_lml_mean"]) - v for l, v in zip(F.light, F.lml)]
    S["glare"] = F.sat_tile_max.values
    S["compression"] = F.block_excess.values
    return S


def score_poc(F: pd.DataFrame, params: dict, bloom: bool = False):
    """score = max_k(subscore_k / thr_k); score >= 1 <=> vượt ngưỡng ít nhất một loại lỗi."""
    S = subscores(F, params)
    R = pd.DataFrame({k: S[k] / params["thr"][k] for k in TYPES})
    if bloom:
        if params.get("thr_bloom") is None:
            raise ValueError("params thiếu thr_bloom (dev không có ví dụ bloom) -> không chạy được poc_v2")
        ex = np.array([v - params["ref_lml"].get(l, params["ref_lml_mean"]) for l, v in zip(F.light, F.lml)])
        R["glare"] = np.maximum(R["glare"].values, ex / params["thr_bloom"])
    return R.max(axis=1).values, R.idxmax(axis=1).values, R


def to_ranked(ids, score, typ) -> pd.DataFrame:
    score = np.asarray(score)
    df = pd.DataFrame({
        "image_id": np.asarray(ids), "score": score,
        "pred_defect_type": np.where(score >= ACTION_THRESHOLD, typ, "none"),
        "pred_action": np.where(score >= ACTION_THRESHOLD, "RECAPTURE", "KEEP"),
    })
    return df.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)


def validate_output(df: pd.DataFrame, ids) -> None:
    """Output contract: image_id + score bắt buộc, đủ ảnh, không trùng, không NaN."""
    miss = [c for c in ("image_id", "score") if c not in df.columns]
    if miss:
        raise ValueError(f"ranked_review.csv thiếu cột bắt buộc: {miss}")
    if df.image_id.duplicated().any() or set(df.image_id) != set(ids) or df.score.isna().any():
        raise ValueError("ranked_review.csv không hợp lệ (trùng/thiếu image_id hoặc score NaN)")


def variant_of(ctx: RunContext) -> str:
    n = (ctx.run_name or "poc").lower()
    return n if n in ("baseline", "poc_v2") else "poc"


# =============================================================================================== 01 load
def load(ctx: RunContext) -> dict:
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
    data = {"manifest": pd.read_csv(manifest) if manifest.exists() else None}
    data["images"] = image_table(ctx.input_dir)
    return data


# =============================================================================================== 02 process
def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    """Trả về bảng đã xếp hạng (score giảm dần) cho MỌI ảnh trong manifest, kèm cột phụ cho bước rank."""
    img, variant = data["images"], variant_of(ctx)
    params = load_params()
    t0 = time.perf_counter()
    if variant == "baseline":
        # mốc so sánh: một ngưỡng Laplacian toàn ảnh (cùng hàm image_ops.laplacian_variance với feature lap_var)
        lv = np.array([laplacian_variance(load_rgb(ctx.input_dir / p)) for p in img.path])
        bt = params["baseline"]["lap_var_threshold"]
        d = pd.DataFrame({"image_id": img.image_id.values, "score": -np.log(lv + 1e-3),
                          "pred_defect_type": np.where(lv <= bt, "blur", "none"),
                          "pred_action": np.where(lv <= bt, "RECAPTURE", "KEEP")})
        d["top_type"] = "blur"
        d["reason"] = [f"laplacian var={v:.0f} (ngưỡng {bt:.0f})" for v in lv]
    else:
        F = compute_features(ctx.input_dir, img)
        sc, typ, R = score_poc(F, params, bloom=(variant == "poc_v2"))
        d = to_ranked(F.image_id, sc, typ)
        top = pd.Series(typ, index=F.image_id)
        d["top_type"] = d.image_id.map(top).values
        f = F.set_index("image_id").loc[d.image_id]
        d["reason"] = [f"{t} x{s:.2f}{'' if s >= ACTION_THRESHOLD else ' (<ngưỡng)'} | lap={lv:.0f} luma={ml:.0f} sat_tile={st:.2f}"
                       for t, s, lv, ml, st in zip(d.top_type, d.score, f.lap_var, f.mean_luma, f.sat_tile_max)]
        pd.concat([F, R.add_prefix("r_")], axis=1).to_csv(ctx.out("features.csv"), index=False)   # cho ablation / failure-modes
    seconds = time.perf_counter() - t0
    d = d.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    validate_output(d, img.image_id)
    d = d.merge(img[["image_id", "path", "split", "light", "camera_id"]], on="image_id", how="left")
    d["slice"] = d.light + "|" + d.camera_id
    ctx.stats.update(variant=variant, ms_per_image=round(1000 * seconds / len(img), 3),
                     n_flagged_recapture=int((d.pred_action == "RECAPTURE").sum()), params_n_dev=params.get("n_dev"))
    return d


# =============================================================================================== 03 rank
def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    """Top-B (budget) trên pool ngoài split dev (= eval) -> review_queue.csv + contact_sheet.html (ROI glare)."""
    budget = int(ctx.param("budget", 20))
    dev = _dev_split(ctx)
    pool = scored[scored.split != dev] if (scored.split != dev).any() else scored
    q = make_review_queue(pool, id_col="image_id", score_col="score", budget=budget, reason_col="reason",
                          action_col="pred_action", slice_col="slice", evidence_col="path")
    write_review_queue(q, ctx.out("review_queue.csv"))
    types = scored.set_index("image_id").top_type
    boxes = {i: [hottest_tile_box(to_gray(load_rgb(ctx.input_dir / p)))] for i, p in zip(q.item_id, q.evidence_path)
             if types.get(i) == "glare"}
    render_contact_sheet(q, ctx.out("contact_sheet.html"), image_root=ctx.input_dir, thumb_width=320, boxes_by_item=boxes,
                         title=f"T02 · top-{budget} cần review ({ctx.run_name}) — approve=giữ gợi ý; "
                               "edit/reject + override_value=KEEP|RECAPTURE để đổi")
    return q


# =============================================================================================== 04 review
def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    """Nếu có out/<run>/review_queue_reviewed.csv: ghi audit mọi quyết định, áp override (edit/reject + KEEP|RECAPTURE)
    -> final_actions.csv. ranked_review.csv giữ nguyên dự đoán của mô hình (diff có thể kiểm toán)."""
    pred = scored[OUT_COLS]
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    if not reviewed_path.exists():
        return pred
    reviewed = load_review_decisions(reviewed_path).copy()             # ValueError nếu decision sai / thiếu reviewer
    ctx.stats["n_reviewed"] = len(reviewed)
    reviewed["override_value"] = reviewed.override_value.astype(str).str.strip().str.upper()
    edit = reviewed.reviewer_decision.isin(["edit", "reject"])
    bad = sorted(set(reviewed[edit & (reviewed.override_value != "")].override_value) - VALID_ACTIONS)
    if bad:
        raise ValueError(f"override_value không hợp lệ {bad}; chỉ nhận {sorted(VALID_ACTIONS)}")
    no_val = reviewed[edit & (reviewed.override_value == "")]
    if len(no_val):
        print(f"[pipeline] CẢNH BÁO: {len(no_val)} dòng edit/reject không có override_value -> giữ gợi ý: {list(no_val.item_id)}")
    for r in reviewed.itertuples(index=False):                         # mọi quyết định đều vào audit (kể cả approve/skip/escalate)
        ctx.audit.log(actor=r.reviewer, action=f"review:{r.reviewer_decision}", item_id=r.item_id,
                      reason=r.review_note, reviewed_at=r.reviewed_at)
    final = apply_overrides(pred, reviewed, id_col="image_id", value_col="pred_action", audit=ctx.audit)
    final = final.rename(columns={"pred_action": "final_action"}).merge(pred[["image_id", "pred_action"]], on="image_id")
    dec = reviewed.set_index("item_id")["reviewer_decision"]
    final["review_status"] = final.image_id.map(dec).fillna("not_reviewed")
    final["needs_escalation"] = final.review_status.eq("escalate")
    final.to_csv(ctx.out("final_actions.csv"), index=False)
    ctx.stats["n_action_changed"] = int((final.final_action != final.pred_action).sum())
    ctx.stats["n_escalated"] = int(final.needs_escalation.sum())
    return pred


# =============================================================================================== 05 decide
def _metric(m: dict, name: str, slice_: str = "all"):
    for x in m["metrics"]:
        if x["name"] == name and x.get("slice", "all") == slice_:
            return x
    return None


def decide_from_metrics(base: dict, poc: dict, owner: str, scope: str = "t02_raw_qc/eval") -> Decision:
    """Chính sách (chốt trước khi xem kết quả): PoC < baseline ⇒ REWORK; còn dùng nhãn proxy hoặc eval < 100 ảnh ⇒ HOLD;
    ngược lại GO. Đọc metrics JSON do `src.evaluate` sinh (định dạng chuẩn của repo)."""
    rb, rp = _metric(base, "Defect Recall@B"), _metric(poc, "Defect Recall@B")
    gt = str((poc.get("extra") or {}).get("gt", ""))
    proxy = "proxy" in gt.lower()
    n_eval = rp.get("n")
    delta = int(rp["numerator"] - rb["numerator"])
    fa = {s: f"{int(x['numerator'])}/{int(x['denominator'])}" for s in ("day", "dusk", "night")
          if (x := _metric(poc, "False alarm in top-B", f"light={s}"))}
    snap = {"n_eval": n_eval, "B": (poc.get("config") or {}).get("budget"), "gt": gt,
            "recall_baseline": rb["value"], "recall_poc": rp["value"], "tp_baseline": rb["numerator"], "tp_poc": rp["numerator"],
            "n_defects_eval": rp["denominator"], "delta_images": delta, "poc_false_alarm_by_light": fa,
            "caveats": ["eval chỉ 40 ảnh (mỗi ảnh = 5% recall) – chênh vài ảnh chưa có ý nghĩa thống kê",
                        "thr_bloom (poc_v2) dựa trên 1 ví dụ dev và poc_v2 là post-hoc",
                        "chưa kiểm 10 ảnh out-of-domain", "dev không có night × cam_A (mức chuẩn night là ngoại suy)"],
            "next_to_flip_to_GO": ["chấm bằng GT chính thức: python -m src.evaluate --track t02 --gt <gt/labels.csv> với Recall@20 PoC ≥ baseline",
                                   "kiểm 10 ảnh out-of-domain không có false alarm hàng loạt",
                                   "fit lại ngưỡng khi dev có ≥5 ảnh sạch mỗi cặp light × camera"]}
    ev = ["reports/metrics_baseline.json", "reports/metrics_poc.json", "reports/table1_baseline_vs_poc.md",
          "reports/review_queue.csv", "reports/contact_sheet_top20.html", "reports/failure_modes.png", "configs/t02_params.json"]
    if rp["value"] < rb["value"]:
        st, code, why = "REWORK", "POC_BELOW_BASELINE", "PoC kém baseline trên eval – cần làm lại feature/ngưỡng."
    elif proxy or (n_eval or 0) < 100:
        st, code = "HOLD", "EVAL_PROXY_LABELS_SMALL_N"
        why = ("PoC dùng được làm bộ xếp hạng hỗ trợ review top-20 nhưng chưa đủ bằng chứng để tự động KEEP/RECAPTURE: "
               "số liệu tính bằng nhãn proxy trên eval nhỏ, chưa có GT chính thức hay kiểm out-of-domain.")
    else:
        st, code, why = "GO", "POC_GE_BASELINE_OFFICIAL_GT", "PoC ≥ baseline trên GT chính thức."
    return Decision(status=st, reason_code=code, reason=why, owner=owner, scope=scope, evidence=ev, metrics=snap)


def _owner_warning(owner: str) -> None:
    if str(owner).strip().lower().startswith(PLACEHOLDER_OWNERS):
        print("[pipeline] CẢNH BÁO: owner đang là placeholder; rubric yêu cầu tên/vai trò người ký thật (configs/lab.yaml: owner).")


def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    owner = ctx.config.get("owner", "TODO-owner")
    _owner_warning(owner)
    variant = variant_of(ctx)
    ev_out = [str(ctx.out("ranked_review.csv"))]
    if variant == "baseline":
        return Decision(status="HOLD", reason_code="BASELINE_ONLY", reason="Baseline Laplacian: chỉ là mốc so sánh, không phát hành.",
                        owner=owner, scope=ctx.track, evidence=ev_out)
    if variant == "poc_v2":
        return Decision(status="HOLD", reason_code="POST_HOC_VARIANT",
                        reason="poc_v2 thêm feature sau khi thấy lỗi của poc trên eval nên không phải kết quả giữ-out sạch.",
                        owner=owner, scope=ctx.track, evidence=ev_out)
    mb, mp = REPORTS / "metrics_baseline.json", REPORTS / "metrics_poc.json"
    if not (mb.exists() and mp.exists()):
        return Decision(status="HOLD", reason_code="NOT_EVALUATED",
                        reason="Chưa có reports/metrics_baseline.json và metrics_poc.json: chạy src.evaluate rồi `python -m src.pipeline decide`.",
                        owner=owner, scope=ctx.track, evidence=ev_out)
    return decide_from_metrics(read_json(mb), read_json(mp), owner)


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored)
    final = review(ctx, scored, queue)
    final[spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(decide(ctx, final), ctx.out("decision.json"), ctx.out("decision_log.csv"))


# =============================================================================================== CLI: fit (dev only)
def fit_ref_params(clean: pd.DataFrame, F_dev: pd.DataFrame) -> dict:
    """Mô hình cộng cho blur: llv_sạch ≈ a_light + b_cam (cam_A = 0); mức sáng chuẩn theo light."""
    lights = sorted(F_dev.light.unique())
    X = np.array([[float(l == k) for k in lights] + [float(c == "cam_B")] for l, c in zip(clean.light, clean.camera_id)])
    coef, *_ = np.linalg.lstsq(X, clean.llv.values, rcond=None)
    params = {"a_light": dict(zip(lights, map(float, coef[:-1]))), "b_cam": {"cam_A": 0.0, "cam_B": float(coef[-1])},
              "a_mean": float(np.mean(coef[:-1])),
              "ref_lml": {l: float(clean[clean.light == l].lml.median()) for l in lights}}
    params["ref_lml_mean"] = float(np.mean(list(params["ref_lml"].values())))
    return params


def fit_thresholds(S: pd.DataFrame, defect_type: np.ndarray, is_defect: np.ndarray):
    """Ngưỡng mỗi loại = trung điểm (max sạch, min lỗi); nếu không tách được dùng (p95 sạch, p10 lỗi)."""
    thr, report = {}, {}
    for k in TYPES:
        c = S.loc[is_defect == 0, k].values
        dd = S.loc[defect_type == k, k].values
        if len(dd) == 0:
            raise ValueError(f"dev không có ví dụ lỗi '{k}' -> không fit được ngưỡng")
        if c.max() < dd.min():
            t, how = (c.max() + dd.min()) / 2, "midpoint(max clean, min defect)"
        else:
            t, how = (np.percentile(c, 95) + np.percentile(dd, 10)) / 2, "midpoint(p95 clean, p10 defect) - không tách hoàn toàn"
        thr[k] = float(t)
        report[k] = dict(clean_max=float(c.max()), defect_min=float(dd.min()), n_defect_dev=int(len(dd)), rule=how)
    return thr, report


def cmd_fit(a) -> None:
    cfg = load_config(a.config)
    dev_split = _dev_split(cfg)
    input_dir = Path(a.input or cfg.get("input") or FIXTURE)
    problems = verify_manifest(input_dir / "manifest.csv")
    if problems:
        raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
    img = image_table(input_dir)
    lab = pd.read_csv(a.labels)
    miss = [c for c in ("image_id", "proxy_defect_type", "proxy_is_defect") if c not in lab.columns]
    if miss:
        raise ValueError(f"{a.labels} thiếu cột {miss}")
    d = img.merge(lab, on="image_id", how="left")
    d = d[d.split == dev_split].reset_index(drop=True)                      # <-- CHỈ split dev
    if d.proxy_is_defect.isna().any():
        raise ValueError("có ảnh dev không có nhãn proxy")
    F = compute_features(input_dir, d)
    y = F.proxy_is_defect.values.astype(int)
    clean = F[y == 0]
    params = fit_ref_params(clean, F)
    thr, report = fit_thresholds(subscores(F, params), F.proxy_defect_type.values, y)
    ex = np.array([v - params["ref_lml"][l] for l, v in zip(F.light, F.lml)])   # bloom (poc_v2, post-hoc)
    cmax = ex[y == 0].max()
    gl = ex[(F.proxy_defect_type == "glare").values]; gl = gl[gl > cmax + BLOOM_MARGIN]
    if len(gl):
        params["thr_bloom"] = float((cmax + gl.min()) / 2)
        params["bloom_fit"] = {"clean_max_excess": float(cmax), "n_dev_bloom_examples": int(len(gl)), "min_bloom_excess": float(gl.min())}
    else:
        params["thr_bloom"] = None; params["bloom_fit"] = {"clean_max_excess": float(cmax), "n_dev_bloom_examples": 0}
    params["thr"] = thr; params["fit_report"] = report
    best = (-1.0, None)                                                      # baseline: ngưỡng Laplacian đơn, chọn theo F1 trên dev
    for t in np.unique(F.lap_var.values):
        pred = (F.lap_var.values <= t).astype(int); tp = int((pred & y).sum())
        f1 = 2 * tp / (pred.sum() + y.sum() + 1e-9)
        if f1 > best[0]:
            best = (f1, float(t))
    params["baseline"] = {"lap_var_threshold": best[1], "dev_f1": float(best[0])}
    params.update(n_dev=int(len(F)), n_dev_clean=int(len(clean)), seed=42)
    write_json(PARAMS_PATH, params)
    print("thr:", {k: round(v, 4) for k, v in thr.items()}, "| baseline:", params["baseline"], "| thr_bloom:", params["thr_bloom"])
    print(f"wrote {PARAMS_PATH.relative_to(ROOT)} (n_dev={len(F)}, n_dev_clean={len(clean)})")


# =============================================================================================== CLI: ablation
def _evaluate(pred: Path, run_name: str, out_json: Path, gt: Path | str) -> int:
    from . import evaluate as _ev            # evaluator chính thức của repo (metrics/tracks.py::eval_t02)
    return _ev.main(["--track", "t02", "--gt", str(gt), "--pred", str(pred), "--run-name", run_name, "--out", str(out_json)])


def cmd_ablation(a) -> None:
    """PoC KHÔNG chuẩn hoá theo light/camera (một tham chiếu chung), ngưỡng fit lại CHỈ trên dev. Cần out/poc/features.csv."""
    lab = pd.read_csv(a.labels)
    F = pd.read_csv(OUT_DIR / "poc" / "features.csv").merge(lab, on="image_id")
    dev = F[F.split == a.dev_split]; clean = dev[dev.proxy_is_defect == 0]
    ref_llv_, ref_lml = clean.llv.mean(), clean.lml.median()
    S = pd.DataFrame({"blur": ref_llv_ - F.llv, "dark": ref_lml - F.lml, "glare": F.sat_tile_max, "compression": F.block_excess})
    m_dev = (F.split == a.dev_split).values
    thr, _ = fit_thresholds(S[m_dev], F.proxy_defect_type.values[m_dev], F.proxy_is_defect.values[m_dev].astype(int))
    R = S / pd.Series(thr)
    out = to_ranked(F.image_id, R.max(axis=1).values, R.idxmax(axis=1).values)
    (OUT_DIR / "ablation_nonorm").mkdir(parents=True, exist_ok=True)
    out[OUT_COLS].to_csv(OUT_DIR / "ablation_nonorm" / "ranked_review.csv", index=False)
    _evaluate(OUT_DIR / "ablation_nonorm" / "ranked_review.csv", "poc_no_normalisation", REPORTS / "metrics_ablation_nonorm.json", a.gt)
    rows = []                                                                # số ảnh SẠCH bị RECAPTURE (eval) theo light/camera
    for name in ["baseline", "ablation_nonorm", "poc", "poc_v2"]:
        p = pd.read_csv(OUT_DIR / name / "ranked_review.csv").merge(F[["image_id", "split", "light", "camera_id", "proxy_is_defect"]], on="image_id")
        e = p[(p.split != a.dev_split) & (p.proxy_is_defect == 0)]
        f = lambda g: f"{int((g.pred_action == 'RECAPTURE').sum())}/{len(g)}"
        rows.append({"run": name, **{f"clean_{l}_flagged": f(g) for l, g in e.groupby("light")},
                     **{f"clean_{c}_flagged": f(g) for c, g in e.groupby("camera_id")}})
    T = pd.DataFrame(rows); REPORTS.mkdir(exist_ok=True); T.to_csv(REPORTS / "clean_flagged_by_slice.csv", index=False)
    print(T.to_string(index=False)); print({k: round(v, 3) for k, v in thr.items()})


# =============================================================================================== CLI: failure-modes
FM_B, FM_W, FM_H, FM_CAP = 20, 320, 180, 64


def _ranks(name: str, ids: set[str]) -> dict[str, int]:
    r = pd.read_csv(OUT_DIR / name / "ranked_review.csv")
    r = r[r.image_id.isin(ids)].sort_values("score", ascending=False, kind="mergesort")
    return {i: k + 1 for k, i in enumerate(r.image_id)}


def pick_cases(labels: Path, dev_split: str) -> pd.DataFrame:
    """Ca được CHỌN TỰ ĐỘNG từ kết quả các run trên pool eval (nhãn proxy chỉ để phân tích, không ảnh hưởng chấm điểm)."""
    allF = pd.read_csv(OUT_DIR / "poc" / "features.csv")
    F = allF.merge(pd.read_csv(labels), on="image_id"); F = F[F.split != dev_split].copy(); ids = set(F.image_id)
    for run in ("baseline", "ablation_nonorm", "poc", "poc_v2"):
        F[f"rank_{run}"] = F.image_id.map(_ranks(run, ids))
    allF = allF.set_index("image_id")
    B = FM_B
    clean = F.proxy_is_defect == 0
    fm = []
    n1 = F[clean & (F.light == "night") & ((F.rank_baseline <= B) | (F.rank_ablation_nonorm <= B))].sort_values("rank_ablation_nonorm").head(3)
    fm += [("FM1 night useful flagged", r) for _, r in n1.iterrows()]
    g2 = F[(F.proxy_defect_type == "glare") & (F.rank_baseline > B) & (F.rank_poc <= B)].sort_values("rank_baseline", ascending=False).head(2)
    fm += [("FM2 local glare missed by baseline", r) for _, r in g2.iterrows()]
    c3 = F[clean & (F.camera_id == "cam_B") & (F.rank_baseline <= B)].sort_values("rank_baseline").head(3)
    fm += [("FM3 cam_B soft optics false alarm", r) for _, r in c3.iterrows()]
    m4 = F[(F.proxy_is_defect == 1) & (F.rank_poc > B)].sort_values("rank_poc").head(3)
    fm += [("FM4 PoC v1 miss (bloom)", r) for _, r in m4.iterrows()]
    out = pd.DataFrame([{"failure_mode": t, **r.to_dict()} for t, r in fm])
    # lap_var(_1)/lap_var(_0) cùng cảnh: ~5x với cam_B (quang học mềm) — bằng chứng metadata cho FM3
    out["lapvar_sibling_ratio"] = [allF.lap_var.get(i.rsplit("_", 1)[0] + "_1", float("nan")) / allF.lap_var[i] if i.endswith("_0") else float("nan")
                                   for i in out.image_id]
    return out


def render_failure_modes(cases: pd.DataFrame, out_png: Path) -> None:
    font = ImageFont.load_default()                                            # ASCII: font mặc định của Pillow không có dấu tiếng Việt
    modes = list(dict.fromkeys(cases.failure_mode)); cols = max(cases.groupby("failure_mode").size())
    canvas = Image.new("RGB", (cols * FM_W, len(modes) * (FM_H + FM_CAP + 18)), "white"); d = ImageDraw.Draw(canvas)
    for ri, mode in enumerate(modes):
        y0 = ri * (FM_H + FM_CAP + 18); d.text((4, y0 + 2), mode, fill="black", font=font)
        for ci, (_, r) in enumerate(cases[cases.failure_mode == mode].iterrows()):
            rgb = load_rgb(FIXTURE / r.path); img = Image.fromarray(rgb)
            if mode.startswith("FM2"):
                b = hottest_tile_box(to_gray(rgb), "")
                ImageDraw.Draw(img).rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], outline="red", width=2)
            x0 = ci * FM_W; canvas.paste(img, (x0, y0 + 16))
            lines = [f"{r.image_id} {r.light}/{r.camera_id} true={r.proxy_defect_type}",
                     f"lap_var={r.lap_var:.0f} luma={r.mean_luma:.0f} sat_tile={r.sat_tile_max:.2f}",
                     f"rank base/abl/poc/v2 = {int(r.rank_baseline)}/{int(r.rank_ablation_nonorm)}/{int(r.rank_poc)}/{int(r.rank_poc_v2)} (B={FM_B})"]
            for li, t in enumerate(lines):
                d.text((x0 + 3, y0 + 16 + FM_H + 3 + li * 13), t, fill="black", font=font)
    canvas.save(out_png)


def cmd_failure_modes(a) -> None:
    cases = pick_cases(Path(a.labels), a.dev_split)
    cols = ["failure_mode", "image_id", "light", "camera_id", "proxy_defect_type", "lap_var", "mean_luma", "sat_tile_max", "llv", "lml",
            "lapvar_sibling_ratio", "rank_baseline", "rank_ablation_nonorm", "rank_poc", "rank_poc_v2"]
    REPORTS.mkdir(exist_ok=True)
    cases[cols].to_csv(REPORTS / "failure_modes.csv", index=False)
    render_failure_modes(cases, REPORTS / "failure_modes.png")
    print(cases[cols].drop(columns=["llv", "lml"]).to_string(index=False))


# =============================================================================================== CLI: decide / all
def cmd_decide(a) -> None:
    owner = a.owner or load_config(a.config).get("owner", "TODO-owner")
    _owner_warning(owner)
    d = decide_from_metrics(read_json(REPORTS / "metrics_baseline.json"), read_json(REPORTS / "metrics_poc.json"), owner, a.scope)
    out = Path(a.out_dir)
    write_decision(d, out / "decision.json", out / "decision_log.csv")
    print(d.canonical, d.reason_code, "| owner:", d.owner, "->", out / "decision.json")


def cmd_all(a) -> None:
    """fit → baseline/poc/poc_v2 → chấm → ablation → bảng 1 → failure-modes → decision → gom file vào reports/."""
    from . import compare as _cmp
    from . import main as runner
    gt = a.gt or str(PROXY_GT_PATH)
    cfg = str(ROOT / "configs" / "lab.yaml")
    cmd_fit(argparse.Namespace(config=cfg, input=None, labels=a.labels))
    for name in ("baseline", "poc", "poc_v2"):
        rc = runner.main(["--config", cfg, "--track", "t02", "--input", str(FIXTURE), "--output", str(OUT_DIR / name),
                          "--run-name", name, "--offline", "--seed", "42"])
        if rc != 0:
            sys.exit(f"src.main {name} thất bại (status={rc})")
    shutil.copy(OUT_DIR / "poc" / "ranked_review.csv", OUT_DIR / "ranked_review.csv")      # file nộp chính = PoC v1
    for name, rn in (("baseline", "baseline"), ("poc", "poc"), ("poc_v2", "poc_v2_posthoc")):
        if _evaluate(OUT_DIR / name / "ranked_review.csv", rn, REPORTS / f"metrics_{rn}.json", gt) != 0:
            sys.exit(f"src.evaluate {name} thất bại")
    ns = argparse.Namespace(labels=a.labels, dev_split="dev", gt=gt)
    cmd_ablation(ns)
    _cmp.main([str(REPORTS / "metrics_baseline.json"), str(REPORTS / "metrics_poc.json"), "--out", str(REPORTS / "table1_baseline_vs_poc.md")])
    cmd_failure_modes(ns)
    cmd_decide(argparse.Namespace(owner=a.owner, config=cfg, scope="t02_raw_qc/eval", out_dir=str(OUT_DIR / "poc")))
    for src_name, dst in (("review_queue.csv", "review_queue.csv"), ("contact_sheet.html", "contact_sheet_top20.html"),
                          ("decision.json", "decision.json"), ("decision_log.csv", "decision_log.csv")):
        shutil.copy(OUT_DIR / "poc" / src_name, REPORTS / dst)
    print("OK — kết quả trong out/ và reports/")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "lab.yaml"))
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit", help="fit tham số/ngưỡng CHỈ trên split dev -> configs/t02_params.json")
    f.add_argument("--input"); f.add_argument("--labels", default=str(LABELS_PATH)); f.set_defaults(fn=cmd_fit)
    ab = sp.add_parser("ablation", help="PoC không chuẩn hoá theo light/camera (cần out/poc/features.csv)")
    ab.add_argument("--labels", default=str(LABELS_PATH)); ab.add_argument("--gt", default=str(PROXY_GT_PATH))
    ab.add_argument("--dev-split", default="dev"); ab.set_defaults(fn=cmd_ablation)
    fm = sp.add_parser("failure-modes", help="ảnh + metadata cho các failure mode -> reports/failure_modes.{png,csv}")
    fm.add_argument("--labels", default=str(LABELS_PATH)); fm.add_argument("--dev-split", default="dev"); fm.set_defaults(fn=cmd_failure_modes)
    de = sp.add_parser("decide", help="GO/HOLD/REWORK từ reports/metrics_{baseline,poc}.json (sau khi chấm)")
    de.add_argument("--owner"); de.add_argument("--scope", default="t02_raw_qc/eval"); de.add_argument("--out-dir", default=str(OUT_DIR / "poc"))
    de.set_defaults(fn=cmd_decide)
    al = sp.add_parser("all", help="toàn bộ chuỗi: fit → run ×3 → evaluate → ablation → compare → failure-modes → decide")
    al.add_argument("--owner"); al.add_argument("--gt"); al.add_argument("--labels", default=str(LABELS_PATH)); al.set_defaults(fn=cmd_all)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
