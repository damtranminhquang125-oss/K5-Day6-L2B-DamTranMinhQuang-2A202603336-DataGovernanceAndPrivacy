"""Test của nhóm cho T02 Raw QC Triage: ca thành công + ca lỗi/biên (chạy offline, không cần gt/ của giảng viên)."""
import json
import shutil
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from src import main as runner
from src import pipeline as P
from src.metrics.tracks import get_track

FX = P.FIXTURE


def _run(out, run_name="poc", extra=()):
    return runner.main(["--track", "t02", "--output", str(out), "--run-name", run_name, "--offline", "--seed", "42", *extra])


def _recall(pred_csv):
    """Defect Recall@B theo evaluator chính thức, GT = nhãn proxy."""
    res = get_track("t02").evaluator(P.PROXY_GT_PATH, pred_csv, {"budget": 20, "split": "eval"})
    return [m for m in res.metrics if m.primary][0]


# ------------------------------------------------------------------------------- thành công
def test_poc_end_to_end_contract(tmp_path):
    out = tmp_path / "poc"
    assert _run(out) == 0
    for f in ["ranked_review.csv", "review_queue.csv", "contact_sheet.html", "decision.json", "decision_log.csv", "features.csv", "run_info.json"]:
        assert (out / f).exists(), f
    r = pd.read_csv(out / "ranked_review.csv")
    assert {"image_id", "score", "pred_defect_type", "pred_action"} <= set(r.columns)
    assert len(r) == 96 and not r.image_id.duplicated().any() and not r.score.isna().any()
    assert r.score.is_monotonic_decreasing                                   # score cao = nghi lỗi hơn, xếp trước
    assert set(r.pred_action) <= {"KEEP", "RECAPTURE"}
    d = json.loads((out / "decision.json").read_text(encoding="utf-8"))
    assert d["owner"].strip() and d["reason_code"] and d["canonical"] in {"GO", "HOLD", "REWORK"}


def test_review_queue_is_top_budget_of_eval_pool_only(tmp_path):
    out = tmp_path / "poc"
    assert _run(out) == 0
    q = pd.read_csv(out / "review_queue.csv")
    split = pd.read_csv(FX / "manifest.csv").set_index("file_id").split
    assert len(q) == 20                                                       # budget lấy từ configs/tracks/t02.yaml
    assert set(split.loc[q.item_id]) == {"eval"}                             # không đưa ảnh dev vào hàng đợi review
    assert q.score.is_monotonic_decreasing


def test_same_seed_same_output(tmp_path):
    assert _run(tmp_path / "a") == 0 and _run(tmp_path / "b") == 0
    assert (tmp_path / "a" / "ranked_review.csv").read_text() == (tmp_path / "b" / "ranked_review.csv").read_text()


def test_poc_not_worse_than_baseline_on_proxy_labels(tmp_path):
    """Báo cả kết quả bằng 0/âm: chỉ khẳng định PoC >= baseline trên nhãn proxy, KHÔNG phải GT chính thức."""
    assert _run(tmp_path / "base", "baseline") == 0 and _run(tmp_path / "poc", "poc") == 0
    b, p = _recall(tmp_path / "base" / "ranked_review.csv"), _recall(tmp_path / "poc" / "ranked_review.csv")
    assert b.denominator == p.denominator == 20
    assert p.value >= b.value


def test_human_override_is_applied_and_audited(tmp_path):
    out = tmp_path / "poc"
    assert _run(out) == 0
    q = pd.read_csv(out / "review_queue.csv", dtype=str, keep_default_na=False)
    target = q.item_id.iloc[0]
    q.loc[0, ["reviewer", "reviewer_decision", "override_value", "review_note"]] = ["tin", "edit", "keep", "glare không ảnh hưởng biển số"]
    q.loc[1, ["reviewer", "reviewer_decision"]] = ["tin", "approve"]
    q.to_csv(out / "review_queue_reviewed.csv", index=False)
    assert _run(out) == 0
    fa = pd.read_csv(out / "final_actions.csv").set_index("image_id")
    assert fa.loc[target, "final_action"] == "KEEP" and fa.loc[target, "pred_action"] == "RECAPTURE"
    assert pd.read_csv(out / "ranked_review.csv").set_index("image_id").loc[target, "pred_action"] == "RECAPTURE"   # dự đoán gốc không bị sửa
    log = [json.loads(l) for l in (out / "audit_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["action"] == "override:edit" and e["item_id"] == target and e["after"] == "KEEP" for e in log)


# ------------------------------------------------------------------------------- lỗi / biên
def test_missing_frozen_params_exit_2(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PARAMS_PATH", tmp_path / "nope.json")
    assert _run(tmp_path / "o") == 2                                          # FileNotFoundError -> exit 2, không im lặng


def test_tampered_image_fails_provenance(tmp_path):
    fx = tmp_path / "fx"
    shutil.copytree(FX, fx, ignore=shutil.ignore_patterns("images.zip", "proxy_gt", "gt"))
    img = fx / "images" / "img_00_0.jpg"
    img.write_bytes(img.read_bytes() + b"\x00")                               # đổi 1 byte -> sai sha256
    rc = runner.main(["--track", "t02", "--input", str(fx), "--output", str(tmp_path / "o"), "--offline"])
    assert rc == 2
    assert not (tmp_path / "o" / "ranked_review.csv").exists()


def test_invalid_override_value_rejected(tmp_path):
    out = tmp_path / "poc"
    assert _run(out) == 0
    q = pd.read_csv(out / "review_queue.csv", dtype=str, keep_default_na=False)
    q.loc[0, ["reviewer", "reviewer_decision", "override_value"]] = ["tin", "edit", "MAYBE"]
    q.to_csv(out / "review_queue_reviewed.csv", index=False)
    assert _run(out) == 2


def test_not_implemented_pipeline_exits_3(tmp_path, monkeypatch):
    (tmp_path / "stub_pipeline.py").write_text("def run(ctx):\n    raise NotImplementedError('todo')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    rc = runner.main(["--track", "t02", "--pipeline", "stub_pipeline", "--output", str(tmp_path / "o"), "--offline"])
    assert rc == 3


def test_without_images_csv_falls_back_to_unknown_group(tmp_path):
    fx = tmp_path / "fx"
    shutil.copytree(FX, fx, ignore=shutil.ignore_patterns("images.zip", "proxy_gt", "gt", "images.csv"))
    man = pd.read_csv(fx / "manifest.csv").query("file_id != 'images'")     # manifest không còn liệt kê file phụ đã bỏ
    man.to_csv(fx / "manifest.csv", index=False)
    assert runner.main(["--track", "t02", "--input", str(fx), "--output", str(tmp_path / "o"), "--offline"]) == 0
    f = pd.read_csv(tmp_path / "o" / "features.csv")
    assert set(f.light) == {"unknown"} and set(f.camera_id) == {"unknown"}    # chuẩn hoá rơi về tham chiếu gộp, vẫn chạy


# ------------------------------------------------------------------------------- unit
def test_blockiness_flat_vs_blocky():
    rng = np.random.default_rng(42)
    smooth = rng.normal(128, 5, (64, 64)).astype(np.float32)
    blocky = np.kron(rng.integers(0, 255, (8, 8)), np.ones((8, 8))).astype(np.float32)
    assert P.blockiness8(smooth) < 1.3 < P.blockiness8(blocky)


def test_hottest_tile_box_finds_saturated_tile():
    g = np.full((180, 320), 100, np.float32)
    g[100:140, 200:260] = 255
    b = P.hottest_tile_box(g)
    assert b["x"] <= 230 < b["x"] + b["w"] and b["y"] <= 120 < b["y"] + b["h"]


def test_to_ranked_threshold_and_order():
    r = P.to_ranked(["a", "b", "c"], [0.4, 2.5, 1.0], ["blur", "glare", "dark"])
    assert r.image_id.tolist() == ["b", "c", "a"]
    assert r.pred_action.tolist() == ["RECAPTURE", "RECAPTURE", "KEEP"]       # score >= 1 -> RECAPTURE
    assert r.pred_defect_type.tolist() == ["glare", "dark", "none"]


def test_fit_thresholds_requires_every_defect_type_in_dev():
    S = pd.DataFrame({k: [0.0, 1.0] for k in P.TYPES})
    with pytest.raises(ValueError, match="dev không có ví dụ lỗi"):
        P.fit_thresholds(S, np.array(["blur", "none"]), np.array([1, 0]))


def _metrics(recall, tp, n=40, gt="fixtures/t02_raw_qc/gt/labels.csv"):
    return {"extra": {"gt": gt}, "config": {"budget": 20},
            "metrics": [{"name": "Defect Recall@B", "slice": "all", "value": recall, "numerator": tp, "denominator": 20, "n": n}]}


def test_decision_policy():
    d = lambda b, p: P.decide_from_metrics(b, p, owner="QA lead").canonical
    assert d(_metrics(0.9, 18), _metrics(0.8, 16)) == "REWORK"                                        # PoC kém baseline
    assert d(_metrics(0.8, 16), _metrics(0.9, 18)) == "HOLD"                                          # eval chỉ 40 ảnh
    assert d(_metrics(0.8, 16, 200), _metrics(0.9, 18, 200)) == "GO"                                  # GT chính thức, N đủ lớn
    assert d(_metrics(0.8, 16, 200, "proxy_gt/labels.csv"), _metrics(0.9, 18, 200, "proxy_gt/labels.csv")) == "HOLD"   # nhãn proxy
