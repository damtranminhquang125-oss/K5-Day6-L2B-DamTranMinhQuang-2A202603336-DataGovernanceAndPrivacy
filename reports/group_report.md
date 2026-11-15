# Group Report — Day 06 Data Lab (Field-to-Release PoC)

> Điền đầy đủ, cụ thể, không viết chung chung. Mọi con số phải lấy từ file do script sinh ra
> (`reports/metrics_*.json`, `reports/table1_baseline_vs_poc.md`) — không gõ tay, không làm tròn mất đơn vị.
> 3–5 trang (hoặc README + report.md tương đương).

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID | G06 |
| Challenge | T02 · Raw QC Triage: ảnh mờ, lóa và thiếu sáng |
| Thành viên | Đàm Trần Minh Quang |
| Git URL | https://github.com/damtranminhquang125-oss/K5-Day6-L2B-DamTranMinhQuang-2A202603336-DataGovernanceAndPrivacy |
| Tag `demo-180m` (commit) | |
| Tag `final-24h` (commit) | |
| Deadline (giờ VN) | |

## 01 · Problem statement

- Ai bị ảnh hưởng / ai chịu hậu quả khi hệ thống sai? Bên dùng bộ ảnh phía sau (gán nhãn, huấn luyện, đọc biển số) và đội hiện trường phải chụp lại. Sai theo hai hướng: **bỏ sót** (ảnh mờ / cháy sáng / thiếu sáng / nén khối lọt vào dataset) hoặc **báo nhầm** (loại mất ảnh đêm hữu ích, tốn lượt review và chụp lại).
- Điểm nghẽn cụ thể: người review chỉ duyệt được **B = 20** ảnh trong pool eval 40 ảnh (20 ảnh lỗi), nên phải xếp hạng sao cho 20 ảnh đó chứa nhiều lỗi nhất. Một ngưỡng Laplacian toàn ảnh không đủ: camera `cam_B` mềm hơn bị báo nhầm, glare cục bộ trên ảnh nét bị bỏ qua, nén khối làm Laplacian *tăng*, ảnh đêm sạch dễ bị coi là thiếu sáng.
- Đầu vào → đầu ra: thư mục ảnh + `manifest.csv` + `images.csv` (`camera_id`, `light`) → `out/ranked_review.csv` (`image_id`, `score`, `pred_defect_type`, `pred_action` KEEP/RECAPTURE) + hàng đợi review top-20 + contact sheet + `decision.json`.
- Tiêu chí GO / HOLD / REWORK (chốt trước khi chạy eval): **REWORK** nếu Recall@B của PoC < baseline; **HOLD** nếu GT là nhãn proxy hoặc eval < 100 ảnh; **GO** nếu PoC ≥ baseline trên GT chính thức với N ≥ 100 (`src/pipeline.py::decide_from_metrics`). *Lưu ý: chính sách này được viết sau lần xem kết quả proxy đầu tiên; nhóm cần xác nhận thời điểm chốt (xem `metric_contract.md`).*
- Giả định của nhóm (cần DPO/legal/domain owner xác nhận): (1) ảnh synthetic đại diện cho ảnh hiện trường; (2) lỗi nào cũng chụp lại được nên `RECAPTURE` là hành động khả thi; (3) bỏ sót một ảnh lỗi nghiêm trọng hơn tốn một lượt review thừa; (4) nhãn proxy phản ánh đúng loại lỗi thấy được bằng mắt.

## 02 · Data & provenance

| Split | N | Nguồn / link | License | Synthetic? |
|---|---|---|---|---|
| dev | 56 ảnh (14 cảnh × 4 biến thể; 28 sạch + 28 lỗi theo nhãn proxy) | `fixtures/t02_raw_qc/manifest.csv` — `synthetic:tools/make_fixtures.py (seed 42)` | CC0-1.0 (synthetic, generated for AI20K Day06) | Có |
| eval / heldout | 40 ảnh (10 cảnh × 4 biến thể; 20 sạch + 20 lỗi theo nhãn proxy) | như trên | như trên | Có |

- Schema & class mapping: `images.csv` = `image_id, path, camera_id` (cam_A/cam_B), `light` (day/dusk/night), `split`. Nhãn proxy = `defect_type ∈ {none, blur, glare, dark, compression}`, `is_defect ∈ {0,1}` (không có `severity`). Eval có 8 compression, 4 blur, 4 dark, 4 glare; dev có 10 compression, 8 blur, 6 dark, 4 glare.
- Seed: 42 · Cách chia split: theo **cảnh gốc** (`group_id` = `img_XX`): dev 14 cảnh, eval 10 cảnh, không cảnh nào nằm ở cả hai split.
- `manifest.csv` verified (sha256): **có** — `python -m tools.verify_manifests` → `14 manifests, 0 failing`; ngoài ra `src.main` kiểm lại mỗi lần chạy (`strict_provenance: true`). Ghi chú: `images.csv` ban đầu chỉ nằm trong `fixtures/t02_raw_qc/images.zip`; đã giải nén ra thư mục fixture, sha256 khớp dòng `images` của manifest.
- Dữ liệu thật dùng (nếu có) và cách xây GT cho nó: **không dùng** (không có `data/real/`). GT dùng để tự kiểm là **nhãn proxy gán bằng mắt** (`fixtures/t02_raw_qc/proxy_labels.csv`, `proxy_gt/labels.csv`); biến thể `_0`/`_1` của mỗi cảnh = sạch, `_2`/`_3` = loại lỗi nhìn thấy. `img_03_3` ban đầu nhìn giống blur nhẹ nhưng được sửa thành compression vì bảng lượng tử JPEG trùng 17 ảnh nén khác.

## 03 · Research & design

- 2–3 references (URL + ngày xem) và **vì sao** chọn kỹ thuật: tài liệu gốc được nêu trong đề T02 là ACDC (ETH Zürich), ISO/IEC TR 5259-6:2026, CVAT automated QA — nhóm điền URL/ngày xem ở §11. Lý do chọn kỹ thuật dựa trên bằng chứng của chính nhóm: (a) Laplacian toàn ảnh không tách được camera mềm (`lap_var` ảnh sạch cam_B thấp hơn khoảng 5 lần cùng cảnh) ⇒ chuẩn hoá theo `light`/`camera_id`; (b) glare cục bộ chỉ chiếm vài % diện tích ⇒ feature theo ô 4×4 (ROI); (c) nén khối tạo biên khối 8px làm Laplacian tăng ⇒ feature blockiness thay vì chỉ nhìn độ nét.
- Sơ đồ data flow: `manifest.csv + images.csv` → **load** (kiểm sha256, bảng ảnh) → **process** (feature: Laplacian, mean luma, tỉ lệ pixel cháy theo ô 4×4, blockiness 8px → điểm từng loại lỗi / ngưỡng → `score = max`) → **rank** (top-B ngoài split dev → `review_queue.csv` + `contact_sheet.html` có khung ROI glare) → **review** (người duyệt `approve/edit/reject/escalate/skip`, override KEEP|RECAPTURE → `final_actions.csv` + `audit_log.jsonl`) → **decide** (`decision.json`, `decision_log.csv`) → report (`reports/`).
- Thành phần offline (model/weights + checksum + license nếu có): **không có model/weights** — chỉ numpy, scipy, Pillow. Tham số đóng băng `configs/t02_params.json` (sinh từ dev bằng `python -m src.pipeline fit`; sha256 bắt đầu `5a4f8956cb493367`, tính lại bằng `sha256sum`).
- Trade-offs đã cân nhắc: luật + ngưỡng thay vì mô hình học vì dev chỉ có 56 ảnh (28 sạch); `score = max` của các tỉ lệ feature/ngưỡng để mỗi loại lỗi giải thích được (`pred_defect_type`); **không dùng** `jpeg_q0` (bảng lượng tử JPEG tách hoàn hảo ảnh nén trong bộ synthetic nhưng là lối tắt metadata, không tổng quát hoá); mô hình cộng `light + camera` để xử lý nhóm chưa thấy (night × cam_A không có trong dev) bằng tham chiếu gộp; đổi lại tham chiếu night chỉ dựa trên 2 ảnh sạch dev nên là ngoại suy.

## 04 · Baseline vs PoC

- Baseline (mô tả, version/config, commit): một ngưỡng Laplacian variance toàn ảnh (`common/image_ops.laplacian_variance`), RECAPTURE nếu `lap_var ≤ 744.96` (chọn theo F1 trên dev = 0.842, `configs/t02_params.json`). Chạy: `python -m src.main --track t02 --output out/baseline --run-name baseline --offline --seed 42`. Commit: ____
- PoC (mô tả, version/config, commit): PoC v1 = điểm theo 4 loại lỗi (blur theo mô hình `light+camera`, dark theo mức sáng chuẩn của `light`, glare theo ô cháy sáng nhiều nhất, compression theo blockiness 8px), `score = max(điểm/ngưỡng)`, RECAPTURE nếu `score ≥ 1`. Chạy: `--run-name poc`. Commit: ____
- Cùng budget / cùng heldout / cùng seed: **xác nhận** — B = 20, cùng 40 ảnh eval, seed 42, cùng nguồn ngưỡng (dev).

**Bảng 1 · Baseline vs PoC & hard slices** (số lấy từ `reports/metrics_baseline.json`, `reports/metrics_poc.json`):

| Metric (unit) | Baseline | PoC | Δ & chú thích |
|---|---|---|---|
| Primary: Defect Recall@B (GT defects in top-B / GT defects) | 0.85 (17/20) | 0.95 (19/20) | +0.10 (+2 ảnh, +11.8%); higher is better; Wilson 95%: [0.64, 0.95] vs [0.76, 0.99] chồng lấn ⇒ chưa kết luận có ý nghĩa thống kê |
| Hard-slice: False alarm in top-B, `light=night` | 0.25 (2/8) | 0.00 (0/8) | −0.25; N = 16 ảnh night (8 ảnh sạch). Slice `day`: baseline = 0, PoC = 0.10 (1/10) |
| Review@B / miss / FP | TP 17 · miss 3 · FP 3 | TP 19 · miss 1 · FP 1 | B = 20 |
| Runtime / 100 items | 0.21 s | 0.78 s | CPU, Linux x86_64, Python 3.12.3 (môi trường chạy thử; dao động giữa các lần chạy, nhóm đo lại trên máy mình) |

Output đầy đủ của `python -m src.compare` (`reports/table1_baseline_vs_poc.md`):

| Metric (unit) | Slice | Baseline | PoC | Δ abs | Δ rel | Better | N |
|---|---|---|---|---|---|---|---|
| **Defect Recall@B** (GT defects in top-B / GT defects) | all | 0.8500 | 0.9500 | 0.1000 | +11.8% | higher | 40 |
| Precision@B (GT defects in top-B / B) | all | 0.8500 | 0.9500 | 0.1000 | +11.8% | higher | 40 |
| Defect-type accuracy (GT defects) (fraction) | all | 0.2000 | 0.8500 | 0.6500 | +325.0% | higher | 20 |
| False alarm in top-B (clean images in top-B / clean images) | light=day | 0.0000 | 0.1000 | 0.1000 | baseline=0 | lower | 20 |
| False alarm in top-B (clean images in top-B / clean images) | light=dusk | 0.5000 | 0.0000 | -0.5000 | -100.0% | lower | 4 |
| False alarm in top-B (clean images in top-B / clean images) | light=night | 0.2500 | 0.0000 | -0.2500 | -100.0% | lower | 16 |

Baseline luôn đoán `blur` nên Type-acc 0.20 (= 4/20 ảnh lỗi là blur) là hệ quả, không phải so sánh công bằng về phân loại.

## 05 · Evaluation protocol

- Công thức primary metric, đơn vị đánh giá, TP/FP/FN, denominator: Recall@B = #ảnh lỗi thật trong top-B / #ảnh lỗi thật trên eval (denominator 20); đơn vị = ảnh; TP = lỗi trong top-B, FP = sạch trong top-B, FN = lỗi ngoài top-B. Vì #lỗi = B = 20 nên Recall@B = Precision@B (xem `reports/metric_contract.md`).
- Threshold / budget B / K và lúc chốt: B = 20 (`configs/tracks/t02.yaml`); ngưỡng đóng băng từ dev trong `configs/t02_params.json`: blur 2.695, dark 0.773, glare 0.0603, compression 0.775 (biên tách sạch/lỗi của compression hẹp: sạch tối đa 0.664 so với lỗi tối thiểu 0.886). Lúc chốt: ____
- Độ bất định (CI/bootstrap/variance) khi N nhỏ: Wilson 95% cho Recall@20 (baseline [0.64, 0.95], PoC [0.76, 0.99]); ảnh trong cùng cảnh không độc lập nên khoảng thật rộng hơn; bootstrap theo cảnh chưa chạy. Mỗi ảnh = 5% Recall.
- Lệnh evaluate chính xác: `python -m src.evaluate --track t02 --gt fixtures/t02_raw_qc/proxy_gt/labels.csv --pred out/poc/ranked_review.csv --run-name poc --out reports/metrics_poc.json` (GT proxy); với GT chính thức đổi `--gt fixtures/t02_raw_qc/gt/labels.csv`.

## 06 · Result slices

Recall@B / False alarm trong top-B (từ `extra.slices` của `reports/metrics_*.json`):

| Slice | N | Baseline | PoC | Ghi chú |
|---|---|---|---|---|
| light=day | 20 | R 0.90 · FA 0.00 | R 1.00 · FA 0.10 | FA của PoC là `img_23_1` (sạch, score 0.67 < 1, vào top-20 vì B lớn hơn số ảnh vượt ngưỡng) |
| light=dusk | 4 | R 1.00 · FA 0.50 | R 1.00 · FA 0.00 | N rất nhỏ (2 ảnh sạch) |
| light=night | 16 | R 0.75 · FA 0.25 | R 0.875 · FA 0.00 | PoC bỏ sót `img_03_2` (glare bloom) |
| defect=compression | 8 | R 0.75 | R 1.00 | baseline bỏ sót 2 ảnh nén ban đêm |
| defect=glare | 4 | R 0.75 | R 0.75 | baseline bỏ sót `img_05_3`; PoC bỏ sót `img_03_2` |
| defect=blur / dark | 4 / 4 | R 1.00 / 1.00 | R 1.00 / 1.00 | cả hai đều bắt hết |
| camera cam_B (ảnh sạch bị RECAPTURE) | 6 | 3/6 | 0/6 | `reports/clean_flagged_by_slice.csv`, đo ở mức `pred_action`, không phải top-B |

- **Failure case 1 — cam_B quang học mềm bị báo nhầm.** `img_06_0`, `img_15_0`, `img_03_0` (ảnh sạch, cam_B): `lap_var` = 276 / 313 / 354, baseline xếp hạng #9 / #10 / #13 (false alarm), PoC xếp #22 / #34 / #28. Metadata: ảnh `_0` có `lap_var` thấp hơn ảnh `_1` cùng cảnh 5.3 / 5.5 / 5.3 lần. Root cause: một ngưỡng toàn cục không thể vừa đúng cho hai camera; PoC trừ offset camera (−0.89 log `lap_var`). Giả định cần xác nhận: các ảnh này là sạch (nếu GT chính thức coi là blur nhẹ thì kết luận đổi).
- **Failure case 2 — glare cục bộ trên ảnh nét.** `img_05_3` (day, cam_A): `lap_var` = 1324 (> ngưỡng 745), baseline hạng #31 (bỏ sót); `sat_tile_max` = 0.189 đưa PoC lên hạng #6. Root cause: Laplacian toàn ảnh bị phần còn lại làm loãng; feature theo ô 4×4 bắt được vùng cháy. Tương tự, nén ban đêm `img_01_3`, `img_13_2` (`lap_var` 1094 / 1207, blockiness 2.78 / 2.49) bị baseline xếp #27 / #28 vì biên khối làm Laplacian tăng.
- **Failure case 3 — ảnh đêm sạch bị coi là thiếu sáng.** Khi bỏ chuẩn hoá (ablation `poc_no_normalisation`) 3/8 ảnh đêm sạch (`img_06_0` #18, `img_06_1` #19, `img_01_1` #20) chiếm cuối top-20 và Recall@20 giảm về 0.85; PoC có chuẩn hoá theo `light` là 0/8. Root cause: ảnh đêm sạch có mean luma ≈ 30 so với 90–120 của ngày/hoàng hôn, quy tắc độ sáng tuyệt đối không phân biệt "cảnh đêm" với "chụp thiếu sáng". Rủi ro còn lại: tham chiếu night chỉ dựa trên 2 ảnh sạch dev (cam_B), night × cam_A chưa từng thấy ở dev.
- **Failure case 4 (lỗi của chính PoC) — glare bloom.** `img_03_2` (night, cam_B): `sat_tile_max` = 0.018 (quầng sáng mịn, ít pixel cháy), PoC hạng #25 (bỏ sót). PoC v2 (post-hoc) thêm feature bloom và đưa về hạng #4, nhưng ngưỡng bloom chỉ dựa trên **1 ví dụ dev** nên mong manh. Hình + metadata: `reports/failure_modes.png`, `reports/failure_modes.csv`.
- Latency / compute (máy, CPU, ms/item): PoC ≈ 8 ms/ảnh, baseline ≈ 2 ms/ảnh (`reports/metrics_*.json` → `runtime_per_100_items_s` = 0.78 / 0.21 s; Linux x86_64, Python 3.12.3). Máy của nhóm: ____

## 07 · Human & decision

- Review queue (file, B, cách ưu tiên): `reports/review_queue.csv` (copy từ `out/poc/review_queue.csv`), B = 20, top-B theo `score` giảm dần trên ảnh ngoài split dev (= eval); 17/20 ảnh được gợi ý RECAPTURE. Xem bằng `reports/contact_sheet_top20.html` (offline, có khung ROI glare đỏ và nút Export decisions CSV); ảnh tổng hợp `demo/top20_demo.png`.
- Cách override + audit log (`out/audit_log.jsonl`): người review lưu quyết định thành `out/poc/review_queue_reviewed.csv` (`reviewer`, `reviewer_decision` ∈ approve/edit/reject/escalate/skip, `override_value` = KEEP|RECAPTURE, `review_note`) rồi chạy lại `src.main`. Mọi quyết định đi vào `audit_log.jsonl`; override ghi `before`/`after`; `final_actions.csv` có `final_action`, `review_status`, `needs_escalation`; `ranked_review.csv` giữ nguyên dự đoán gốc để kiểm toán. Cơ chế được test tự động (`tests/test_t02.py::test_human_override_is_applied_and_audited`). **Bằng chứng review của người thật: chưa có** — nhóm điền: reviewer ____, số ảnh đã duyệt ____, số override ____.
- Decision policy + `reason_code` + owner: policy ở §01; kết quả hiện tại `HOLD` / `EVAL_PROXY_LABELS_SMALL_N` (`reports/decision.json`, `reports/decision_log.csv`). Owner: ____ (điền ở `configs/lab.yaml` → `owner`; hiện là placeholder "TODO").
- Khi nào HOLD/REWORK thay vì tự động release: HOLD khi GT chỉ là nhãn proxy hoặc eval < 100 ảnh, chưa kiểm 10 ảnh out-of-domain, hoặc owner chưa ký; REWORK khi PoC < baseline trên eval. Điều kiện để chuyển sang GO: chấm bằng GT chính thức với Recall@20 PoC ≥ baseline, kiểm 10 ảnh out-of-domain không báo nhầm hàng loạt, và fit lại ngưỡng khi dev có ≥ 5 ảnh sạch mỗi cặp light × camera.

## 08 · Reproduce & tests

​```bash
make setup          # hoặc: python -m venv .venv && pip install -r requirements.txt
python -m src.main --input fixtures/t02_raw_qc --config configs/lab.yaml --output out/ --offline --seed 42
python -m src.evaluate --track t02 --gt fixtures/t02_raw_qc/proxy_gt/labels.csv --pred out/ranked_review.csv --out reports/metrics_poc.json
pytest -q
# toàn bộ chuỗi (baseline + PoC + ablation + failure modes + decision): python -m src.pipeline all
​```

- Output của `pytest -q`: 62 passed, 39 skipped, 0 failed (39 test cần `gt/` của giảng viên nên bị skip). Nhóm chạy lại và dán output thật: ____
- Máy / CPU / OS / Python: ____ · git sha: ____

## 09 · Limitations & ethics

- Rủi ro pháp lý / PII / license: dữ liệu là synthetic (CC0-1.0), không có PII thật. Nếu chuyển sang ảnh thật (biển số, người) cần che/xoá PII và kiểm license trước khi dùng — nhóm chưa làm.
- Điều chưa biết; false alarms; nếu scale 10× thì sao: GT là nhãn proxy trên 40 ảnh eval (mỗi ảnh = 5% Recall; 2 ảnh chênh giữa PoC và baseline chưa có ý nghĩa thống kê); nhãn gán sau khi đã xem ảnh eval; 1 false alarm của PoC (`img_23_1`); ngưỡng compression có biên hẹp và ngưỡng bloom chỉ từ 1 ví dụ dev; dev không có night × cam_A; chưa kiểm out-of-domain (10 ảnh). Ở 10× dữ liệu thì runtime ≈ 8 ms/ảnh vẫn nhẹ trên CPU, nhưng rủi ro chính là phân phối camera/ánh sáng mới làm lệch tham chiếu.
- Không tuyên bố "đạt GDPR / luật Việt Nam" từ checklist tự động; giả định cần DPO/legal xác nhận: ____
- Kế hoạch cải tiến: chấm bằng GT chính thức; bootstrap theo cảnh; thêm ảnh sạch cho từng cặp light × camera và 10 ảnh out-of-domain (ảnh đêm thật là hard negative tốt); tách ngưỡng theo nhóm camera/ROI; thay ngưỡng bloom bằng nhiều ví dụ dev hơn.

## 10 · Team contribution

**Bảng 2 · Individual contribution evidence** (đối chiếu với `git shortlog -sne --all` và `reports/team_contributions.csv`)

| Học viên | Git author/email | Module / vai trò | Commit SHA / diff chứng minh | Trình bày? |
|---|---|---|---|---|
| Member 1 | damtranminhquang125-oss | Tất cả | | |


## 11 · References

| URL | Ngày xem | 1–2 dòng ý nghĩa (không chép nguyên văn) |
|---|---|---|

---

**Bảng 3 · Checklist chốt khi nộp**

- [ ] Link repo và tag cuối cùng trong README; input mẫu không chứa PII thật.
- [ ] Đúng 1 lệnh run, 1 lệnh evaluate, 1 lệnh tests; không cần download khi chạy.
- [ ] Có baseline/PoC metrics, ≥1 failure case, reviewer evidence, decision log.
- [ ] Có license/source provenance, data splits, seed, offline environment, GT/eval N.
- [ ] Mỗi thành viên có ≥1 commit thực chất và được nêu minh bạch trong bảng trên.
