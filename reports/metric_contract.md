# Metric contract (khai báo bắt buộc — chốt ở phút 30, trước khi chạy eval)

> Các metric chuẩn đã được cài trong `src/metrics/tracks.py`. File này là nơi nhóm **cam kết** cách dùng chúng.
> Không đổi testset, threshold hay trọng số sau khi đã thấy kết quả eval.

| Mục | Khai báo của nhóm |
|---|---|
| Track | T02 · Raw QC Triage |
| Primary metric (tên đúng như evaluator in ra) | `Defect Recall@B [PRIMARY]` (kèm `Precision@B`; phụ: `False alarm in top-B` theo `light=day/dusk/night`, `Defect-type accuracy (GT defects)`) |
| Tăng tốt hay giảm tốt? | Recall@B, Precision@B, Type-acc: **higher**; False alarm in top-B: **lower** |
| Đơn vị đánh giá (image / region / pair / release / batch / case …) | image (1 dòng `image_id` trong `ranked_review.csv`) |
| Công thức + numerator / denominator | Recall@B = #ảnh lỗi thật trong top-B / #ảnh lỗi thật trên split eval (denominator = 20). Precision@B = #ảnh lỗi thật trong top-B / B. False alarm@B (mỗi slice `light`) = #ảnh sạch trong top-B / #ảnh sạch của slice. Type-acc = mean(`defect_type` == `pred_defect_type`) trên ảnh lỗi thật. TP = ảnh lỗi trong top-B; FP = ảnh sạch trong top-B; FN = ảnh lỗi ngoài top-B. |
| Cách ghép prediction ↔ GT (IoU? id? tolerance?) | Ghép đúng theo `image_id` (không tolerance). Chỉ giữ prediction có `image_id` thuộc split eval của GT; xếp `score` giảm dần, sắp xếp ổn định (`mergesort`, hoà thì giữ thứ tự file). |
| Candidate pool cố định | 40 ảnh split `eval` (10 cảnh gốc × 4 biến thể; 20 sạch + 20 lỗi theo nhãn proxy). Pipeline xếp hạng cả 96 ảnh, evaluator chỉ chấm trên eval; hàng đợi review cũng chỉ lấy từ ảnh ngoài split dev. |
| Review budget B / K | B = 20 (`configs/tracks/t02.yaml`: `params.budget`, `eval_params.budget`); K không dùng |
| Seed | 42 (pipeline xác định, không có thành phần ngẫu nhiên; seed chỉ để ghi nhận trong `run_info.json`) |
| Threshold policy (chọn trên dev như thế nào) | Chỉ dùng split `dev` (56 ảnh, 28 sạch): (1) ngưỡng mỗi loại lỗi = trung điểm (max ảnh sạch, min ảnh lỗi cùng loại) — nếu không tách được thì (p95 sạch, p10 lỗi); (2) tham chiếu độ nét = mô hình cộng `light + camera` fit trên ảnh sạch dev, mức sáng chuẩn theo `light`; (3) baseline = một ngưỡng Laplacian toàn ảnh chọn theo F1 trên dev. Đóng băng trong `configs/t02_params.json` (sinh bởi `python -m src.pipeline fit`). `score ≥ 1` ⇒ `RECAPTURE`. |
| Eval split (không dùng để tune) | `eval` (tách theo cảnh gốc `group_id`: dev 14 cảnh, eval 10 cảnh, không cảnh nào ở cả hai) |
| Chỉ số lỗi xấu (miss / false alarm / false reject) | Miss = ảnh lỗi ngoài top-B (FN). False alarm = ảnh sạch trong top-B (theo `light`). False reject = ảnh sạch bị `pred_action = RECAPTURE` (chỉ số phụ, theo camera: `reports/clean_flagged_by_slice.csv`). |
| Slices sẽ báo cáo (≥2) | `light` = day / dusk / night; loại lỗi = blur / compression / dark / glare; camera `cam_A` / `cam_B` (chỉ số phụ, mức `pred_action`) |
| Cách báo độ bất định khi N nhỏ (Wilson CI / bootstrap) | Wilson 95% cho Recall@B: baseline 17/20 → [0.64, 0.95]; PoC 19/20 → [0.76, 0.99] (hai khoảng chồng lấn). Lưu ý: 40 ảnh đến từ 10 cảnh nên không độc lập → khoảng thực tế còn rộng hơn; bootstrap theo cảnh **chưa chạy**. |

Nếu baseline = 0 thì ghi rõ "baseline = 0" thay vì phần trăm cải thiện. (Trong Bảng 1: `False alarm in top-B`, `light=day`: **baseline = 0**, PoC = 0.10.)

## Ghi chú trung thực (đọc trước khi dùng số liệu)

- **GT của các số liệu hiện có là nhãn proxy gán bằng mắt** (`fixtures/t02_raw_qc/proxy_gt/labels.csv`), không phải `gt/labels.csv` chính thức. Nhãn được gán sau khi xem toàn bộ 96 ảnh (kể cả eval) rồi mới thiết kế feature ⇒ có rủi ro rò rỉ gián tiếp dù ngưỡng chỉ fit trên dev. Khi có GT thật, chạy lại `python -m src.evaluate --track t02 --gt fixtures/t02_raw_qc/gt/labels.csv ...`.
- **PoC v2 (`poc_v2`, thêm feature bloom) là post-hoc**: thêm sau khi thấy lỗi của PoC v1 trên eval. Chỉ báo như ablation, **không** dùng làm kết quả chính; kết quả chính là PoC v1 (`poc`).
- Chính sách quyết định GO/HOLD/REWORK (`src/pipeline.py::decide_from_metrics`) được viết sau lần xem kết quả proxy đầu tiên.
