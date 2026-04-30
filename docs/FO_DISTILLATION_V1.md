# FO Distillation v1

Runbook này định nghĩa nhánh **Feature-Only Distillation (FO v1)** trên pipeline hiện tại.

Control branch giữ nguyên:

- public AdaFace-only
- clean-core finetune
- full-dataset finetune
- best overall hiện tại:
  - pairwise `88.66%`
  - FRR `67.71%` tại `FAR≈0.001`

FO v1 dùng trực tiếp codepath đã có trong `train_phase3.py`:

- `--kd-mode embedding_mse`
- teacher: `3_edgeface_training/weights/AdaFace_IR101.pt`
- student: `width_preset=widened`, `rank_ratio=0.7`

## 1. FO Smoke

```bash
cd /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training
source .venv/bin/activate

python scripts/train_phase3.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --checkpoints-dir checkpoints \
  --epochs 1 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --width-preset widened \
  --rank-ratio 0.7 \
  --kd-mode embedding_mse \
  --kd-alpha 25 \
  --teacher-backbone ir101_adaface \
  --teacher-weights weights/AdaFace_IR101.pt \
  --skip-student-bootstrap \
  --max-classes 64 \
  --max-samples-per-class 50 \
  --max-train-batches-per-epoch 200 \
  --max-val-batches 50 \
  --output-prefix phase3_fo_public_smoke_kd25
```

Smoke pass nếu:

- total loss và KD loss đều hữu hạn
- train/val accuracy không collapse về gần `0`
- không có lỗi bootstrap teacher

## 2. FO Mid Sweep

Chạy ba run độc lập với `kd_alpha ∈ {10, 25, 50}`:

```bash
python scripts/train_phase3.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --checkpoints-dir checkpoints \
  --epochs 5 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --width-preset widened \
  --rank-ratio 0.7 \
  --kd-mode embedding_mse \
  --kd-alpha 10 \
  --teacher-backbone ir101_adaface \
  --teacher-weights weights/AdaFace_IR101.pt \
  --skip-student-bootstrap \
  --max-classes 64 \
  --max-samples-per-class 100 \
  --max-train-batches-per-epoch 500 \
  --max-val-batches 100 \
  --output-prefix phase3_fo_public_mid_kd10
```

```bash
python scripts/train_phase3.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --checkpoints-dir checkpoints \
  --epochs 5 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --width-preset widened \
  --rank-ratio 0.7 \
  --kd-mode embedding_mse \
  --kd-alpha 25 \
  --teacher-backbone ir101_adaface \
  --teacher-weights weights/AdaFace_IR101.pt \
  --skip-student-bootstrap \
  --max-classes 64 \
  --max-samples-per-class 100 \
  --max-train-batches-per-epoch 500 \
  --max-val-batches 100 \
  --output-prefix phase3_fo_public_mid_kd25
```

```bash
python scripts/train_phase3.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --checkpoints-dir checkpoints \
  --epochs 5 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --width-preset widened \
  --rank-ratio 0.7 \
  --kd-mode embedding_mse \
  --kd-alpha 50 \
  --teacher-backbone ir101_adaface \
  --teacher-weights weights/AdaFace_IR101.pt \
  --skip-student-bootstrap \
  --max-classes 64 \
  --max-samples-per-class 100 \
  --max-train-batches-per-epoch 500 \
  --max-val-batches 100 \
  --output-prefix phase3_fo_public_mid_kd50
```

Chọn winner bằng rule chính thức:

1. best val accuracy
2. nếu chênh trong `0.2` điểm, chọn final train loss thấp hơn
3. nếu vẫn sát nhau, chọn `kd_alpha` nhỏ hơn

Helper script:

```bash
python scripts/select_public_winner.py \
  checkpoints/phase3_fo_public_mid_kd10_metrics.json \
  checkpoints/phase3_fo_public_mid_kd25_metrics.json \
  checkpoints/phase3_fo_public_mid_kd50_metrics.json
```

## 3. FO e10

Thay `KD_ALPHA_WINNER` bằng `10`, `25`, hoặc `50` theo kết quả `mid`.

```bash
python scripts/train_phase3.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --checkpoints-dir checkpoints \
  --epochs 10 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --width-preset widened \
  --rank-ratio 0.7 \
  --kd-mode embedding_mse \
  --kd-alpha KD_ALPHA_WINNER \
  --teacher-backbone ir101_adaface \
  --teacher-weights weights/AdaFace_IR101.pt \
  --skip-student-bootstrap \
  --max-train-batches-per-epoch 1000 \
  --max-val-batches 200 \
  --output-prefix phase3_fo_public_e10
```

Cho FO đi downstream đúng một lần nếu:

- best public `Val Acc >= 4.60%`, hoặc
- best public `Val Acc` không thấp hơn `phase3_stagedkd_public_e10` quá `0.15` điểm nhưng learning curve sạch hơn

## 4. Internal Downstream

### Clean-core

```bash
python scripts/train_phase3.py \
  --dataset-root ../2_face_dataset_clean_core_split \
  --checkpoints-dir checkpoints \
  --student-weights checkpoints/phase3_fo_public_e10_best.pth \
  --epochs 30 \
  --batch-size 32 \
  --num-workers 0 \
  --learning-rate 5e-5 \
  --width-preset widened \
  --rank-ratio 0.7 \
  --kd-mode none \
  --skip-teacher-bootstrap \
  --output-prefix phase3_fo_to_clean_core
```

### Full finetune

```bash
python scripts/train_phase3.py \
  --dataset-root ../2_face_dataset_split \
  --checkpoints-dir checkpoints \
  --student-weights checkpoints/phase3_fo_to_clean_core_best.pth \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 0 \
  --learning-rate 5e-5 \
  --width-preset widened \
  --rank-ratio 0.7 \
  --kd-mode none \
  --skip-teacher-bootstrap \
  --output-prefix phase3_fo_to_full_ft
```

### Evaluate

```bash
python scripts/evaluate_paper_metrics.py \
  --checkpoint checkpoints/phase3_fo_to_full_ft_best.pth \
  --dataset-root ../2_face_dataset_split \
  --num-workers 0 \
  --report-json checkpoints/phase3_fo_to_full_ft_eval.json
```

Promote FO branch only if:

- pairwise `> 88.66%`, hoặc
- pairwise trong khoảng `88.36% - 88.66%` nhưng FRR giảm ít nhất `5` điểm tuyệt đối

## 5. Failure and Fallback Rules

- Nếu FO public branch fail kỹ thuật hoặc `phase3_fo_public_e10` dưới `4.28%`, dừng direct FO.
- Nếu FO downstream thua baseline hơn `0.5` pairwise point, chốt negative result và không sang ReFO ngay.
- Nếu FO gần thắng, mới cân nhắc:
  - ReFO
  - staged FO (`IR101 -> TA -> student` với `embedding_mse`)
  - doorway adaptation khi có dữ liệu chưa gán nhãn
