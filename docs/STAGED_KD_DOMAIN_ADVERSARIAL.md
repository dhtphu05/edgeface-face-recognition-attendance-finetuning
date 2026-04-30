# Staged KD + Domain-Adversarial Refinement

This document defines the new experimental branch:

1. `AdaFace IR101 -> Teacher Assistant (TA)`
2. `Teacher Assistant -> widened EdgeFace student`
3. Internal finetuning
4. Optional domain-adversarial refinement on internal data

The current control branch remains the AdaFace-only winner at `88.66%` pairwise accuracy.

## Model Roles

- Teacher: `3_edgeface_training/weights/AdaFace_IR101.pt`
- Teacher Assistant:
  - `width_preset=teacher_assistant`
  - `stage_channels=[96, 192, 384, 768]`
  - `rank_ratio=0.8`
- Final student:
  - `width_preset=widened`
  - `rank_ratio=0.7`

## A. IR101 -> TA Public KD

### A1. Build IR101 teacher centers for subset smoke

```bash
cd /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training
source .venv/bin/activate

python scripts/build_teacher_centers.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --teacher-backbone ir101_adaface \
  --teacher-weights weights/AdaFace_IR101.pt \
  --output checkpoints/ta_ir101_teacher_centers_smoke.pt \
  --batch-size 16 \
  --num-workers 0 \
  --max-classes 64 \
  --max-samples-per-class 50
```

### A2. TA smoke

```bash
python scripts/train_phase3.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --checkpoints-dir checkpoints \
  --epochs 1 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --width-preset teacher_assistant \
  --rank-ratio 0.8 \
  --kd-mode adadistill \
  --kd-alpha 0 \
  --teacher-backbone ir101_adaface \
  --teacher-centers-path checkpoints/ta_ir101_teacher_centers_smoke.pt \
  --adadistill-weight 0.5 \
  --teacher-logit-scale 64.0 \
  --skip-student-bootstrap \
  --skip-teacher-bootstrap \
  --max-classes 64 \
  --max-samples-per-class 50 \
  --max-train-batches-per-epoch 200 \
  --max-val-batches 50 \
  --output-prefix phase3_ta_public_smoke
```

### A3. TA mid

```bash
python scripts/build_teacher_centers.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --teacher-backbone ir101_adaface \
  --teacher-weights weights/AdaFace_IR101.pt \
  --output checkpoints/ta_ir101_teacher_centers_mid.pt \
  --batch-size 16 \
  --num-workers 0 \
  --max-classes 64 \
  --max-samples-per-class 100

python scripts/train_phase3.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --checkpoints-dir checkpoints \
  --epochs 5 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --width-preset teacher_assistant \
  --rank-ratio 0.8 \
  --kd-mode adadistill \
  --kd-alpha 0 \
  --teacher-backbone ir101_adaface \
  --teacher-centers-path checkpoints/ta_ir101_teacher_centers_mid.pt \
  --adadistill-weight 0.5 \
  --teacher-logit-scale 64.0 \
  --skip-student-bootstrap \
  --skip-teacher-bootstrap \
  --max-classes 64 \
  --max-samples-per-class 100 \
  --max-train-batches-per-epoch 500 \
  --max-val-batches 100 \
  --output-prefix phase3_ta_public_mid
```

### A4. TA e10

```bash
python scripts/build_teacher_centers.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --teacher-backbone ir101_adaface \
  --teacher-weights weights/AdaFace_IR101.pt \
  --output checkpoints/ta_ir101_teacher_centers_webface12m.pt \
  --batch-size 16 \
  --num-workers 0

python scripts/train_phase3.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --checkpoints-dir checkpoints \
  --epochs 10 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --width-preset teacher_assistant \
  --rank-ratio 0.8 \
  --kd-mode adadistill \
  --kd-alpha 0 \
  --teacher-backbone ir101_adaface \
  --teacher-centers-path checkpoints/ta_ir101_teacher_centers_webface12m.pt \
  --adadistill-weight 0.5 \
  --teacher-logit-scale 64.0 \
  --skip-student-bootstrap \
  --skip-teacher-bootstrap \
  --max-train-batches-per-epoch 1000 \
  --max-val-batches 200 \
  --output-prefix phase3_ta_public_e10
```

## B. TA -> Student Public KD

### B1. Build TA centers from the best TA public checkpoint

```bash
python scripts/build_teacher_centers.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --teacher-backbone edgeface_ta \
  --teacher-weights checkpoints/phase3_ta_public_e10_best.pth \
  --output checkpoints/ta_student_teacher_centers_webface12m.pt \
  --batch-size 16 \
  --num-workers 0
```

### B2. Student smoke

```bash
python scripts/build_teacher_centers.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --teacher-backbone edgeface_ta \
  --teacher-weights checkpoints/phase3_ta_public_e10_best.pth \
  --output checkpoints/ta_student_teacher_centers_smoke.pt \
  --batch-size 16 \
  --num-workers 0 \
  --max-classes 64 \
  --max-samples-per-class 50

python scripts/train_phase3.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --checkpoints-dir checkpoints \
  --epochs 1 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --width-preset widened \
  --rank-ratio 0.7 \
  --kd-mode adadistill \
  --kd-alpha 0 \
  --teacher-backbone edgeface_ta \
  --teacher-centers-path checkpoints/ta_student_teacher_centers_smoke.pt \
  --adadistill-weight 0.5 \
  --teacher-logit-scale 64.0 \
  --skip-student-bootstrap \
  --skip-teacher-bootstrap \
  --max-classes 64 \
  --max-samples-per-class 50 \
  --max-train-batches-per-epoch 200 \
  --max-val-batches 50 \
  --output-prefix phase3_stagedkd_public_smoke
```

### B3. Student mid

```bash
python scripts/build_teacher_centers.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --teacher-backbone edgeface_ta \
  --teacher-weights checkpoints/phase3_ta_public_e10_best.pth \
  --output checkpoints/ta_student_teacher_centers_mid.pt \
  --batch-size 16 \
  --num-workers 0 \
  --max-classes 64 \
  --max-samples-per-class 100

python scripts/train_phase3.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --checkpoints-dir checkpoints \
  --epochs 5 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --width-preset widened \
  --rank-ratio 0.7 \
  --kd-mode adadistill \
  --kd-alpha 0 \
  --teacher-backbone edgeface_ta \
  --teacher-centers-path checkpoints/ta_student_teacher_centers_mid.pt \
  --adadistill-weight 0.5 \
  --teacher-logit-scale 64.0 \
  --skip-student-bootstrap \
  --skip-teacher-bootstrap \
  --max-classes 64 \
  --max-samples-per-class 100 \
  --max-train-batches-per-epoch 500 \
  --max-val-batches 100 \
  --output-prefix phase3_stagedkd_public_mid
```

### B4. Student e10

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
  --kd-mode adadistill \
  --kd-alpha 0 \
  --teacher-backbone edgeface_ta \
  --teacher-centers-path checkpoints/ta_student_teacher_centers_webface12m.pt \
  --adadistill-weight 0.5 \
  --teacher-logit-scale 64.0 \
  --skip-student-bootstrap \
  --skip-teacher-bootstrap \
  --max-train-batches-per-epoch 1000 \
  --max-val-batches 200 \
  --output-prefix phase3_stagedkd_public_e10
```

## C. Internal Finetuning

### C1. Clean-core

```bash
python scripts/train_phase3.py \
  --dataset-root ../2_face_dataset_clean_core_split \
  --checkpoints-dir checkpoints \
  --student-weights checkpoints/phase3_stagedkd_public_e10_best.pth \
  --epochs 30 \
  --batch-size 32 \
  --num-workers 0 \
  --learning-rate 5e-5 \
  --width-preset widened \
  --rank-ratio 0.7 \
  --kd-mode none \
  --skip-teacher-bootstrap \
  --output-prefix phase3_stagedkd_to_clean_core
```

### C2. Full dataset

```bash
python scripts/train_phase3.py \
  --dataset-root ../2_face_dataset_split \
  --checkpoints-dir checkpoints \
  --student-weights checkpoints/phase3_stagedkd_to_clean_core_best.pth \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 0 \
  --learning-rate 5e-5 \
  --width-preset widened \
  --rank-ratio 0.7 \
  --kd-mode none \
  --skip-teacher-bootstrap \
  --output-prefix phase3_stagedkd_to_full_ft
```

### C3. Evaluation

```bash
python scripts/evaluate_paper_metrics.py \
  --checkpoint checkpoints/phase3_stagedkd_to_full_ft_best.pth \
  --dataset-root ../2_face_dataset_split \
  --num-workers 0 \
  --report-json checkpoints/phase3_stagedkd_to_full_ft_eval.json
```

## D. Optional Domain-Adversarial Refinement

Only run this if the staged-KD branch is close enough to the control branch.

```bash
python scripts/train_phase3.py \
  --dataset-root ../2_face_dataset_split \
  --checkpoints-dir checkpoints \
  --student-weights checkpoints/phase3_stagedkd_to_clean_core_best.pth \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 0 \
  --learning-rate 5e-5 \
  --width-preset widened \
  --rank-ratio 0.7 \
  --kd-mode none \
  --domain-adversarial \
  --domain-loss-weight 0.05 \
  --domain-label-source session \
  --skip-teacher-bootstrap \
  --output-prefix phase3_stagedkd_domainadv_full_ft
```

Then evaluate:

```bash
python scripts/evaluate_paper_metrics.py \
  --checkpoint checkpoints/phase3_stagedkd_domainadv_full_ft_best.pth \
  --dataset-root ../2_face_dataset_split \
  --num-workers 0 \
  --report-json checkpoints/phase3_stagedkd_domainadv_full_ft_eval.json
```

## Acceptance Rule

Promote the staged-KD branch only if final internal evaluation:

- beats `88.66%` pairwise accuracy, or
- matches it while clearly improving FRR at `FAR≈0.001`
