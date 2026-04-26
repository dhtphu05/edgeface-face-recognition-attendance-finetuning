# AdaDistill Faithful Branch

This branch keeps the validated AdaFace-only pipeline unchanged and adds a separate AdaDistill experiment path using the face-domain teacher checkpoint:

- `3_edgeface_training/weights/AdaFace_IR101.pt`

The student remains the widened EdgeFace model. Distillation happens on the public dataset stage through precomputed teacher class centers.

## New Components

- `3_edgeface_training/models/iresnet_adaface_teacher.py`
  - IR-style AdaFace teacher wrapper that matches the `AdaFace_IR101.pt` checkpoint layout.
- `3_edgeface_training/scripts/build_teacher_centers.py`
  - Builds one normalized teacher center per public identity.
- `3_edgeface_training/core_losses/adadistill_loss.py`
  - Adds the adaptive center-based KD term.
- `3_edgeface_training/scripts/train_phase3.py`
  - Now supports:
    - `--kd-mode none`
    - `--kd-mode embedding_mse`
    - `--kd-mode adadistill`
  - Also supports:
    - `--teacher-backbone`
    - `--teacher-centers-path`
    - `--adadistill-weight`
    - `--teacher-logit-scale`
    - `--max-classes`
    - `--max-samples-per-class`

## Stage A: Teacher-Center Smoke Build

```bash
cd /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training
source .venv/bin/activate

python scripts/build_teacher_centers.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --teacher-weights weights/AdaFace_IR101.pt \
  --output checkpoints/adadistill_teacher_centers_smoke.pt \
  --batch-size 16 \
  --num-workers 0 \
  --max-classes 64 \
  --max-samples-per-class 50
```

## Stage B: Public AdaDistill Smoke

```bash
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
  --teacher-backbone ir101_adaface \
  --teacher-centers-path checkpoints/adadistill_teacher_centers_smoke.pt \
  --adadistill-weight 0.5 \
  --teacher-logit-scale 64.0 \
  --skip-student-bootstrap \
  --skip-teacher-bootstrap \
  --max-classes 64 \
  --max-samples-per-class 50 \
  --max-train-batches-per-epoch 200 \
  --max-val-batches 50 \
  --output-prefix phase3_adadistill_public_smoke
```

## Stage C: Public AdaDistill Mid

First build a matching center cache for the chosen subset budget:

```bash
python scripts/build_teacher_centers.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --teacher-weights weights/AdaFace_IR101.pt \
  --output checkpoints/adadistill_teacher_centers_mid.pt \
  --batch-size 16 \
  --num-workers 0 \
  --max-classes 64 \
  --max-samples-per-class 100
```

Then run:

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
  --kd-mode adadistill \
  --kd-alpha 0 \
  --teacher-backbone ir101_adaface \
  --teacher-centers-path checkpoints/adadistill_teacher_centers_mid.pt \
  --adadistill-weight 0.5 \
  --teacher-logit-scale 64.0 \
  --skip-student-bootstrap \
  --skip-teacher-bootstrap \
  --max-classes 64 \
  --max-samples-per-class 100 \
  --max-train-batches-per-epoch 500 \
  --max-val-batches 100 \
  --output-prefix phase3_adadistill_public_mid
```

## Stage D: Public AdaDistill E10

Build the full-label-space teacher centers:

```bash
python scripts/build_teacher_centers.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --teacher-weights weights/AdaFace_IR101.pt \
  --output checkpoints/adadistill_teacher_centers_webface12m.pt \
  --batch-size 16 \
  --num-workers 0
```

Then train:

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
  --teacher-backbone ir101_adaface \
  --teacher-centers-path checkpoints/adadistill_teacher_centers_webface12m.pt \
  --adadistill-weight 0.5 \
  --teacher-logit-scale 64.0 \
  --skip-student-bootstrap \
  --skip-teacher-bootstrap \
  --max-train-batches-per-epoch 1000 \
  --max-val-batches 200 \
  --output-prefix phase3_adadistill_public_e10
```

## Downstream Internal Finetuning

### Clean-Core

```bash
python scripts/train_phase3.py \
  --dataset-root ../2_face_dataset_clean_core_split \
  --checkpoints-dir checkpoints \
  --student-weights checkpoints/phase3_adadistill_public_e10_best.pth \
  --epochs 30 \
  --batch-size 32 \
  --num-workers 0 \
  --learning-rate 5e-5 \
  --width-preset widened \
  --rank-ratio 0.7 \
  --kd-mode none \
  --skip-teacher-bootstrap \
  --output-prefix phase3_adadistill_to_clean_core
```

### Full Dataset

```bash
python scripts/train_phase3.py \
  --dataset-root ../2_face_dataset_split \
  --checkpoints-dir checkpoints \
  --student-weights checkpoints/phase3_adadistill_to_clean_core_best.pth \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 0 \
  --learning-rate 5e-5 \
  --width-preset widened \
  --rank-ratio 0.7 \
  --kd-mode none \
  --skip-teacher-bootstrap \
  --output-prefix phase3_adadistill_to_full_ft
```

### Internal Evaluation

```bash
python scripts/evaluate_paper_metrics.py \
  --checkpoint checkpoints/phase3_adadistill_to_full_ft_best.pth \
  --dataset-root ../2_face_dataset_split \
  --num-workers 0 \
  --report-json checkpoints/phase3_adadistill_to_full_ft_eval.json
```

## Promotion Rule

Promote the AdaDistill branch only if:

- pairwise accuracy exceeds `88.66%`
- FRR does not worsen materially at `FAR ≈ 0.001`
- the gain survives the full internal downstream path, not just public validation
