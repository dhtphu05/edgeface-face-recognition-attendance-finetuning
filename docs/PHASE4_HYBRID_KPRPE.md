# Phase 4 Hybrid KP-RPE Branch

This runbook defines the new Phase 4 branch:

1. `edgeface_hybrid_kprpe` backbone
2. Surveillance-tuned AdaFace with explicit `h`
3. Partial FC for public warmup
4. Mixed-domain adaptation

## Stage A: Clean-core bring-up

```bash
cd /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training
source .venv/bin/activate

python scripts/train_phase3.py \
  --dataset-root ../2_face_dataset_clean_core_split \
  --backbone edgeface_hybrid_kprpe \
  --width-preset widened \
  --rank-ratio 0.7 \
  --attention-heads 4 \
  --attention-depth 1 \
  --kprpe-hidden-dim 32 \
  --kd-mode none \
  --classifier-mode full \
  --adaface-margin 0.4 \
  --adaface-scale 64.0 \
  --adaface-h 0.30 \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 0 \
  --learning-rate 5e-5 \
  --motion-blur-prob 0.15 \
  --motion-blur-kernel-size 7 \
  --skip-student-bootstrap \
  --output-prefix phase4_hybrid_clean_core_bringup
```

## Stage B: Public warmup with Partial FC

```bash
python scripts/train_phase3.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/public_face_dataset/Face_Recognition/train \
  --backbone edgeface_hybrid_kprpe \
  --width-preset widened \
  --rank-ratio 0.7 \
  --attention-heads 4 \
  --attention-depth 1 \
  --kprpe-hidden-dim 32 \
  --kd-mode none \
  --classifier-mode partial_fc \
  --partial-fc-sample-rate 0.10 \
  --partial-fc-min-negatives 2048 \
  --partial-fc-seed 1234 \
  --adaface-margin 0.4 \
  --adaface-scale 64.0 \
  --adaface-h 0.30 \
  --epochs 10 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --motion-blur-prob 0.20 \
  --motion-blur-kernel-size 9 \
  --max-train-batches-per-epoch 1000 \
  --max-val-batches 200 \
  --skip-student-bootstrap \
  --output-prefix phase4_hybrid_public_warmup
```

## Norm analysis

```bash
python scripts/analyze_feature_norms.py \
  --checkpoint checkpoints/phase4_hybrid_public_warmup_best.pth \
  --dataset-root ../2_face_dataset_clean_core_split ../2_face_dataset_split \
  --report-json checkpoints/phase4_hybrid_norm_report.json
```

If doorway-like/internal norms are visibly compressed, rerun the next stage with `--adaface-h 0.28`.

## Mixed-domain adaptation

Prepare a merged dataset when external surveillance sets are available:

```bash
python scripts/prepare_mixed_domain_dataset.py \
  --source internal=/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/2_face_dataset_split \
  --source cox=/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/external_datasets/cox_s2v \
  --source chokepoint=/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/external_datasets/chokepoint \
  --output-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/mixed_surveillance_dataset \
  --copy-mode symlink \
  --clear-output
```

Then train:

```bash
python scripts/train_phase3.py \
  --dataset-root /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/mixed_surveillance_dataset \
  --backbone edgeface_hybrid_kprpe \
  --width-preset widened \
  --rank-ratio 0.7 \
  --attention-heads 4 \
  --attention-depth 1 \
  --kprpe-hidden-dim 32 \
  --kd-mode none \
  --classifier-mode full \
  --adaface-margin 0.4 \
  --adaface-scale 64.0 \
  --adaface-h 0.30 \
  --epochs 15 \
  --batch-size 16 \
  --num-workers 0 \
  --learning-rate 5e-5 \
  --motion-blur-prob 0.30 \
  --motion-blur-kernel-size 9 \
  --student-weights checkpoints/phase4_hybrid_public_warmup_best.pth \
  --skip-teacher-bootstrap \
  --output-prefix phase4_hybrid_surveillance_adapt
```
