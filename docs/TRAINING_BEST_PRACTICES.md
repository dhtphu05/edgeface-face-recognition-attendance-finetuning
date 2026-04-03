# Dynamic Face Dataset and Hybrid Training Guide

This document defines the recommended workflow for building a robust training dataset from offline videos and training the EdgeFace pipeline with the best practical accuracy/stability tradeoff for this repository.

It is written for the current workspace layout:

- `scripts/import_video_to_dataset.py`: video-to-dataset extraction
- `2_face_dataset/`: aligned training images
- `3_edgeface_training/scripts/train_phase3.py`: base training
- `3_edgeface_training/scripts/prune_phase4.py`: structured pruning
- `3_edgeface_training/scripts/finetune_phase5.py`: recovery finetuning
- `3_edgeface_training/scripts/evaluate_paper_metrics.py`: offline evaluation

## 1. Recommended Strategy

Use a hybrid workflow:

- Local machine:
  - record raw videos
  - extract aligned faces from video
  - inspect dataset quality
  - optionally keep a backup of raw videos and the final curated dataset
- Colab or remote GPU:
  - run Phase 3, Phase 4, and Phase 5 training
  - save checkpoints and logs to persistent storage
- Local machine again:
  - pull the best checkpoint back
  - run validation and deployment tests

This split gives the best operational result because video import is data-heavy and iterative, while training is compute-heavy.

## 2. Why Video-Based Extraction Beats Static Studio Images

The model will be deployed in uncontrolled conditions, so the training set must contain the same kinds of variation:

- motion blur from walking and head movement
- real door lighting and shadow transitions
- pose changes from camera approach angles
- partial occlusion from glasses, masks, hair, or hand movement
- mild sensor noise from phone cameras

Do not over-clean the extracted frames. If the face is still detectable and identifiable, keep it. That natural noise improves robustness.

## 3. Data Collection Best Practices

For each student, record multiple short videos instead of one perfectly clean video.

Recommended capture plan per identity:

- `3-5` videos per student
- `15-45` seconds per video
- mix distances: near, mid, far
- mix head pose: front, left/right yaw, slight pitch up/down
- mix lighting: indoor bright, side light, dim light, backlit doorway
- include walking toward the camera and slight motion

Avoid these failure cases:

- extreme blur where the face is not recognizable by a human
- large occlusion covering most of the face
- very dark scenes with no visible facial structure
- repeated static standing in a single pose for the whole video

## 4. Extraction Pipeline

The extraction script is implemented in [scripts/import_video_to_dataset.py](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/scripts/import_video_to_dataset.py).

It performs:

1. video decode with OpenCV
2. frame skipping for diversity
3. RetinaFace detection through InsightFace
4. largest-face selection per sampled frame
5. size filtering
6. 5-point landmark alignment with `face_align.norm_crop`
7. save to `dataset/{student_id}/face_XXXX.jpg`

### Recommended command

```bash
python3 scripts/import_video_to_dataset.py \
  "../raw_videos/video_phu.mp4" \
  "Phu_102250219" \
  "../2_face_dataset" \
  --frame-skip 3 \
  --min-face-size 60
```

### Recommended extraction settings

- `frame_skip=3`:
  - good default for normal phone video
  - reduces near-duplicate frames
- `frame_skip=5` to `8`:
  - use when the video is long or high FPS and results are too redundant
- `min_face_size=60`:
  - good default for training quality
- `min_face_size=80`:
  - use when far-distance faces add too much noise
- `min_face_size=50`:
  - acceptable when you specifically want more far-field robustness

### Target number of images per identity

Practical target:

- minimum acceptable: `80-120` aligned images
- better baseline: `150-300` aligned images
- strong coverage: `300-500` aligned images collected from multiple videos

More images only help if they add variation. Thousands of near-duplicate frames are less useful than a few hundred diverse samples.

## 5. Dataset Quality Control

After extraction, manually inspect each identity folder in `2_face_dataset/<student_id>`.

Remove:

- wrong person crops
- side faces so extreme that the identity is ambiguous
- severe detector failures
- blank or corrupted output images
- too many near-identical sequential frames

Keep:

- mild motion blur
- slight misalignment that still preserves the face
- mixed lighting
- natural expression changes

Best practice is to do one quick human review pass after every import batch.

## 6. Class Balance Rules

Training degrades if some students have far more samples than others.

Recommended balance:

- keep most identities within about `0.5x` to `2x` of the median sample count
- if one student has much more data, downsample that folder
- if one student has too little data, record another video instead of relying on heavy augmentation

Real diversity is more valuable than synthetic augmentation.

## 7. Local vs Hybrid vs Full Colab

### Train local if

- you have an NVIDIA GPU with stable CUDA
- you expect many tuning iterations
- your dataset is updated frequently

### Use hybrid if

- local extraction is convenient but local training is slow
- you want reproducible dataset curation locally
- you want remote GPU only for the expensive steps

### Use full Colab only if

- local hardware is too weak
- you accept session interruption and storage sync overhead

For this repository, hybrid is usually the best operational choice.

## 8. Hybrid Workflow

### Step 1: Extract locally

For each video:

```bash
cd /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace
python3 scripts/import_video_to_dataset.py \
  "../raw_videos/video_phu.mp4" \
  "Phu_102250219" \
  "../2_face_dataset" \
  --frame-skip 3 \
  --min-face-size 60
```

### Step 2: Verify dataset structure

Expected layout:

```text
2_face_dataset/
  Phu_102250219/
    face_0000.jpg
    face_0001.jpg
    ...
  Student_B/
    face_0000.jpg
    ...
```

### Step 3: Move dataset to Colab or remote GPU

Preferred options:

- zip `2_face_dataset/` and upload once
- sync to Google Drive
- store under a stable path and do not rename folders mid-project

### Step 4: Recreate the same environment remotely

Install the training dependencies from [3_edgeface_training/requirements.txt](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/requirements.txt), then add the extraction/runtime packages when needed:

```bash
pip install -r 3_edgeface_training/requirements.txt
pip install insightface onnxruntime opencv-python
```

### Step 5: Train phases remotely

Phase 3:

```bash
cd 3_edgeface_training
python scripts/train_phase3.py
```

Phase 4:

```bash
python scripts/prune_phase4.py
```

Phase 5:

```bash
python scripts/finetune_phase5.py
```

Phase 5 with explicit paths:

```bash
python scripts/finetune_phase5.py \
  --dataset-root ../2_face_dataset \
  --input checkpoints/phase4_pruned_model.pth \
  --output checkpoints/phase5_final_model.pth \
  --epochs 3 \
  --batch-size 16 \
  --lr 5e-4
```

### Step 6: Evaluate and archive

```bash
python scripts/evaluate_paper_metrics.py
```

Archive:

- final checkpoint
- best intermediate checkpoint
- exact dataset snapshot used for training
- notes on sample counts and training date

## 9. Training Best Practices

### Data first, hyperparameters second

If accuracy is poor, first suspect:

- low-quality dataset
- class imbalance
- identity contamination between folders
- too many duplicate frames
- insufficient lighting and pose variation

Only after that should you tune optimizer settings.

### Preserve train/validation integrity

Do not evaluate only on the same near-duplicate frames used for training. If possible:

- reserve some videos entirely for validation
- keep validation identities consistent across runs
- avoid splitting consecutive frames from one short segment across train and validation

Video-level separation is better than random frame-level separation.

### Keep checkpoints organized

For every serious run, record:

- dataset version
- frame skip
- min face size
- number of students
- images per student
- training start date
- checkpoint filename

This is necessary to compare experiments meaningfully.

### Avoid over-pruning too early

Phase 4 reduces model capacity. If Phase 3 accuracy is not yet stable, pruning too early compounds the problem. Ensure the base model is reasonable before pruning.

### Finetune immediately after pruning

Pruning damages representation quality. Always run Phase 5 right after Phase 4 and compare the pruned-finetuned model against the Phase 3 baseline.

## 10. Common Failure Patterns

### High train accuracy, poor real-world performance

Likely causes:

- dataset too clean
- not enough lighting variation
- no motion blur examples
- no distance variation

Fix:

- collect new videos in realistic conditions
- keep mild blur and difficult lighting samples

### Poor validation accuracy from the start

Likely causes:

- label mistakes
- bad face crops
- too few images per student
- class count too high relative to dataset size

Fix:

- inspect per-student folders manually
- remove failed crops
- add more videos for underrepresented identities

### Good local results, worse deployment results

Likely causes:

- camera domain mismatch
- deployment alignment pipeline differs from training alignment
- training images too centered and clean

Fix:

- collect videos from the same camera or installation angle
- keep the same `112x112` alignment convention end-to-end
- retrain using footage closer to deployment conditions

## 11. Minimum Reproducibility Checklist

Before calling a run "good", confirm:

- raw videos are backed up
- extracted dataset is versioned or archived
- training dependencies are recorded
- checkpoint names are not overwritten carelessly
- evaluation was run on a held-out set or held-out videos
- final model was tested on real camera conditions

## 12. Practical Recommendation for This Repository

For best real-world performance:

- use local offline video extraction
- curate the dataset manually after each import
- train on Colab or a remote GPU if your local GPU is weak
- keep validation videos separate from training videos
- preserve realistic noise instead of aggressively cleaning the dataset

If you only optimize one thing, optimize data diversity. For this pipeline, dataset quality has more impact than small optimizer changes.
