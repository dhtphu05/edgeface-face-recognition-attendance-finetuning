# Public Pretraining Strategy

This document defines the recommended training flow for this repository when a large public face dataset is available on Google Drive and the internal student dataset is still too small to produce strong verification performance by itself.

The best-fit strategy for the current codebase is:

1. pretrain the student backbone on a large public face dataset
2. finetune on the internal clean-core dataset
3. finetune again on the full internal dataset
4. evaluate only on the internal held-out split

This strategy is preferred over KD-first because the current pipeline already supports stable AdaFace-only training, while the KD path has not yet demonstrated better results than the clean baseline.

## 1. Why This Is the Best Fit

The current repository already has:

- a configurable student backbone in `3_edgeface_training/models/edgeface_xxs.py`
- a stable AdaFace training path in `3_edgeface_training/scripts/train_phase3.py`
- a held-out evaluation path in `3_edgeface_training/scripts/evaluate_paper_metrics.py`
- a working internal data curation workflow based on clean-core and full-dataset splits

The main problem is not just model capacity. The model still lacks strong face-domain pretraining.

Using a large public face dataset helps the student learn:

- global facial structure
- cross-pose consistency
- illumination robustness
- intra-class compactness for face embeddings

Then the internal finetuning stages adapt that general face representation to the specific student population and deployment conditions.

## 2. Recommended Training Flow

### Stage A: Public Face Pretraining on Colab

Use the large public dataset on Google Drive to train the student model itself.

Recommended setup:

- model: `widened`
- `rank_ratio=0.7`
- loss: `AdaFace`
- no KD
- no pruning
- use Colab GPU
- store checkpoints on Drive

Goal:

- produce a strong generic face embedding checkpoint
- do not optimize for the student identities yet

Important rule:

- do not merge public identities into `2_face_dataset`
- public data is for backbone/embedding pretraining, not for the final student label space

### Stage B: Internal Clean-Core Finetuning

Take the best public-pretrained checkpoint and finetune it on:

- `2_face_dataset_clean_core_split`

Use the cleanest internal images first because they stabilize identity separation and prevent the harder samples from dominating too early.

Recommended setup:

- lower learning rate than public pretraining
- still no KD
- still no pruning

Goal:

- transfer the generic face embedding into the student domain using the cleanest labels

### Stage C: Internal Full-Dataset Finetuning

Take the best clean-core checkpoint and finetune it on:

- `2_face_dataset_split`

This stage adds:

- motion blur
- more pose variation
- more uncontrolled lighting
- harder intra-class examples

Recommended setup:

- lower learning rate again
- shorter run than the clean-core phase
- still avoid pruning unless the final internal metric is already strong

Goal:

- improve robustness without destroying the identity structure learned in Stage B

### Stage D: Internal Evaluation

Always evaluate on the held-out internal split.

Primary metrics:

- pairwise accuracy
- FAR
- FRR at FAR=`1e-3`
- threshold
- params
- FLOPs
- latency/FPS

Only these internal held-out results should be used to judge whether the pipeline is improving for the actual attendance problem.

## 3. What Not to Do

Do not:

- mix public identities directly into `2_face_dataset/<student_id>`
- use the public dataset as if it were the final deployment task
- prune before a strong internal finetuned baseline exists
- prefer KD over public pretraining at this stage

KD may still be explored later, but it is not the highest-ROI path right now.

## 4. Recommended Colab Workflow

### Public pretraining

Use Drive-mounted public data for Stage A.

Store:

- checkpoints
- metrics JSON
- final chosen public-pretrain checkpoint

Suggested output naming:

- `phase3_public_pretrain_best.pth`

### Pull checkpoint back into this repository

After Stage A, copy the best public-pretrain checkpoint into:

- `3_edgeface_training/checkpoints/`

Then use it as the student bootstrap for internal finetuning.

## 5. Recommended Internal Finetuning Flow

### Clean-core finetune

Example command:

```bash
cd /Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training
source .venv/bin/activate

python scripts/train_phase3.py \
  --dataset-root ../2_face_dataset_clean_core_split \
  --width-preset widened \
  --rank-ratio 0.7 \
  --kd-alpha 0 \
  --skip-teacher-bootstrap \
  --student-weights checkpoints/phase3_public_pretrain_best.pth \
  --epochs 40 \
  --learning-rate 5e-5 \
  --num-workers 0 \
  --output-prefix phase3_public_to_clean_core
```

### Full-dataset finetune

Example command:

```bash
python scripts/train_phase3.py \
  --dataset-root ../2_face_dataset_split \
  --width-preset widened \
  --rank-ratio 0.7 \
  --kd-alpha 0 \
  --skip-teacher-bootstrap \
  --student-weights checkpoints/phase3_public_to_clean_core_best.pth \
  --epochs 10 \
  --learning-rate 5e-5 \
  --num-workers 0 \
  --output-prefix phase3_cleancore_to_full_ft
```

### Final evaluation

```bash
python scripts/evaluate_paper_metrics.py \
  --checkpoint checkpoints/phase3_cleancore_to_full_ft_best.pth \
  --dataset-root ../2_face_dataset_split \
  --num-workers 0 \
  --report-json checkpoints/phase3_cleancore_to_full_ft_eval.json
```

## 6. Decision Rule for This Project

The best current strategy is:

- public face-domain pretraining first
- clean-core internal finetuning second
- full internal finetuning third

This is the most compatible path with the current repository, the most stable path relative to the training code already working, and the path most likely to improve verification performance without introducing new KD instability.
