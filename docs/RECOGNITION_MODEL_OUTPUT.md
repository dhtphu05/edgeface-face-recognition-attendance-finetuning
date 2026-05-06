# Recognition Model Output

## Purpose
This document defines the actual output of the current face recognition model and how that output must be used inside the attendance system.

This is important because the model does **not** directly output a student ID.

## Current Best Model
- Checkpoint:
  - `/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/checkpoints/phase4_hybrid_v2_to_full_ft_best.pth`
- Backbone:
  - `edgeface_hybrid_kprpe`
- Input:
  - aligned face image, size `112x112`

## What the Model Outputs
The recognition model outputs two things:

1. `embedding`
2. `norm`

### 1. Embedding
- type:
  - floating-point vector
- dimension:
  - `512`
- role:
  - this is the identity representation used for matching

In the recognition pipeline, the embedding is L2-normalized before matching.

### 2. Norm
- type:
  - scalar float
- role:
  - quality-related signal
  - used as a weight during tracklet aggregation

The norm is **not** the predicted identity.

## Important Clarification
The model does **not** output:
- `student_id`
- `class_name`
- final attendance decision

Those are produced later by:
- gallery matching
- thresholding
- track-level aggregation logic

## Recognition Pipeline
The actual pipeline is:

```text
Aligned Face (112x112)
-> Recognition Model
-> 512-d Embedding + Norm
-> L2 Normalization
-> Cosine Similarity vs Gallery Embeddings
-> Student ID / Match Status
```

## Expected Runtime Output
At runtime, the useful result after model forward is:

```json
{
  "embedding": [float, float, "..."],
  "norm": 18.7
}
```

The embedding should then be matched against a gallery store:

```json
{
  "student_id": "102230176",
  "similarity": 0.84,
  "status": "matched"
}
```

## How the Embedding Is Used
### Gallery matching
- each student in the gallery has a prototype embedding
- cosine similarity is computed between:
  - query embedding
  - gallery prototype embedding

The student with the highest valid similarity is the candidate match.

### Tracklet aggregation
If the system has tracking, multiple frame-level embeddings from the same `track_id` are aggregated.

Recommended rule:

```python
w_i = max(norm_i, eps)
agg = sum(w_i * normalize(e_i)) / sum(w_i)
agg = normalize(agg)
```

Where:
- `e_i` = frame embedding
- `norm_i` = frame norm
- `agg` = final track embedding

## Why Norm Matters
The `norm` helps weight better frames more strongly.

Typical use:
- clearer frame -> usually higher norm -> larger contribution
- blurrier or lower-quality frame -> usually lower norm -> smaller contribution

This is especially useful for:
- doorway video
- non-frontal faces
- short motion blur events

## Required Input to Produce Valid Output
To get a valid embedding, the face must be:
- correctly detected
- aligned using `5` landmarks
- resized to `112x112`
- normalized using the training convention

Normalization rule:

```python
img = img.astype("float32") / 255.0
img = (img - 0.5) / 0.5
```

Do not use ImageNet normalization for this model.

## Integration Rule
If another subsystem is integrating this recognizer, it should assume:

### Input
```json
{
  "track_id": 7,
  "bbox": [x1, y1, x2, y2],
  "landmarks_5": [[x, y], [x, y], [x, y], [x, y], [x, y]]
}
```

### Model-side output
```json
{
  "embedding": [512 floats],
  "norm": 18.7
}
```

### Final recognition output
```json
{
  "track_id": 7,
  "student_id": "102230176",
  "similarity": 0.84,
  "status": "matched"
}
```

## What Should Be Stored
The system should store:
- gallery embeddings per student
- optional frame embeddings for debugging
- track-level aggregated embeddings if needed for audit

The system does not need to store raw model logits, because identity is resolved through embedding similarity, not softmax classification at inference time.

## Bottom Line
The recognition model is an **embedding extractor**, not a classifier API.

Its main output is:
- a `512`-dimensional identity embedding

Its auxiliary output is:
- a `norm` value used for quality-aware aggregation

Final identity is obtained only after:
- gallery comparison
- similarity thresholding
- optionally tracklet aggregation
