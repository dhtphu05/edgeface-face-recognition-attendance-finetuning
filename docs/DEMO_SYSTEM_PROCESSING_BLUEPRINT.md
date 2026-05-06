# Demo System Processing Blueprint

## Purpose
This document describes the full processing flow of the current demo system so it can be rebuilt as a new system.

This version intentionally leaves the face detector / face aligner implementation abstract.

Reason:
- the current demo uses a temporary detector stack
- the new system will replace that part with another model

This document therefore focuses on:
- system flow
- module responsibilities
- required inputs and outputs
- data contracts
- recognition logic
- video / multi-person processing
- web integration behavior

## What the Demo System Does
The current demo system has two main modes:

1. single-image recognition
2. multi-person video recognition

Both share the same recognition core:
- face alignment
- embedding extraction
- gallery matching
- optional track-level aggregation

## Current Best Recognition Model
- checkpoint:
  - `/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/checkpoints/phase4_hybrid_v2_to_full_ft_best.pth`
- ONNX export:
  - `/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/onnx/phase4_hybrid_v2_to_full_ft_best.onnx`
- input:
  - aligned face image `112x112`
- output:
  - `embedding` (`512-d`)
  - `normalized_embedding` (`512-d`)
  - `norm`

## High-Level System Architecture

```text
Web UI / API
-> Setup
-> Build gallery
-> Create session
-> Run recognition
-> Show results

Recognition Core
-> Align face
-> Preprocess
-> Model inference
-> Embedding matching
-> Decision logic

Video Core
-> Video read
-> Face detection
-> Tracking
-> Tracklet aggregation
-> Track-level recognition
-> Save manifest / summaries / previews
```

## Required Module Split for the New System
The new system should be split into these modules.

### 1. Detector module
Responsibility:
- detect faces from a frame
- return `bbox`
- return `5 landmarks`

This module is intentionally left open for replacement.

Required output per detection:

```json
{
  "bbox": [x1, y1, x2, y2],
  "landmarks_5": [[x, y], [x, y], [x, y], [x, y], [x, y]],
  "det_score": 0.99
}
```

### 2. Tracker module
Responsibility:
- assign a stable `track_id` across frames
- maintain active/inactive tracks
- retire stale tracks

This is required for the multi-person attendance pipeline.

### 3. Alignment module
Responsibility:
- take:
  - frame
  - `bbox`
  - `5 landmarks`
- return aligned face crop

Output:
- `112x112` face image in BGR or RGB, consistently defined across the system

### 4. Recognition module
Responsibility:
- preprocess aligned face
- run recognition model
- output:
  - embedding
  - normalized embedding
  - norm

### 5. Gallery module
Responsibility:
- load student reference images
- build per-student prototype embedding
- save / load gallery embedding store

### 6. Matching module
Responsibility:
- compare query embedding with gallery prototypes
- compute cosine similarity
- return:
  - best student
  - similarity
  - status

### 7. Aggregation module
Responsibility:
- aggregate multiple frame embeddings for one track
- use quality-aware weighting by `norm`

### 8. Web/API module
Responsibility:
- user interaction
- running setup
- selecting checkpoint
- building gallery
- launching image / video recognition
- returning results to frontend

## Single-Image Processing Flow
This is the current logical flow for image recognition in the demo.

```text
Input image
-> detect face
-> align face to 112x112
-> preprocess
-> model inference
-> get embedding + norm
-> cosine similarity against gallery
-> best match / confidence / status
-> log attendance record
```

### Input
- image from:
  - upload
  - camera capture

### Step 1: detection
- one or more faces may be found
- for single-image mode, current logic expects one main face

### Step 2: alignment
- use `5 landmarks`
- output fixed `112x112`

### Step 3: preprocessing
Normalization must match training:

```python
img = img.astype("float32") / 255.0
img = (img - 0.5) / 0.5
img = np.transpose(img, (2, 0, 1))
```

### Step 4: recognition output
The model returns:
- `embedding`
- `norm`

### Step 5: matching
- compute cosine similarity with each gallery identity
- sort by similarity
- take the best candidate

### Step 6: decision
Typical fields:
- `recognized`
- `student_id`
- `similarity_score`
- `confidence`
- `top_matches`

## Multi-Person Video Processing Flow
This is the current logical flow for video recognition in the demo.

```text
Open video
-> sample frames
-> detect faces in each sampled frame
-> assign/update tracks
-> align each face
-> run recognition model
-> append embedding+norm to track
-> aggregate embeddings per track
-> match aggregated embedding against gallery
-> save results
```

## Video Processing Steps
### 1. Open video
- read frame-by-frame
- metadata:
  - total frames
  - frame index

### 2. Frame sampling
- process every `N`th frame
- current default:
  - `frame_skip = 3`

### 3. Detection
For each sampled frame:
- detect all faces
- discard detections smaller than threshold

Current practical threshold:
- `min_face_size = 50`

### 4. Tracking
Each detection must be assigned to a track.

Minimum required track state:
- `track_id`
- `first_frame`
- `last_seen_frame`
- `last_bbox`
- `total_detections`
- list of aligned face crops
- list of embeddings
- list of norms
- active/inactive status

### 5. Alignment
For each tracked detection:
- align to `112x112`

### 6. Recognition
For each aligned face:
- run recognizer
- get:
  - `embedding`
  - `norm`

### 7. Track accumulation
Append to track:
- face crop
- frame index
- detection score
- bbox size
- normalized recognition embedding
- norm

### 8. Track retirement
If a track is not seen for more than a frame gap:
- mark it inactive

Current practical value:
- `max_inactive_frames = 30`

### 9. Track finalization
After the video ends:
- for each track:
  - aggregate recent embeddings
  - compare with gallery
  - decide:
    - `auto_matched`
    - `needs_review`
    - `unknown`
- save aligned crops if the track is long enough

### 10. Output writing
The current demo writes:
- track image folders
- `manifest.json`
- `summary.txt`

## Tracklet Aggregation Logic
The demo uses norm-weighted aggregation.

For track embeddings `e_i` and norms `n_i`:

```python
w_i = max(n_i, eps)
agg = sum(w_i * normalize(e_i)) / sum(w_i)
agg = normalize(agg)
```

Recommended defaults:
- `aggregation_window = 8`
- `min_track_length = 5`

Tracks shorter than `5` valid frames should not produce final attendance decisions.

## Gallery Build Flow
The demo supports gallery building from student image folders.

### Gallery input
Expected folder structure:

```text
gallery_root/
  102230176/
    image_001.jpg
    image_002.jpg
  102230190/
    image_001.jpg
```

Current default gallery:
- `/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/2_face_dataset_clean_core_split/train`

### Gallery build steps
1. iterate over each student folder
2. load each image
3. align if needed, or treat as aligned crop if already canonical
4. run recognizer on each image
5. average the embeddings
6. L2-normalize the final student prototype

### Gallery output
Per student:

```json
{
  "student_id": "102230176",
  "prototype_embedding": [512 floats],
  "image_count": 100
}
```

## Matching Logic
### Similarity function
- cosine similarity

### Decision thresholds
Current practical defaults:
- `accept_threshold = 0.80`
- `review_margin = 0.03`

Decision:
- if best similarity >= threshold and margin over second-best is sufficient:
  - `auto_matched`
- otherwise:
  - `needs_review`

## Web Demo Flow
The current web demo has four important flows.

### A. Load available checkpoints
Backend scans:
- checkpoint directory

Frontend shows model list.

### B. Build gallery
Frontend sends:
- `checkpoint_path`
- `device`
- `gallery_dir`

Backend:
1. loads recognizer
2. loads gallery images
3. computes gallery embeddings
4. stores recognizer + gallery in memory

### C. Create attendance session
Frontend sends:
- `session_name`
- selected checkpoint

Backend:
- creates a session DB row

### D. Run recognition
#### For image mode
Frontend sends:
- base64 image
- `session_id`

Backend:
- decodes image
- detects and aligns face
- runs recognizer
- matches with gallery
- stores record

#### For video mode
Frontend sends:
- video path
- gallery path
- thresholds
- aggregation window
- checkpoint path

Backend:
- launches video processing pipeline
- writes outputs
- returns summary and track results

## Files Produced by the Video Demo
For each processed video:

```text
group_tracks/<video_stem>/
  manifest.json
  summary.txt
  tracks/
    track_001/
      face_0000.jpg
      face_0001.jpg
```

### manifest.json
Contains:
- source video
- frame counts
- detections kept
- tracks created
- tracks kept
- dropped tracks
- auto matched tracks
- per-track metadata

### summary.txt
Human-readable summary:
- counts
- kept tracks
- suggested IDs
- scores

## Required Contracts for the New System
### Detector -> Recognition pipeline

```json
{
  "frame_id": 123,
  "track_id": 7,
  "bbox": [x1, y1, x2, y2],
  "landmarks_5": [[x, y], [x, y], [x, y], [x, y], [x, y]],
  "det_score": 0.98
}
```

### Recognition -> Matching layer

```json
{
  "track_id": 7,
  "embedding": [512 floats],
  "norm": 18.7
}
```

### Matching -> Attendance layer

```json
{
  "track_id": 7,
  "student_id": "102230176",
  "similarity": 0.84,
  "status": "matched",
  "frames_used": 8
}
```

## What Must Be Replaced in the New System
The following part should be treated as a placeholder:
- face detector
- face landmark extractor
- current temporary face alignment source

The new system must plug in:
- a new face detector
- a new landmark provider
- optionally a new tracker

The recognition model itself does not need to change.

## What Must Stay Consistent
The following must stay identical to preserve recognition quality:
- aligned face size:
  - `112x112`
- normalization:
  - `(img / 255 - 0.5) / 0.5`
- embedding dimension:
  - `512`
- gallery matching by cosine similarity
- tracklet aggregation by norm-weighted averaging

## Recommended Rebuild Order
1. rebuild gallery module
2. rebuild single-image recognition path
3. define detector output contract
4. add tracking
5. add tracklet aggregation
6. add video result writing
7. add web / API integration

## Minimal Recognition-Only API
If the new system wants a clean recognizer service, expose:

### extract
```python
extract(aligned_face_112x112) -> {
    "embedding": np.ndarray,
    "norm": float
}
```

### match
```python
match(embedding, gallery) -> {
    "student_id": str | None,
    "similarity": float,
    "status": str
}
```

### aggregate
```python
aggregate(track_embeddings, track_norms) -> {
    "track_embedding": np.ndarray,
    "avg_norm": float
}
```

## Bottom Line
The current demo is not just:
- detect
- recognize

It is a full attendance-oriented pipeline:
- setup
- gallery build
- single-image recognition
- video processing
- tracking
- tracklet aggregation
- gallery match
- summary writing
- web presentation

For the new system, the detector stack can be replaced completely.

What must remain stable is:
- the recognizer input format
- the embedding pipeline
- the gallery protocol
- the track-level decision logic
