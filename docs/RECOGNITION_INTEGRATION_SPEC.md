# Recognition Integration Spec

## Purpose
This document defines the integration contract for plugging the current face recognition module into the larger attendance system.

Scope:
- your system: video/frame ingestion, face detection, tracking, orchestration
- my module: face alignment, embedding extraction, gallery matching, track-level recognition decision

This spec is written for the current best model and must be treated as the source of truth for integration.

## Current Best Model
- Checkpoint:
  - `/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/checkpoints/phase4_hybrid_v2_to_full_ft_best.pth`
- Backbone:
  - `edgeface_hybrid_kprpe`
- Architecture params:
  - `attention_heads = 4`
  - `attention_depth = 1`
  - `kprpe_hidden_dim = 32`
- Input size:
  - `112x112`
- Embedding dimension:
  - `512`

## Verified Offline Metrics
- Pairwise Accuracy:
  - `94.27%`
- FRR at `FAR≈0.001`:
  - `33.97%`
- Params:
  - `5.52M`
- FLOPs:
  - `839.87M`

## Hard Requirement
Tracking is required.

Reason:
- frame-level recognition alone is not stable enough for doorway attendance
- the same student may be blurred, partially occluded, or off-angle in several frames
- final attendance decisions must be made at the track level, not the single-frame level

Minimal acceptable pipeline:

```text
Video/Camera
-> Face Detection
-> Face Tracking
-> Face Crop + 5 Landmarks per tracked detection
-> Recognition Model
-> Tracklet Aggregation
-> Gallery Matching
-> Attendance Decision
```

If the current system only has detection and no tracking, integration should stop at debugging or temporary frame-level testing. It should not be treated as final attendance logic.

## Responsibilities Split
### Upstream system responsibilities
- read video or camera frames
- detect all faces in each frame
- maintain a stable `track_id` for each person across frames
- output per detection:
  - `frame_id`
  - `track_id`
  - `bbox`
  - `landmarks_5`
  - either:
    - full frame, or
    - face crop plus the original landmarks in crop coordinates

### Recognition module responsibilities
- align face to `112x112`
- normalize input using the model training convention
- extract:
  - `embedding`
  - `norm`
- aggregate embeddings per track
- match aggregated track embedding against the student gallery
- return a recognition decision for each track

## Required Input Contract
Each detected face must contain:

```json
{
  "frame_id": 123,
  "track_id": 7,
  "bbox": [x1, y1, x2, y2],
  "landmarks_5": [
    [x_left_eye, y_left_eye],
    [x_right_eye, y_right_eye],
    [x_nose, y_nose],
    [x_left_mouth, y_left_mouth],
    [x_right_mouth, y_right_mouth]
  ]
}
```

Preferred upstream payload per frame:

```json
{
  "frame_id": 123,
  "timestamp_ms": 4567,
  "faces": [
    {
      "track_id": 7,
      "bbox": [100, 80, 180, 170],
      "landmarks_5": [[120, 110], [155, 112], [138, 130], [124, 148], [150, 149]]
    }
  ]
}
```

## Recognition Preprocessing
### Alignment
- align every face to `112x112`
- use the provided `5` landmarks
- do not skip alignment

### Normalization
Use the exact training normalization:

```python
img = img.astype("float32") / 255.0
img = (img - 0.5) / 0.5
```

Do not use ImageNet normalization for this model.

## Recognition Output Contract
For each track, the recognition module should return:

```json
{
  "track_id": 7,
  "student_id": "102230176",
  "similarity": 0.84,
  "norm": 18.7,
  "frames_used": 8,
  "status": "matched"
}
```

Allowed `status` values:
- `matched`
- `review`
- `unknown`

## Gallery Requirements
The recognition side requires a gallery embedding store.

Minimum content per student:
- `student_id`
- `prototype_embedding`
- optional:
  - `image_count`
  - `source_images`
  - `updated_at`

Current recommended source gallery:
- `/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/2_face_dataset_clean_core_split/train`

Recommended build rule:
- extract embeddings from all gallery images of a student
- average them
- L2-normalize the final prototype

## Tracklet Aggregation
Track-level recognition must use norm-weighted aggregation.

For a track with embeddings `e_i` and norms `n_i`:

```python
w_i = max(n_i, eps)
agg = sum(w_i * normalize(e_i)) / sum(w_i)
agg = normalize(agg)
```

Recommended defaults:
- `aggregation_window = 8`
- `min_track_length = 5`

Tracks shorter than `5` valid frames should not trigger final attendance decisions.

## Matching Rule
Similarity metric:
- cosine similarity

Recommended thresholds for the current model:
- `accept_threshold = 0.80`
- `review_margin = 0.03`

Decision rule:
- if `similarity >= 0.80`:
  - `matched`
- if `0.77 <= similarity < 0.80`:
  - `review`
- if `< 0.77`:
  - `unknown`

These values are starting defaults. They should be tuned later on a held-out validation split, not directly on the final test set.

## Minimal Runtime API
The following three functions are sufficient to integrate the recognizer cleanly:

### 1. Extract
```python
extract_embedding(frame_bgr, landmarks_5) -> {
    "embedding": np.ndarray,   # 512-d, L2-normalized
    "norm": float
}
```

### 2. Aggregate
```python
aggregate_track(track_embeddings, track_norms) -> np.ndarray
```

### 3. Match
```python
match_to_gallery(aggregated_embedding, gallery) -> {
    "student_id": str | None,
    "similarity": float,
    "status": str
}
```

## Required Integration Behavior
### Must do
- keep a stable `track_id`
- accumulate embeddings per track
- delay final attendance recognition until enough frames are collected
- make recognition decisions at the track level

### Must not do
- do not finalize attendance from a single frame
- do not skip face alignment
- do not use ImageNet normalization
- do not mix another recognizer's embedding space with this gallery

## Recommended Processing Order
1. Detect faces in each frame.
2. Assign or update `track_id`.
3. For each tracked face:
   - align with `5` landmarks
   - run recognition model
   - append `embedding` and `norm` to the track state
4. When a track reaches `min_track_length`:
   - aggregate the most recent `aggregation_window` embeddings
   - match to the gallery
   - assign `matched`, `review`, or `unknown`
5. Emit attendance events only after the track-level decision is stable.

## Temporary Fallback Mode
If tracking is not implemented yet, the system may run a temporary debug mode:
- per-frame detection
- per-frame recognition
- no final attendance event

This mode is only acceptable for debugging, not for production attendance.

## Reference Implementation Locations
- recognition engine:
  - `/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/1_django_collection/core_app/face_recognition_engine.py`
- video test script:
  - `/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/scripts/import_group_video_to_tracks.py`
- trained checkpoint:
  - `/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/checkpoints/phase4_hybrid_v2_to_full_ft_best.pth`

## Integration Acceptance Checklist
- detector returns `bbox + 5 landmarks + track_id`
- recognizer loads the winner checkpoint correctly
- gallery is built from the clean-core student set
- embeddings are aggregated per track
- final decisions are emitted only at track level
- output includes:
  - `track_id`
  - `student_id`
  - `similarity`
  - `status`
  - `frames_used`

## Final Note
The recognition module is ready.

The remaining system-side requirement is not a better detector. It is stable tracking and correct handoff of:
- `track_id`
- `bbox`
- `5 landmarks`

Without tracking, the system is only a face recognizer.  
With tracking and tracklet aggregation, it becomes an attendance pipeline.
