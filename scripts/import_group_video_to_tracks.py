from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align


IMAGE_SIZE = 112
VERY_STRONG_EMBEDDING_DISTANCE = 0.25
EMBEDDING_EMA_MOMENTUM = 0.8
DEFAULT_AUTO_MATCH_THRESHOLD = 0.85
DEFAULT_REVIEW_MARGIN = 0.03
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Detection:
    bbox: np.ndarray
    kps: np.ndarray
    embedding: np.ndarray
    det_score: float
    frame_bgr: np.ndarray
    frame_idx: int

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def min_side(self) -> float:
        return min(self.width, self.height)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class GalleryIdentity:
    student_id: str
    prototype_embedding: np.ndarray
    image_count: int


@dataclass
class Track:
    track_id: int
    prototype_embedding: np.ndarray
    last_bbox: np.ndarray
    last_seen_frame: int
    first_frame: int
    total_detections: int = 0
    pending_faces: list[np.ndarray] = field(default_factory=list)
    pending_frame_indices: list[int] = field(default_factory=list)
    pending_scores: list[float] = field(default_factory=list)
    pending_bbox_sizes: list[float] = field(default_factory=list)
    saved_image_count: int = 0
    active: bool = True
    status: str = "active"

    def update(self, detection: Detection) -> None:
        detection_embedding = normalize_embedding(detection.embedding)
        self.prototype_embedding = normalize_embedding(
            (EMBEDDING_EMA_MOMENTUM * self.prototype_embedding)
            + ((1.0 - EMBEDDING_EMA_MOMENTUM) * detection_embedding)
        )
        self.last_bbox = detection.bbox.astype(np.float32)
        self.last_seen_frame = detection.frame_idx
        self.total_detections += 1
        self.pending_scores.append(float(detection.det_score))
        self.pending_bbox_sizes.append(float(detection.min_side))
        self.pending_frame_indices.append(detection.frame_idx)
        aligned_face = face_align.norm_crop(
            detection.frame_bgr,
            landmark=detection.kps,
            image_size=IMAGE_SIZE,
        )
        self.pending_faces.append(aligned_face)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm <= 0:
        return embedding
    return embedding / norm


def cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_norm = normalize_embedding(lhs)
    rhs_norm = normalize_embedding(rhs)
    return float(np.clip(np.dot(lhs_norm, rhs_norm), -1.0, 1.0))


def cosine_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return 1.0 - cosine_similarity(lhs, rhs)


def compute_iou(lhs_bbox: np.ndarray, rhs_bbox: np.ndarray) -> float:
    x1 = max(float(lhs_bbox[0]), float(rhs_bbox[0]))
    y1 = max(float(lhs_bbox[1]), float(rhs_bbox[1]))
    x2 = min(float(lhs_bbox[2]), float(rhs_bbox[2]))
    y2 = min(float(lhs_bbox[3]), float(rhs_bbox[3]))
    intersection_w = max(0.0, x2 - x1)
    intersection_h = max(0.0, y2 - y1)
    intersection = intersection_w * intersection_h
    if intersection <= 0:
        return 0.0

    lhs_area = max(0.0, float(lhs_bbox[2] - lhs_bbox[0])) * max(0.0, float(lhs_bbox[3] - lhs_bbox[1]))
    rhs_area = max(0.0, float(rhs_bbox[2] - rhs_bbox[0])) * max(0.0, float(rhs_bbox[3] - rhs_bbox[1]))
    union = lhs_area + rhs_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _build_face_app() -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def _validate_inputs(
    video_path: Path,
    output_base_dir: Path,
    frame_skip: int,
    min_face_size: int,
    max_tracks: int | None,
    max_inactive_frames: int,
    min_track_length: int,
    embedding_match_threshold: float,
    iou_match_threshold: float,
    gallery_dir: Path | None,
    auto_match_threshold: float,
    review_margin: float,
) -> None:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if frame_skip <= 0:
        raise ValueError("frame_skip must be greater than 0.")
    if min_face_size <= 0:
        raise ValueError("min_face_size must be greater than 0.")
    if max_tracks is not None and max_tracks <= 0:
        raise ValueError("max_tracks must be greater than 0 when provided.")
    if max_inactive_frames <= 0:
        raise ValueError("max_inactive_frames must be greater than 0.")
    if min_track_length <= 0:
        raise ValueError("min_track_length must be greater than 0.")
    if embedding_match_threshold <= 0:
        raise ValueError("embedding_match_threshold must be greater than 0.")
    if iou_match_threshold < 0:
        raise ValueError("iou_match_threshold must be >= 0.")
    if gallery_dir is not None and not gallery_dir.is_dir():
        raise FileNotFoundError(f"Gallery directory not found: {gallery_dir}")
    if not (0.0 <= auto_match_threshold <= 1.0):
        raise ValueError("auto_match_threshold must be between 0 and 1.")
    if review_margin < 0:
        raise ValueError("review_margin must be >= 0.")
    output_base_dir.mkdir(parents=True, exist_ok=True)


def retire_inactive_tracks(
    tracks: list[Track],
    frame_idx: int,
    max_inactive_frames: int,
) -> int:
    retired_count = 0
    for track in tracks:
        if track.active and (frame_idx - track.last_seen_frame) > max_inactive_frames:
            track.active = False
            track.status = "inactive"
            retired_count += 1
    return retired_count


def build_detections(
    app: FaceAnalysis,
    frame_bgr: np.ndarray,
    frame_idx: int,
    min_face_size: int,
) -> list[Detection]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    faces = app.get(frame_rgb)
    detections: list[Detection] = []

    for face in faces:
        bbox = getattr(face, "bbox", None)
        kps = getattr(face, "kps", None)
        embedding = getattr(face, "embedding", None)
        det_score = float(getattr(face, "det_score", 0.0))
        if bbox is None or kps is None or embedding is None:
            continue
        detection = Detection(
            bbox=np.asarray(bbox, dtype=np.float32),
            kps=np.asarray(kps, dtype=np.float32),
            embedding=np.asarray(embedding, dtype=np.float32),
            det_score=det_score,
            frame_bgr=frame_bgr,
            frame_idx=frame_idx,
        )
        if detection.min_side < min_face_size:
            continue
        detections.append(detection)

    return detections


def choose_track_for_detection(
    detection: Detection,
    tracks: list[Track],
    embedding_match_threshold: float,
    iou_match_threshold: float,
    used_track_ids: set[int],
) -> Track | None:
    candidates: list[tuple[float, float, Track]] = []
    for track in tracks:
        if not track.active or track.track_id in used_track_ids:
            continue
        embedding_distance = cosine_distance(detection.embedding, track.prototype_embedding)
        bbox_iou = compute_iou(detection.bbox, track.last_bbox)
        strong_embedding = embedding_distance <= VERY_STRONG_EMBEDDING_DISTANCE
        if embedding_distance <= embedding_match_threshold and (
            bbox_iou >= iou_match_threshold or strong_embedding
        ):
            candidates.append((embedding_distance, -bbox_iou, track))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def list_gallery_images(gallery_dir: Path) -> list[Path]:
    images: list[Path] = []
    for class_dir in sorted(path for path in gallery_dir.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(image_path)
    return images


def build_gallery(app: FaceAnalysis, gallery_dir: Path) -> list[GalleryIdentity]:
    identities: list[GalleryIdentity] = []
    print(f"Building gallery from {gallery_dir} ...")
    for class_dir in sorted(path for path in gallery_dir.iterdir() if path.is_dir()):
        embeddings: list[np.ndarray] = []
        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue
            detections = build_detections(app=app, frame_bgr=image_bgr, frame_idx=0, min_face_size=1)
            if not detections:
                continue
            largest = max(detections, key=lambda detection: detection.area)
            embeddings.append(normalize_embedding(largest.embedding))

        if not embeddings:
            print(f"⚠️ No usable gallery embeddings for {class_dir.name}")
            continue
        prototype = normalize_embedding(np.mean(np.stack(embeddings, axis=0), axis=0))
        identities.append(
            GalleryIdentity(
                student_id=class_dir.name,
                prototype_embedding=prototype,
                image_count=len(embeddings),
            )
        )
    print(f"Built gallery identities={len(identities)}")
    return identities


def suggest_gallery_match(
    track_embedding: np.ndarray,
    gallery: list[GalleryIdentity],
    auto_match_threshold: float,
    review_margin: float,
) -> dict[str, object]:
    if not gallery:
        return {
            "suggested_student_id": None,
            "similarity_score": None,
            "second_best_student_id": None,
            "second_best_similarity": None,
            "review_status": "no_gallery",
        }

    scores = sorted(
        (
            (
                cosine_similarity(track_embedding, identity.prototype_embedding),
                identity.student_id,
                identity.image_count,
            )
            for identity in gallery
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    best_similarity, best_student_id, best_gallery_count = scores[0]
    if len(scores) > 1:
        second_similarity, second_student_id, _ = scores[1]
    else:
        second_similarity, second_student_id = None, None

    margin_ok = (
        second_similarity is None or (best_similarity - second_similarity) >= review_margin
    )
    if best_similarity >= auto_match_threshold and margin_ok:
        review_status = "auto_matched"
    else:
        review_status = "needs_review"

    return {
        "suggested_student_id": best_student_id,
        "similarity_score": round(float(best_similarity), 4),
        "second_best_student_id": second_student_id,
        "second_best_similarity": None if second_similarity is None else round(float(second_similarity), 4),
        "review_status": review_status,
        "gallery_image_count": best_gallery_count,
    }


def finalize_tracks(
    *,
    tracks: list[Track],
    video_output_dir: Path,
    min_track_length: int,
    gallery: list[GalleryIdentity],
    auto_match_threshold: float,
    review_margin: float,
) -> tuple[list[dict[str, object]], int]:
    tracks_dir = video_output_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    track_entries: list[dict[str, object]] = []
    dropped_count = 0
    for track in tracks:
        suggestion = suggest_gallery_match(
            track_embedding=track.prototype_embedding,
            gallery=gallery,
            auto_match_threshold=auto_match_threshold,
            review_margin=review_margin,
        )
        entry = {
            "track_id": track.track_id,
            "folder": str(tracks_dir / f"track_{track.track_id:03d}"),
            "image_count": len(track.pending_faces),
            "first_frame": track.first_frame,
            "last_frame": track.last_seen_frame,
            "average_bbox_size": round(float(np.mean(track.pending_bbox_sizes)), 2) if track.pending_bbox_sizes else 0.0,
            "average_detection_score": round(float(np.mean(track.pending_scores)), 4) if track.pending_scores else 0.0,
            "status": "kept" if len(track.pending_faces) >= min_track_length else "dropped_short_track",
            **suggestion,
        }

        if len(track.pending_faces) < min_track_length:
            dropped_count += 1
            track_entries.append(entry)
            continue

        track_dir = tracks_dir / f"track_{track.track_id:03d}"
        track_dir.mkdir(parents=True, exist_ok=True)
        for index, aligned_face in enumerate(track.pending_faces):
            output_path = track_dir / f"face_{index:04d}.jpg"
            if not cv2.imwrite(str(output_path), aligned_face):
                raise RuntimeError(f"Failed to write aligned face: {output_path}")
        track.saved_image_count = len(track.pending_faces)
        entry["image_count"] = track.saved_image_count
        track_entries.append(entry)

    return track_entries, dropped_count


def write_outputs(
    *,
    video_output_dir: Path,
    video_path: Path,
    total_frames: int,
    sampled_frames: int,
    total_detections_kept: int,
    total_tracks_created: int,
    total_tracks_retired: int,
    track_entries: list[dict[str, object]],
    dropped_short_tracks: int,
    gallery_dir: Path | None,
) -> None:
    final_tracks_kept = sum(1 for entry in track_entries if entry["status"] == "kept")
    auto_matched_tracks = sum(1 for entry in track_entries if entry.get("review_status") == "auto_matched")
    manifest = {
        "source_video": str(video_path),
        "gallery_dir": None if gallery_dir is None else str(gallery_dir),
        "total_frames_read": total_frames,
        "sampled_frames": sampled_frames,
        "total_detections_kept": total_detections_kept,
        "total_tracks_created": total_tracks_created,
        "total_tracks_retired": total_tracks_retired,
        "final_tracks_kept": final_tracks_kept,
        "dropped_short_tracks": dropped_short_tracks,
        "auto_matched_tracks": auto_matched_tracks,
        "tracks": track_entries,
    }
    manifest_path = video_output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_lines = [
        f"source_video: {video_path}",
        f"gallery_dir: {gallery_dir if gallery_dir is not None else 'none'}",
        f"total_frames_read: {total_frames}",
        f"sampled_frames: {sampled_frames}",
        f"total_detections_kept: {total_detections_kept}",
        f"total_tracks_created: {total_tracks_created}",
        f"total_tracks_retired: {total_tracks_retired}",
        f"final_tracks_kept: {final_tracks_kept}",
        f"dropped_short_tracks: {dropped_short_tracks}",
        f"auto_matched_tracks: {auto_matched_tracks}",
        "",
        "Kept tracks:",
    ]
    kept_entries = [entry for entry in track_entries if entry["status"] == "kept"]
    for entry in kept_entries:
        summary_lines.append(
            f"- track_{entry['track_id']:03d}: images={entry['image_count']} "
            f"frames={entry['first_frame']}->{entry['last_frame']} "
            f"suggested={entry.get('suggested_student_id')} score={entry.get('similarity_score')} "
            f"status={entry.get('review_status')}"
        )
    if not kept_entries:
        summary_lines.append("- none")

    summary_lines.extend(["", "Dropped tracks:"])
    dropped_entries = [entry for entry in track_entries if entry["status"] != "kept"]
    for entry in dropped_entries:
        summary_lines.append(
            f"- track_{entry['track_id']:03d}: status={entry['status']} images={entry['image_count']}"
        )
    if not dropped_entries:
        summary_lines.append("- none")

    summary_path = video_output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def process_group_video_to_tracks(
    video_path: str,
    output_base_dir: str,
    frame_skip: int = 3,
    min_face_size: int = 50,
    max_tracks: int | None = None,
    max_inactive_frames: int = 30,
    min_track_length: int = 5,
    embedding_match_threshold: float = 0.45,
    iou_match_threshold: float = 0.20,
    gallery_dir: str | None = None,
    auto_match_threshold: float = DEFAULT_AUTO_MATCH_THRESHOLD,
    review_margin: float = DEFAULT_REVIEW_MARGIN,
) -> None:
    video_path_obj = Path(video_path).expanduser().resolve()
    output_base_dir_obj = Path(output_base_dir).expanduser().resolve()
    gallery_dir_obj = None if gallery_dir is None else Path(gallery_dir).expanduser().resolve()
    _validate_inputs(
        video_path=video_path_obj,
        output_base_dir=output_base_dir_obj,
        frame_skip=frame_skip,
        min_face_size=min_face_size,
        max_tracks=max_tracks,
        max_inactive_frames=max_inactive_frames,
        min_track_length=min_track_length,
        embedding_match_threshold=embedding_match_threshold,
        iou_match_threshold=iou_match_threshold,
        gallery_dir=gallery_dir_obj,
        auto_match_threshold=auto_match_threshold,
        review_margin=review_margin,
    )

    video_output_dir = output_base_dir_obj / video_path_obj.stem
    if video_output_dir.exists():
        shutil.rmtree(video_output_dir)
    video_output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing InsightFace detector + recognizer...")
    app = _build_face_app()
    gallery = build_gallery(app, gallery_dir_obj) if gallery_dir_obj is not None else []

    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path_obj}")

    total_frames_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"Processing group video={video_path_obj} frames={total_frames_reported} "
        f"frame_skip={frame_skip} min_face_size={min_face_size} max_tracks={max_tracks} "
        f"max_inactive_frames={max_inactive_frames} min_track_length={min_track_length} "
        f"gallery_dir={gallery_dir_obj}"
    )

    tracks: list[Track] = []
    next_track_id = 1
    frame_idx = 0
    sampled_frames = 0
    total_detections_kept = 0
    total_tracks_retired = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            retired_now = retire_inactive_tracks(tracks, frame_idx, max_inactive_frames)
            total_tracks_retired += retired_now

            if (frame_idx - 1) % frame_skip != 0:
                continue

            sampled_frames += 1
            detections = build_detections(
                app=app,
                frame_bgr=frame,
                frame_idx=frame_idx,
                min_face_size=min_face_size,
            )
            if not detections:
                continue

            detections.sort(key=lambda det: det.area, reverse=True)
            used_track_ids: set[int] = set()
            for detection in detections:
                matched_track = choose_track_for_detection(
                    detection=detection,
                    tracks=tracks,
                    embedding_match_threshold=embedding_match_threshold,
                    iou_match_threshold=iou_match_threshold,
                    used_track_ids=used_track_ids,
                )

                if matched_track is None:
                    if max_tracks is not None and next_track_id > max_tracks:
                        continue
                    matched_track = Track(
                        track_id=next_track_id,
                        prototype_embedding=normalize_embedding(detection.embedding),
                        last_bbox=detection.bbox.astype(np.float32),
                        last_seen_frame=detection.frame_idx,
                        first_frame=detection.frame_idx,
                    )
                    tracks.append(matched_track)
                    next_track_id += 1
                    print(
                        f"Created track_{matched_track.track_id:03d} at frame={frame_idx} "
                        f"bbox=({detection.width:.1f}x{detection.height:.1f})"
                    )

                matched_track.update(detection)
                used_track_ids.add(matched_track.track_id)
                total_detections_kept += 1

                if total_detections_kept % 50 == 0:
                    active_track_count = sum(1 for track in tracks if track.active)
                    print(
                        f"Saved {total_detections_kept} candidate faces | "
                        f"tracks_created={len(tracks)} active_tracks={active_track_count} retired_tracks={total_tracks_retired}"
                    )
    finally:
        cap.release()

    track_entries, dropped_short_tracks = finalize_tracks(
        tracks=tracks,
        video_output_dir=video_output_dir,
        min_track_length=min_track_length,
        gallery=gallery,
        auto_match_threshold=auto_match_threshold,
        review_margin=review_margin,
    )
    write_outputs(
        video_output_dir=video_output_dir,
        video_path=video_path_obj,
        total_frames=frame_idx,
        sampled_frames=sampled_frames,
        total_detections_kept=total_detections_kept,
        total_tracks_created=len(tracks),
        total_tracks_retired=total_tracks_retired,
        track_entries=track_entries,
        dropped_short_tracks=dropped_short_tracks,
        gallery_dir=gallery_dir_obj,
    )

    kept_track_count = sum(1 for entry in track_entries if entry["status"] == "kept")
    auto_matched_tracks = sum(1 for entry in track_entries if entry.get("review_status") == "auto_matched")
    print(
        "Finished group video import. "
        f"frames_read={frame_idx} sampled_frames={sampled_frames} total_detections_kept={total_detections_kept} "
        f"tracks_created={len(tracks)} tracks_kept={kept_track_count} auto_matched_tracks={auto_matched_tracks} "
        f"dropped_short_tracks={dropped_short_tracks} output_dir={video_output_dir}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a group video, track people, save aligned face crops per track, and suggest student matches."
    )
    parser.add_argument("video_path", help="Path to the group video file.")
    parser.add_argument("output_base_dir", help="Base directory where track folders will be created.")
    parser.add_argument("--frame-skip", type=int, default=3, help="Sample every Nth frame. Default: 3.")
    parser.add_argument("--min-face-size", type=int, default=50, help="Skip faces smaller than this threshold in pixels.")
    parser.add_argument("--max-tracks", type=int, default=None, help="Optional cap on the total number of created tracks.")
    parser.add_argument(
        "--max-inactive-frames",
        type=int,
        default=30,
        help="Retire a track if unseen for more than this many frames.",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=5,
        help="Drop tracks shorter than this many saved aligned faces.",
    )
    parser.add_argument(
        "--embedding-match-threshold",
        type=float,
        default=0.45,
        help="Maximum cosine distance allowed for embedding-based track matching.",
    )
    parser.add_argument(
        "--iou-match-threshold",
        type=float,
        default=0.20,
        help="Minimum IoU required unless the embedding match is very strong.",
    )
    parser.add_argument(
        "--gallery-dir",
        default=None,
        help="Optional existing student dataset root used to suggest student IDs for each track.",
    )
    parser.add_argument(
        "--auto-match-threshold",
        type=float,
        default=DEFAULT_AUTO_MATCH_THRESHOLD,
        help="Minimum cosine similarity required to mark a track as auto_matched.",
    )
    parser.add_argument(
        "--review-margin",
        type=float,
        default=DEFAULT_REVIEW_MARGIN,
        help="Required gap between best and second-best gallery matches to auto accept a suggestion.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process_group_video_to_tracks(
        video_path=args.video_path,
        output_base_dir=args.output_base_dir,
        frame_skip=args.frame_skip,
        min_face_size=args.min_face_size,
        max_tracks=args.max_tracks,
        max_inactive_frames=args.max_inactive_frames,
        min_track_length=args.min_track_length,
        embedding_match_threshold=args.embedding_match_threshold,
        iou_match_threshold=args.iou_match_threshold,
        gallery_dir=args.gallery_dir,
        auto_match_threshold=args.auto_match_threshold,
        review_margin=args.review_margin,
    )
