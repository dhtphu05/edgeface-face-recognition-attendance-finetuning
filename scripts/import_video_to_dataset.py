from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align


IMAGE_SIZE = 112
FILENAME_PATTERN = re.compile(r"^face_(\d{4,})\.jpg$", re.IGNORECASE)


def _validate_inputs(
    video_path: Path,
    student_id: str,
    output_base_dir: Path,
    frame_skip: int,
    min_face_size: int,
) -> None:
    if not student_id.strip():
        raise ValueError("student_id must not be empty.")
    if frame_skip <= 0:
        raise ValueError("frame_skip must be greater than 0.")
    if min_face_size <= 0:
        raise ValueError("min_face_size must be greater than 0.")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    output_base_dir.mkdir(parents=True, exist_ok=True)


def _build_face_app() -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def _next_image_index(output_dir: Path) -> int:
    max_index = -1
    for image_path in output_dir.glob("face_*.jpg"):
        match = FILENAME_PATTERN.match(image_path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def process_video_to_dataset(
    video_path: str,
    student_id: str,
    output_base_dir: str,
    frame_skip: int = 3,
    min_face_size: int = 50,
) -> None:
    video_path_obj = Path(video_path).expanduser().resolve()
    output_base_dir_obj = Path(output_base_dir).expanduser().resolve()
    _validate_inputs(
        video_path=video_path_obj,
        student_id=student_id,
        output_base_dir=output_base_dir_obj,
        frame_skip=frame_skip,
        min_face_size=min_face_size,
    )

    output_dir = output_base_dir_obj / student_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing InsightFace RetinaFace detector...")
    app = _build_face_app()

    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path_obj}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    next_index = _next_image_index(output_dir)

    print(
        f"Processing video={video_path_obj} student_id={student_id} "
        f"frames={total_frames} frame_skip={frame_skip} min_face_size={min_face_size}"
    )

    frame_idx = 0
    sampled_frames = 0
    detected_frames = 0
    small_face_skips = 0
    saved_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            if (frame_idx - 1) % frame_skip != 0:
                continue

            sampled_frames += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = app.get(rgb_frame)
            if not faces:
                continue

            largest_face = max(
                faces,
                key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
            )
            detected_frames += 1

            face_width = float(largest_face.bbox[2] - largest_face.bbox[0])
            face_height = float(largest_face.bbox[3] - largest_face.bbox[1])
            if min(face_width, face_height) < min_face_size:
                small_face_skips += 1
                continue

            aligned_face = face_align.norm_crop(
                frame,
                landmark=largest_face.kps,
                image_size=IMAGE_SIZE,
            )

            output_path = output_dir / f"face_{next_index:04d}.jpg"
            if not cv2.imwrite(str(output_path), aligned_face):
                raise RuntimeError(f"Failed to write image: {output_path}")

            next_index += 1
            saved_count += 1

            if saved_count % 50 == 0:
                print(f"Saved {saved_count} aligned faces to {output_dir}")
    finally:
        cap.release()

    print(
        "Finished video import. "
        f"frames_read={frame_idx} sampled_frames={sampled_frames} "
        f"frames_with_faces={detected_frames} skipped_small_faces={small_face_skips} "
        f"images_saved={saved_count} output_dir={output_dir}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a video and extract aligned face crops for training."
    )
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("student_id", help="Student identifier used for dataset subdirectory.")
    parser.add_argument("output_base_dir", help="Base directory for dataset output.")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=3,
        help="Sample every Nth frame. Default: 3.",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=50,
        help="Skip faces smaller than this threshold in pixels. Default: 50.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process_video_to_dataset(
        video_path=args.video_path,
        student_id=args.student_id,
        output_base_dir=args.output_base_dir,
        frame_skip=args.frame_skip,
        min_face_size=args.min_face_size,
    )
