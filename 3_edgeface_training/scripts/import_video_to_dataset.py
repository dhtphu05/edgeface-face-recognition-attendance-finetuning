from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def process_video_to_dataset(
    video_path,
    student_id,
    output_base_dir,
    frame_skip=3,
    min_face_size=50,
):
    """
    Trích xuất khuôn mặt chuẩn 112x112 từ video và lưu vào thư mục dataset.

    Tham số:
    - video_path: Đường dẫn tới file video đã quay.
    - student_id: Mã sinh viên/Tên lớp.
    - output_base_dir: Thư mục lưu dataset.
    - frame_skip: Lấy 1 frame sau mỗi N frames để tránh ảnh giống nhau y hệt.
    - min_face_size: Bỏ qua các khuôn mặt quá nhỏ/quá xa (< 50 pixels).
    """

    video_path = Path(video_path)
    output_base_dir = Path(output_base_dir)

    if frame_skip <= 0:
        raise ValueError("frame_skip phải lớn hơn 0.")
    if min_face_size <= 0:
        raise ValueError("min_face_size phải lớn hơn 0.")

    # 1. Khởi tạo mô hình RetinaFace từ InsightFace.
    print("🔄 Đang khởi tạo mô hình Detection & Alignment...")
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    # 2. Chuẩn bị thư mục đầu ra
    output_dir = output_base_dir / str(student_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Đếm số lượng ảnh đã có sẵn trong thư mục để đặt tên file không bị trùng
    existing_files = len([path for path in output_dir.iterdir() if path.is_file()])
    img_counter = existing_files

    # 3. Mở luồng đọc Video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎥 Bắt đầu xử lý video {video_path} ({total_frames} frames) cho ID: {student_id}")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Bỏ qua các frame lặp lại (Frame Skipping)
        if frame_idx % frame_skip != 0:
            continue

        # Thuật toán InsightFace yêu cầu ảnh màu RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)

        # Nếu không tìm thấy mặt nào, bỏ qua
        if len(faces) == 0:
            continue

        # Ưu tiên lấy khuôn mặt to nhất trong khung hình
        largest_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        # Kiểm tra kích thước khuôn mặt
        bbox_width = largest_face.bbox[2] - largest_face.bbox[0]
        if bbox_width < min_face_size:
            continue

        # 4. Alignment và cắt ảnh 112x112 bằng 5 landmarks
        aligned_face = face_align.norm_crop(frame, landmark=largest_face.kps, image_size=112)

        # 5. Lưu file
        filename = f"face_{student_id}_{img_counter:04d}.jpg"
        save_path = output_dir / filename
        cv2.imwrite(str(save_path), aligned_face)

        img_counter += 1
        saved_count += 1

        if saved_count % 50 == 0:
            print(f"   -> Đã trích xuất {saved_count} ảnh từ video...")

    cap.release()
    print(f"✅ Hoàn tất! Đã lưu {saved_count} ảnh khuôn mặt kích thước 112x112 vào thư mục: {output_dir}")


if __name__ == "__main__":
    # CẤU HÌNH ĐƯỜNG DẪN TẠI ĐÂY
    # Lưu ý: Hãy tạo một thư mục 'raw_videos' để chứa các video bạn quay bằng điện thoại
    VIDEO_FILE = WORKSPACE_ROOT / "raw_videos" / "video_phu.mp4"
    STUDENT_ID = "Phu_102250219"
    DATASET_DIR = WORKSPACE_ROOT / "2_face_dataset"

    process_video_to_dataset(
        video_path=VIDEO_FILE,
        student_id=STUDENT_ID,
        output_base_dir=DATASET_DIR,
        frame_skip=3,  # Nếu video mượt 60fps, có thể tăng lên 5 hoặc 10
        min_face_size=60,  # Lọc bỏ rác
    )
