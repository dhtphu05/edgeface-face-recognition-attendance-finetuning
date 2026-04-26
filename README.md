# Attendance Workspace

Workspace này được chuẩn hóa theo 3 phân vùng:

- `1_django_collection` trỏ sang dự án Django hiện tại `Face-Aligner-For-Dataset`
- `2_face_dataset` trỏ sang kho ảnh đã căn chỉnh `Face-Aligner-For-Dataset/dataset`
- `3_edgeface_training` là môi trường huấn luyện PyTorch

## Cách dùng nhanh

```bash
cd Attendance_Workspace/3_edgeface_training
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/audit_face_dataset.py --dataset-root ../2_face_dataset --report-json checkpoints/dataset_audit.json
python scripts/create_heldout_split.py --source-root ../2_face_dataset --output-root ../2_face_dataset_split
python scripts/train_phase3.py --dataset-root ../2_face_dataset_split --width-preset widened --rank-ratio 0.7 --kd-alpha 0 --skip-student-bootstrap --skip-teacher-bootstrap --epochs 10 --output-prefix phase3_smoke_nokd
python scripts/train_phase3.py --dataset-root ../2_face_dataset_split --width-preset widened --rank-ratio 0.7 --kd-alpha 0 --skip-student-bootstrap --skip-teacher-bootstrap --epochs 40 --output-prefix phase3_baseline_nokd
python scripts/evaluate_paper_metrics.py --checkpoint checkpoints/phase3_baseline_nokd_best.pth --dataset-root ../2_face_dataset_split --num-workers 0 --report-json checkpoints/phase3_baseline_nokd_eval.json
python scripts/train_phase3.py --dataset-root ../2_face_dataset_split --width-preset widened --rank-ratio 0.7 --kd-alpha 100 --skip-student-bootstrap --skip-teacher-bootstrap --teacher-pretrained-imagenet --epochs 40 --output-prefix phase3_kd100_imagenet
python scripts/prune_phase4.py --input checkpoints/phase3_baseline_nokd_best.pth --output checkpoints/phase4_pruned_model.pth --prune-ratio 0.01
python scripts/finetune_phase5.py --dataset-root ../2_face_dataset_split
```

## Video nhóm

Để tách video nhóm thành các track review thủ công trước khi gán nhãn:

```bash
pip install insightface onnxruntime opencv-python
python scripts/import_group_video_to_tracks.py \
  ../raw_videos/group_video.mp4 \
  ../group_tracks \
  --frame-skip 3 \
  --min-face-size 60 \
  --min-track-length 5 \
  --gallery-dir ../2_face_dataset
```

Đầu ra sẽ có dạng:

```text
group_tracks/<video_stem>/
  manifest.json
  summary.txt
  tracks/
    track_001/
    track_002/
    ...
```

Khi dùng `--gallery-dir`, `manifest.json` và `summary.txt` sẽ có thêm:

- `suggested_student_id`
- `similarity_score`
- `second_best_student_id`
- `review_status`

Các track có `review_status=auto_matched` là gợi ý mạnh, nhưng vẫn nên review trước khi nhập vào dataset thật.

## Tài liệu vận hành

- Quy trình hybrid training, best practices thu thập dữ liệu từ video, và checklist tối ưu hiệu suất:
  [docs/TRAINING_BEST_PRACTICES.md](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/docs/TRAINING_BEST_PRACTICES.md)
- Hướng dẫn setup training trên Google Colab với dataset đã nằm trên Google Drive:
  [docs/COLAB_TRAINING_SETUP.md](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/docs/COLAB_TRAINING_SETUP.md)
- Ghi chú checkpoint về mức độ khớp giữa setup Colab hiện tại và pipeline nghiên cứu:
  [docs/COLAB_PIPELINE_CHECKPOINT.md](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/docs/COLAB_PIPELINE_CHECKPOINT.md)
- Chiến lược tốt nhất hiện tại cho bài toán này khi có public face dataset lớn:
  [docs/PUBLIC_PRETRAINING_STRATEGY.md](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/docs/PUBLIC_PRETRAINING_STRATEGY.md)
- Bảng đối chiếu trạng thái thực tế của pipeline `8.2 -> 8.6`:
  [docs/PIPELINE_STATUS.md](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/docs/PIPELINE_STATUS.md)
- Báo cáo học thuật tiến độ theo format nghiên cứu:
  [docs/ACADEMIC_PROGRESS_REPORT.md](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/docs/ACADEMIC_PROGRESS_REPORT.md)

## Colab

Notebooks Colab:

- [3_edgeface_training/notebooks/colab_public_pretrain.ipynb](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/notebooks/colab_public_pretrain.ipynb)
- [3_edgeface_training/notebooks/colab_internal_finetune.ipynb](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/notebooks/colab_internal_finetune.ipynb)
- [3_edgeface_training/notebooks/colab_internal_evaluate.ipynb](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/notebooks/colab_internal_evaluate.ipynb)

## Ghi chú

- `models/edgeface_xxs.py` hiện là backbone gọn để bootstrap pipeline.
- `models/loralin_conv.py` là scaffold cho lớp LoRaLin-Conv, chưa phải công thức paper đầy đủ.
- `core_losses/adaface_loss.py` là bản khởi tạo thực dụng để giúp pipeline chạy end-to-end.
