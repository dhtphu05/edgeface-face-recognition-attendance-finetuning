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
python scripts/train_phase3.py
python scripts/prune_phase4.py
python scripts/finetune_phase5.py
```

## Tài liệu vận hành

- Quy trình hybrid training, best practices thu thập dữ liệu từ video, và checklist tối ưu hiệu suất:
  [docs/TRAINING_BEST_PRACTICES.md](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/docs/TRAINING_BEST_PRACTICES.md)

## Ghi chú

- `models/edgeface_xxs.py` hiện là backbone gọn để bootstrap pipeline.
- `models/loralin_conv.py` là scaffold cho lớp LoRaLin-Conv, chưa phải công thức paper đầy đủ.
- `core_losses/adaface_loss.py` là bản khởi tạo thực dụng để giúp pipeline chạy end-to-end.
