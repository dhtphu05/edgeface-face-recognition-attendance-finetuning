# Colab Training Setup

Tài liệu này mô tả setup Colab theo đúng flow vận hành hiện tại của repo:

1. public pretraining trên dataset lớn
2. clean-core finetune trên dữ liệu nội bộ
3. full-dataset finetune trên dữ liệu nội bộ
4. internal held-out evaluation

Chiến lược này giữ nguyên:

- `widened`
- `rank_ratio=0.7`
- `AdaFace-only`
- không KD ở giai đoạn này
- không pruning ở giai đoạn này

## 1. Cấu trúc thư mục trên Drive

Tối thiểu nên có:

```text
MyDrive/
  Attendance_Workspace/
    3_edgeface_training/
      requirements.txt
      scripts/
      models/
      core_losses/
      dataloaders/
      checkpoints/
      notebooks/
    2_face_dataset_clean_core_split/
      train/
      val/
      test/
    2_face_dataset_split/
      train/
      val/
      test/
  Face_Recognition/
    raw_videos/
    train/
      n002_044/
        n000002/
        n000003/
        ...
      n045_086/
        n000045/
        ...
    val/
      n000001/
      n000002/
      ...
```

Lưu ý quan trọng:

- public dataset dùng để train hiện tại là `Face_Recognition/train`
- không dùng trực tiếp `Face_Recognition/val` trong `train_phase3.py`
- lý do: `val/` của dataset public này không cùng class space với `train/`

## 2. Các notebook đã được chuẩn bị

### Public pretraining

- [colab_public_pretrain.ipynb](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/notebooks/colab_public_pretrain.ipynb)

Notebook này có 4 preset:

- `smoke`
- `mid`
- `e10`
- `full`

Mặc định hiện tại nên bắt đầu bằng:

- `ACTIVE_STAGE = 'mid'`

### Internal finetuning

- [colab_internal_finetune.ipynb](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/notebooks/colab_internal_finetune.ipynb)

Notebook này có 2 preset:

- `clean_core`
- `full_ft`

### Internal evaluation

- [colab_internal_evaluate.ipynb](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/notebooks/colab_internal_evaluate.ipynb)

Notebook này dùng để evaluate checkpoint cuối trên internal held-out split.

## 3. Stage A: Public Pretraining

Notebook:

- [colab_public_pretrain.ipynb](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/notebooks/colab_public_pretrain.ipynb)

Path mặc định:

- `PROJECT_ROOT=/content/drive/MyDrive/Attendance_Workspace/3_edgeface_training`
- `DATASET_ROOT=/content/drive/MyDrive/Face_Recognition/train`

Presets:

### `smoke`

- `epochs = 3`
- `batch_size = 32`
- `max_train_batches_per_epoch = 200`
- `max_val_batches = 50`

Mục tiêu:

- xác nhận pipeline chạy đúng

### `mid`

- `epochs = 5`
- `batch_size = 32`
- `max_train_batches_per_epoch = 500`
- `max_val_batches = 100`

Mục tiêu:

- xác nhận learning signal có thật

### `e10`

- `epochs = 10`
- `batch_size = 64`
- `max_train_batches_per_epoch = 1000`
- `max_val_batches = 200`

Mục tiêu:

- tạo run public pretraining đầu tiên đủ ý nghĩa

### `full`

- `epochs = 20`
- `batch_size = 64`
- `max_train_batches_per_epoch = 2000`
- `max_val_batches = 300`

Mục tiêu:

- tạo checkpoint public-pretrained mạnh nhất để seed internal finetuning

## 4. Decision Rule cho public pretraining

### Sau `smoke`

Yêu cầu:

- không crash
- loss giảm
- checkpoint được lưu
- runtime là `cuda`

### Sau `mid`

Yêu cầu:

- train loss giảm tiếp so với smoke
- train accuracy tăng rõ hơn smoke
- val accuracy có xu hướng đi lên

Nếu không đạt:

- giảm `num_workers` nếu Drive I/O kém
- giữ nguyên kiến trúc và loss
- không bật KD

### Sau `e10`

Yêu cầu:

- validation vẫn đang tăng
- run này tốt hơn `mid`

Nếu đạt:

- mới chuyển sang `full`

Nếu không đạt:

- dừng scale tiếp
- xem lại learning curve trước khi đốt thêm GPU

## 5. Stage B: Clean-core finetune

Notebook:

- [colab_internal_finetune.ipynb](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/notebooks/colab_internal_finetune.ipynb)

Preset:

- `ACTIVE_STAGE = 'clean_core'`

Mặc định:

- checkpoint đầu vào:
  - `checkpoints/phase3_public_pretrain_best.pth`
- dataset:
  - `2_face_dataset_clean_core_split`
- `epochs = 30`
- `learning_rate = 5e-5`
- `batch_size = 64`
- no KD

Mục tiêu:

- học identity sinh viên trên tập sạch nhất

## 6. Stage C: Full-dataset finetune

Notebook:

- [colab_internal_finetune.ipynb](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/notebooks/colab_internal_finetune.ipynb)

Preset:

- `ACTIVE_STAGE = 'full_ft'`

Mặc định:

- checkpoint đầu vào:
  - `checkpoints/phase3_public_to_clean_core_best.pth`
- dataset:
  - `2_face_dataset_split`
- `epochs = 10`
- `learning_rate = 5e-5`
- `batch_size = 64`
- no KD

Mục tiêu:

- hấp thụ blur, pose, và lighting variation từ full dataset

## 7. Stage D: Internal evaluation

Notebook:

- [colab_internal_evaluate.ipynb](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/notebooks/colab_internal_evaluate.ipynb)

Đầu ra mong đợi:

- pairwise accuracy
- FAR
- FRR tại FAR `1e-3`
- threshold
- params / FLOPs / latency / FPS

Metric cuối cùng để ra quyết định vẫn phải là:

- internal held-out evaluation

Không dùng public-dataset accuracy làm metric cuối cho bài toán điểm danh.

## 8. Operational Best Practices

- mỗi run phải có `output_prefix` riêng
- không overwrite checkpoint tốt nhất
- giữ `KD disabled`
- không pruning trước khi internal verification đủ mạnh
- ưu tiên tăng `max_train_batches_per_epoch` trước khi tăng full epoch
- nếu Drive đọc chậm, giảm `num_workers` xuống `1` hoặc `0`
- nếu GPU memory yếu, giảm `batch_size` trước khi đổi kiến trúc

## 9. Kết luận

Flow tốt nhất hiện tại trên Colab là:

1. `public_pretrain: smoke -> mid -> e10 -> full`
2. `internal_finetune: clean_core`
3. `internal_finetune: full_ft`
4. `internal_evaluate`

Đây là flow đúng nhất với codebase hiện tại và đúng với trạng thái thực nghiệm hiện có.
