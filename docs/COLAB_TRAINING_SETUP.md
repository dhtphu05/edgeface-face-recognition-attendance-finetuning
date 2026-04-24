# Colab Training Setup

Tài liệu này mô tả cách chạy huấn luyện trên Google Colab khi dataset đã nằm trên Google Drive.

Flow khuyến nghị cho repo hiện tại:

1. public face pretraining trên dataset lớn ở Drive
2. finetune trên `clean-core`
3. finetune tiếp trên full internal dataset

Không dùng KD hoặc pruning ở bước đầu trên Colab.

## 1. Cấu trúc thư mục nên có trên Google Drive

Ví dụ:

```text
MyDrive/
  Face-Recognition-Workspace/
    Attendance_Workspace/
      3_edgeface_training/
        requirements.txt
        scripts/
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

Điểm quan trọng:

- public dataset phải có cấu trúc `train/` và `val/`
- `train/` có thể là dạng shard nhiều tầng như `n002_044/n000002/...`; loader hiện tại đã hỗ trợ trực tiếp
- internal split cũng nên có `train/`, `val/`, `test/`
- checkpoint nên lưu ngay trong `3_edgeface_training/checkpoints/` để không mất khi runtime reset

## 2. Mở notebook bằng Colab

Notebook đã được chuẩn bị sẵn tại:

- [3_edgeface_training/notebooks/colab_public_pretrain.ipynb](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/notebooks/colab_public_pretrain.ipynb)

Bạn có thể dùng một trong hai cách:

1. Upload notebook này lên Google Drive rồi chọn `Open with > Google Colaboratory`
2. Cài extension Colab trong VS Code và mở trực tiếp file `.ipynb`

Khi chạy, cần chọn runtime GPU trong Colab:

- `Runtime > Change runtime type > GPU`

## 3. Stage A: Public Pretraining trên dataset lớn

Notebook hiện mặc định cho stage này.

Path mặc định trong notebook:

- `PROJECT_ROOT=/content/drive/MyDrive/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training`
- `DATASET_ROOT=/content/drive/MyDrive/Face_Recognition`

Notebook sẽ:

- mount Google Drive
- kiểm tra tồn tại của project và dataset
- cài `requirements.txt`
- chạy `train_phase3.py` với cấu hình:
  - `width_preset=widened`
  - `rank_ratio=0.7`
  - `kd_alpha=0`
  - `skip_student_bootstrap=True`
  - `skip_teacher_bootstrap=True`

Checkpoint đầu ra dự kiến:

- `checkpoints/phase3_public_pretrain_best.pth`
- `checkpoints/phase3_public_pretrain_metrics.json`

## 4. Stage B: Finetune trên clean-core

Sau khi có checkpoint pretrain tốt nhất, chạy tiếp trên Colab hoặc local:

```bash
cd /content/drive/MyDrive/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training

python scripts/train_phase3.py \
  --dataset-root /content/drive/MyDrive/Face-Recognition-Workspace/Attendance_Workspace/2_face_dataset_clean_core_split \
  --checkpoints-dir /content/drive/MyDrive/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/checkpoints \
  --width-preset widened \
  --rank-ratio 0.7 \
  --kd-alpha 0 \
  --skip-teacher-bootstrap \
  --student-weights checkpoints/phase3_public_pretrain_best.pth \
  --epochs 40 \
  --learning-rate 5e-5 \
  --batch-size 64 \
  --num-workers 2 \
  --output-prefix phase3_public_to_clean_core
```

## 5. Stage C: Finetune trên full internal dataset

```bash
cd /content/drive/MyDrive/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training

python scripts/train_phase3.py \
  --dataset-root /content/drive/MyDrive/Face-Recognition-Workspace/Attendance_Workspace/2_face_dataset_split \
  --checkpoints-dir /content/drive/MyDrive/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/checkpoints \
  --width-preset widened \
  --rank-ratio 0.7 \
  --kd-alpha 0 \
  --skip-teacher-bootstrap \
  --student-weights checkpoints/phase3_public_to_clean_core_best.pth \
  --epochs 10 \
  --learning-rate 5e-5 \
  --batch-size 64 \
  --num-workers 2 \
  --output-prefix phase3_cleancore_to_full_ft
```

## 6. Stage D: Evaluate trên internal held-out split

```bash
cd /content/drive/MyDrive/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training

python scripts/evaluate_paper_metrics.py \
  --checkpoint checkpoints/phase3_cleancore_to_full_ft_best.pth \
  --dataset-root /content/drive/MyDrive/Face-Recognition-Workspace/Attendance_Workspace/2_face_dataset_split \
  --num-workers 0 \
  --report-json checkpoints/phase3_cleancore_to_full_ft_eval.json
```

## 7. Cấu hình Colab khuyến nghị

Mặc định an toàn:

- `batch_size=64`
- `num_workers=2`
- `epochs=20` cho public pretrain smoke run đầu tiên

Nếu GPU nhỏ hoặc Colab báo out-of-memory:

- giảm `batch_size` xuống `32`
- nếu vẫn lỗi, giảm xuống `16`

Nếu Drive I/O chậm:

- giữ `num_workers` thấp (`2` hoặc `0`)
- tránh đặt quá cao vì Colab không phải lúc nào cũng đọc Drive tốt

## 8. Những gì không nên làm

Không nên:

- train full public dataset ngay với KD
- dùng pruning trong public pretraining stage
- trộn class public dataset vào `2_face_dataset`
- kỳ vọng checkpoint public là metric cuối; metric cuối vẫn phải là held-out internal split

## 9. Checklist chạy thật

Trước khi bấm run trên Colab, kiểm tra:

- runtime đã là GPU
- notebook mount đúng Drive
- `PROJECT_ROOT` đúng thư mục project
- `DATASET_ROOT` đúng thư mục dataset public
- trong dataset có `train/` và `val/`
- checkpoint output nằm trên Drive, không nằm ở `/content` tạm thời

## 10. Kết luận

Setup tốt nhất cho repo hiện tại là:

- Colab cho public pretraining
- local hoặc Colab cho internal finetuning
- internal held-out split là chuẩn đánh giá cuối

Đây là hướng phù hợp nhất với codebase hiện tại và không làm thay đổi flow tối ưu `8.2 -> 8.6`.
