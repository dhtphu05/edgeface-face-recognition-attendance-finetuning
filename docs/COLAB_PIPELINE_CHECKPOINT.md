# Colab Pipeline Checkpoint

Tài liệu này ghi lại trạng thái hiện tại của setup Google Colab so với pipeline nghiên cứu mục tiêu.

## Mục tiêu của file này

File này dùng để trả lời ngắn gọn 3 câu hỏi:

1. setup Colab hiện tại đang hỗ trợ phần nào của pipeline
2. phần nào đã đúng với hướng nghiên cứu
3. phần nào vẫn chưa hoàn thành và cần làm tiếp

## 1. Kết luận ngắn

Setup Colab hiện tại:

- đúng với hướng tối ưu thực dụng của repo
- đúng với phần trọng tâm hiện tại là tăng chất lượng `Phase 3`
- chưa khớp 100% với toàn bộ pipeline nghiên cứu `8.2 -> 8.6`

Nói ngắn gọn:

- **đúng hướng**
- **đúng với codebase hiện tại**
- **chưa phải bản hoàn chỉnh cuối cùng của pipeline nghiên cứu**

## 2. Những gì setup Colab hiện tại đã hỗ trợ

### 2.1. Public face pretraining

Đã hỗ trợ:

- mount Google Drive
- train trực tiếp từ dataset public lớn trên Drive
- hỗ trợ cấu trúc dataset hiện tại:
  - `train/` dạng shard như `n002_044/n000002/...`
  - `val/` dạng class trực tiếp như `val/n000001/...`
- lưu checkpoint về Drive

Điều này phù hợp với hướng:

- tăng face-domain knowledge trước khi finetune trên dữ liệu sinh viên nội bộ

### 2.2. AdaFace-only baseline

Đã hỗ trợ:

- `width_preset=widened`
- `rank_ratio=0.7`
- `kd_alpha=0`
- `skip_student_bootstrap`
- `skip_teacher_bootstrap`

Điều này phù hợp với thực nghiệm hiện tại, vì nhánh `AdaFace-only` đang là nhánh ổn định nhất.

### 2.3. Flow huấn luyện nhiều giai đoạn

Setup hiện tại đã bám đúng flow thực dụng:

1. public pretrain trên Colab
2. clean-core finetune
3. full-dataset finetune
4. internal held-out evaluation

Flow này tương thích với định hướng nghiên cứu, dù chưa phải phiên bản đầy đủ cuối cùng.

## 3. Những gì setup Colab hiện tại chưa khớp hoàn toàn với pipeline nghiên cứu

### 3.1. Chưa có pretrained transfer đúng kiểu MS1MV2-compatible

Pipeline nghiên cứu kỳ vọng:

- student và teacher được nạp pretrained weights tương thích từ một nguồn face-domain lớn như MS1MV2

Trạng thái hiện tại:

- setup Colab đang chọn hướng `train student trực tiếp trên public dataset`
- chưa có `student/teacher checkpoint` tương thích đúng kiểu MS1MV2 bootstrap

Kết luận:

- **chưa khớp hoàn toàn narrative học thuật ban đầu**
- nhưng **hợp lý hơn về mặt triển khai thực tế**

### 3.2. Chưa có Knowledge Distillation hiệu quả

Pipeline nghiên cứu kỳ vọng:

- dùng ResNet-101 teacher
- dùng embedding KD loss
- duy trì cân bằng gradient với AdaFace

Trạng thái hiện tại:

- KD path đã có code
- nhưng teacher hiện chưa tương thích đủ tốt
- run tốt nhất vẫn là `kd_alpha=0`

Kết luận:

- setup Colab hiện tại **chưa hoàn thành phần KD**
- Colab hiện chỉ đang phục vụ nhánh baseline mạnh hơn cho `Phase 3`

### 3.3. Chưa đi tới pruning và recovery

Pipeline nghiên cứu kỳ vọng:

- sau baseline mạnh mới pruning
- sau pruning mới recovery finetune

Trạng thái hiện tại:

- setup Colab chưa triển khai phần này
- đây là quyết định chủ động để tránh prune quá sớm khi baseline verification còn yếu

Kết luận:

- `Phase 4` và `Phase 5` chưa nằm trong setup Colab hiện tại

### 3.4. Chưa hoàn tất export và deployment

Pipeline nghiên cứu kỳ vọng:

- export mô hình
- benchmark và deploy trên phần cứng mục tiêu

Trạng thái hiện tại:

- mới có benchmark params/FLOPs/latency/FPS trong evaluate
- chưa có pipeline export/deploy đầy đủ

Kết luận:

- `Phase 6` chưa hoàn thành

## 4. Đối chiếu nhanh theo từng phase

| Phase | Mục tiêu nghiên cứu | Setup Colab hiện tại | Trạng thái |
|---|---|---|---|
| 8.2 | Tái cấu trúc kiến trúc | Đã dùng backbone `widened` hiện tại | Đã hỗ trợ |
| 8.3 | AdaFace + KD | Hiện hỗ trợ mạnh phần AdaFace-only | Đạt một phần |
| 8.4 | Structural pruning | Chưa triển khai trong Colab setup này | Chưa đạt |
| 8.5 | Recovery finetune | Chưa triển khai trong Colab setup này | Chưa đạt |
| 8.6 | Export & deployment | Chưa triển khai trong Colab setup này | Chưa đạt |

## 5. Kết luận học thuật nên dùng

Nếu cần mô tả trung thực trong báo cáo, nên viết:

> Setup Google Colab hiện tại được sử dụng như một hạ tầng tăng cường cho Giai đoạn 3, tập trung vào public face pretraining và domain adaptation nội bộ theo nhánh AdaFace-only. Hệ thống này phù hợp với mục tiêu cải thiện chất lượng biểu diễn embedding trước khi thực hiện các giai đoạn chưng cất tri thức, cắt tỉa cấu trúc, tinh chỉnh phục hồi và triển khai. Do đó, Colab setup hiện tại phù hợp với định hướng nghiên cứu tổng thể, nhưng chưa đại diện cho phiên bản hoàn chỉnh cuối cùng của toàn bộ pipeline tối ưu.

## 6. Bước kế tiếp để khớp hơn với pipeline nghiên cứu

Thứ tự nên làm:

1. chạy public pretraining trên Colab
2. finetune trên clean-core
3. finetune trên full dataset nội bộ
4. đánh giá lại internal held-out split
5. nếu baseline đủ mạnh, quay lại thử KD với teacher tương thích
6. chỉ sau đó mới prune
7. recovery finetune sau prune
8. cuối cùng mới export/deploy

## 7. Tổng kết

Checkpoint hiện tại:

- setup Colab **đúng hướng**
- setup Colab **đúng với pipeline thực nghiệm hiện tại**
- setup Colab **chưa hoàn chỉnh toàn bộ pipeline nghiên cứu**

Đây là một checkpoint trung gian hợp lệ, không phải trạng thái cuối.
