# Bảng Đối Chiếu Pipeline Tối Ưu Face Recognition

## Mục tiêu

Bảng này dùng để đối chiếu giữa pipeline kỳ vọng và trạng thái triển khai thực tế hiện tại của hệ thống.

| Hạng mục | Mô tả kỳ vọng | Trạng thái hiện tại | Đã đạt / Chưa đạt | Cần làm gì để khớp 100% |
|---|---|---|---|---|
| 8.2.1 Mở rộng kiến trúc | Khước từ mô hình `<200K` params, đưa EdgeFace-XXS vào vùng ngọt `600K-800K` params | Đã mở rộng backbone sang preset `widened`, nhưng hiện mới khoảng `527.5K` params | Đạt một phần | Tăng nhẹ stage channels để model nằm trong dải `600K-800K` nếu muốn khớp hoàn toàn narrative |
| 8.2.2 LoRaLin-Conv | Tích hợp LoRaLin-Conv vào các lớp `1x1`, ưu tiên Stage 3 và Stage 4, với `γ = 0.6` | LoRaLin đã được tích hợp, nhưng cấu hình thực tế tốt nhất đang dùng `rank_ratio=0.7`; cách đặt block chưa khớp hoàn toàn mô tả “chỉ Stage 3 & 4” | Đạt một phần | Chuẩn hóa lại kiến trúc đúng theo mô tả paper: chỉ rõ block nào dùng LoRaLin và cố định `γ` nếu cần |
| 8.3.1 Pretrained student | Student được nạp pretrained weights tương thích từ tập lớn như MS1MV2 | Chưa có pretrained student tương thích; file `edgeface_xxs.pt` hiện không khớp kiến trúc đang dùng | Chưa đạt | Pretrain lại chính student backbone trên public face dataset lớn hoặc có checkpoint tương thích thật |
| 8.3.2 Pretrained teacher | Teacher ResNet-101 được nạp pretrained face-domain weights tương thích | `resnet101_adaface.pt` hiện không khớp `ResNet101Teacher` trong repo | Chưa đạt | Dùng teacher face-domain tương thích thật hoặc huấn luyện/convert lại checkpoint teacher |
| 8.3.3 AdaFace baseline | Huấn luyện cơ sở bằng AdaFace + AdamW + LR `1e-4` | Đã chạy ổn định, đây là nhánh tốt nhất hiện tại | Đã đạt | Giữ làm baseline chính |
| 8.3.4 Embedding KD Loss | Student bám teacher bằng MSE embedding 512 chiều, `alpha` đủ lớn để cân bằng gradient | Code KD có sẵn nhưng nhánh KD chưa chứng minh hiệu quả; best run hiện tại vẫn là `kd_alpha=0` | Chưa đạt | Chỉ bật lại KD sau khi có teacher tương thích và so sánh thực nghiệm với baseline sạch |
| 8.3.5 Domain adaptation | Fine-tune trên dữ liệu cục bộ để thích nghi miền triển khai | Đã làm tốt hơn trước nhờ clean-core, held-out split, group-track review | Đạt một phần | Tăng thêm dữ liệu sạch theo session và pretrain trên public face dataset trước khi finetune nội bộ |
| 8.4.1 Structural pruning | Cắt tỉa cấu trúc 5% kênh yếu nhất ở lớp sâu, tránh phá Stage 1 | Script pruning đã có, nhưng chưa có baseline đủ mạnh để prune một cách đáng tin | Chưa đạt | Chỉ prune sau khi baseline verification đủ mạnh; thử ladder `0.01 -> 0.02 -> 0.03 -> 0.05` |
| 8.4.2 Hiệu quả pruning | Sau pruning, accuracy vẫn giữ gần baseline | Chưa có bằng chứng thực nghiệm | Chưa đạt | Chạy prune trên checkpoint mạnh nhất rồi evaluate lại |
| 8.5.1 Fine-tune recovery | Sau pruning, finetune 10 epoch với LR nhỏ để phục hồi accuracy | Script có sẵn nhưng chưa có chuỗi prune -> recovery -> eval tốt | Chưa đạt | Thực thi đủ chuỗi Phase 4 -> Phase 5 và so sánh với baseline trước pruning |
| 8.5.2 Accuracy recovery | Accuracy sau recovery tiệm cận baseline | Chưa có kết quả xác nhận | Chưa đạt | Chỉ công bố sau khi có số liệu recovery thật |
| 8.6.1 Export | Xuất mô hình sang định dạng phục vụ triển khai edge | Chưa phải phần hoàn thiện trong pipeline hiện tại | Chưa đạt | Thêm bước export checkpoint sang định dạng deploy thực tế |
| 8.6.2 Deployment benchmarking | Đo latency/FPS trên phần cứng mục tiêu | Đã có đo Params/FLOPs/Latency/FPS trong script evaluate | Đạt một phần | Benchmark trực tiếp trên thiết bị mục tiêu như Raspberry Pi / edge device thực tế |
| 8.6.3 End-to-end deployment | Mô hình được đưa vào pipeline triển khai cuối | Chưa hoàn thiện | Chưa đạt | Thêm bước inference, thresholding, export, và test thực địa |

## Tóm tắt trạng thái hiện tại

### Các phần đã đạt tốt

- Backbone đã được mở rộng lên khoảng `527.5K params`
- AdaFace baseline đã chạy ổn định
- Có held-out split để đánh giá
- Có pipeline xử lý video nhóm thành track để giảm nhiễu nhãn
- Có workflow clean-core để tạo tập dữ liệu lõi sạch hơn

### Các phần mới đạt một phần

- Domain adaptation trên dữ liệu nội bộ
- Đo lường độ phức tạp phần cứng
- Tái cấu trúc kiến trúc nhưng chưa khớp hoàn toàn narrative `600K-800K params`, `γ=0.6`, `Stage 3-4 only`

### Các phần chưa đạt

- Pretrained student tương thích thật
- Teacher face-domain tương thích thật
- KD hiệu quả
- Pruning + recovery có số liệu tốt
- Export & deployment hoàn chỉnh

## Đánh giá tổng thể

### Theo góc nhìn “code scaffold”

- Khoảng `60%` pipeline đã có cấu trúc và script

### Theo góc nhìn “đã chạy ổn và có số liệu đáng tin”

- Khoảng `40% - 45%`

## Thứ tự ưu tiên để khớp 100%

1. Pretrain student backbone trên public face dataset lớn
2. Finetune trên `clean-core`
3. Finetune tiếp trên full internal dataset
4. Chỉ sau đó mới thử lại KD với teacher tương thích
5. Khi baseline đủ mạnh mới bước sang pruning
6. Sau pruning mới làm recovery finetune
7. Cuối cùng mới export và benchmark deployment

## Kết luận

Hiện tại hệ thống **đã hoàn thành tốt phần tái cấu trúc cơ bản và baseline AdaFace**, nhưng **chưa hoàn thành phần pretrained transfer, KD, pruning-recovery, và deployment** theo đúng mô tả đầy đủ của pipeline mục tiêu.

Nếu cần mô tả trung thực trong báo cáo, nên viết:

- **8.2 đã hoàn thành ở mức thực dụng**
- **8.3 mới hoàn thành nhánh AdaFace baseline**
- **8.4, 8.5, 8.6 hiện mới ở mức scaffold / chưa xác nhận đầy đủ bằng thực nghiệm**
