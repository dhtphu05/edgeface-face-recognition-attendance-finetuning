# Báo Cáo Học Thuật Tiến Độ Hệ Thống Face Recognition

## 1. Mục tiêu nghiên cứu

Mục tiêu của đề tài là xây dựng một hệ thống nhận diện khuôn mặt có khả năng hoạt động ổn định trong bối cảnh điểm danh thực tế, nơi dữ liệu đầu vào không còn mang tính kiểm soát như trong môi trường phòng thí nghiệm. Hệ thống cần đạt được đồng thời ba yêu cầu chính:

- độ chính xác nhận diện đủ cao để sử dụng trong môi trường thực tế và hướng tới mức công bố học thuật
- dung lượng mô hình đủ nhỏ để triển khai trên các thiết bị biên
- khả năng chống chịu với các biến thiên khó như mờ chuyển động, góc nhìn lệch, ánh sáng không đồng đều và khác biệt giữa các phiên ghi hình

Trên cơ sở đó, nghiên cứu không lựa chọn hướng huấn luyện thuần túy trên bộ ảnh tĩnh chất lượng cao, mà tập trung xây dựng một pipeline khai thác dữ liệu từ video thực tế, sau đó huấn luyện một backbone gọn nhẹ theo hướng tối ưu cho edge deployment.

## 2. Kiến trúc hệ thống

Hệ thống hiện tại được tổ chức theo các phân hệ chính sau:

### 2.1. Thu thập và chuẩn hóa dữ liệu

Nguồn dữ liệu nội bộ được xây dựng từ video quay thực tế. Thay vì đưa trực tiếp từng khung hình vào tập huấn luyện, hệ thống thực hiện các bước:

- giải mã video bằng OpenCV
- phát hiện khuôn mặt bằng InsightFace
- căn chỉnh theo 5 landmarks về kích thước `112x112`
- lưu trữ ảnh đã align vào cấu trúc dữ liệu phù hợp cho huấn luyện

Đối với video nhóm, pipeline đã được mở rộng sang dạng:

- `group video -> track -> review -> assign identity -> import into dataset`

Hướng tiếp cận này làm giảm sai lệch nhãn so với phương pháp trích xuất trực tiếp theo từng sinh viên.

### 2.2. Mạng nơ-ron học sinh

Backbone học sinh hiện tại là biến thể mở rộng của `EdgeFaceXXS`, được cài đặt trong:

- [edgeface_xxs.py](/Users/mac/Desktop/Side_Project/Face-Recognition-Workspace/Attendance_Workspace/3_edgeface_training/models/edgeface_xxs.py)

Kiến trúc hỗ trợ nhiều preset độ rộng, trong đó preset đang dùng hiệu quả nhất là `widened`. Mô hình hiện có khoảng:

- `527,502` tham số
- `56.71M` FLOPs

Các lớp `LoRaLin-Conv` đã được tích hợp để giảm chi phí tính toán và giữ cấu trúc không gian hình học của ảnh đầu vào.

### 2.3. Huấn luyện và đánh giá

Nhánh huấn luyện ổn định nhất hiện tại là:

- `AdaFace-only`
- `AdamW`
- không dùng bootstrap weights không tương thích
- không dùng KD ở các run tốt nhất

Quy trình dữ liệu đã được mở rộng thành hai tầng:

- `clean-core dataset`: chỉ giữ các ảnh có tên bắt đầu bằng `student_id`, được xem là tập sạch nhất
- `full dataset`: bao gồm cả các ảnh khó hơn, được bổ sung từ group tracks sau khi review

Đánh giá được thực hiện trên held-out split nội bộ bằng các chỉ số:

- pairwise accuracy
- FAR
- FRR tại `FAR = 10^-3`
- threshold
- params
- FLOPs
- latency / FPS

## 3. Kết quả hiện tại

Sau quá trình tinh chỉnh pipeline dữ liệu và huấn luyện, hệ thống đã đạt được các kết quả quan trọng sau:

### 3.1. Về độ ổn định của pipeline

Pipeline huấn luyện ban đầu từng bị ảnh hưởng bởi:

- bootstrap checkpoint không tương thích với kiến trúc student hiện tại
- teacher checkpoint không tương thích với kiến trúc `ResNet101Teacher`
- nhánh KD gây suy giảm validation thay vì cải thiện

Sau khi loại bỏ các bootstrap sai và quay về nhánh `AdaFace-only`, hệ thống bắt đầu hội tụ ổn định hơn.

### 3.2. Về hiệu năng mô hình

Kết quả tốt nhất hiện tại trên các nhánh nội bộ cho thấy:

- mô hình `widened` đạt khoảng `527.5K params`
- tốc độ suy luận đo được khoảng `713 - 983 FPS` tùy môi trường đo
- đây là mức phù hợp cho định hướng edge deployment

### 3.3. Về kết quả nhận diện

Một số mốc thực nghiệm nổi bật:

- baseline nội bộ tốt trước giai đoạn mở rộng dữ liệu khó:
  - `Val Acc` tốt nhất khoảng `89.23%`
  - pairwise accuracy khoảng `85.85%`
  - `FRR ≈ 53.86%` tại `FAR ≈ 0.001`

- mô hình huấn luyện trên `clean-core dataset`:
  - pairwise accuracy `86.13%`
  - `FAR = 0.001004`
  - `FRR = 67.60%`

- mô hình huấn luyện lại trên full dataset sau khi thêm track mới:
  - pairwise accuracy `85.36%`
  - `FRR = 73.65%`

Các kết quả này cho thấy:

- mô hình đã học được cấu trúc phân lớp cơ bản
- `clean-core` giúp cải thiện nhẹ verification
- tuy nhiên khả năng giữ gần các cặp ảnh cùng người giữa các điều kiện khó vẫn còn hạn chế

## 4. Hạn chế

Mặc dù pipeline hiện tại đã hoàn thiện đáng kể về mặt kỹ thuật, hệ thống vẫn còn một số hạn chế quan trọng:

### 4.1. Thiếu face-domain pretraining tương thích

Checkpoint pretrained hiện có trong workspace chưa tương thích với kiến trúc student và teacher đang dùng. Vì vậy:

- chưa thể tận dụng đúng pretrained transfer learning theo chuẩn face recognition
- nhánh KD chưa đạt hiệu quả thực nghiệm

### 4.2. Dữ liệu nội bộ còn nhỏ và khó

Mặc dù số lượng ảnh đã tăng lên, dữ liệu nội bộ vẫn bị giới hạn ở:

- ít danh tính
- số phiên ghi hình chưa nhiều
- biến thiên nội lớp lớn khi thêm dữ liệu khó từ video nhóm

Điều này làm pairwise verification chưa tăng tương xứng dù validation classification có cải thiện.

### 4.3. Chưa hoàn thiện giai đoạn pruning và recovery

Các script cho pruning và fine-tuning recovery đã có, nhưng:

- chưa có baseline đủ mạnh để bước sang pruning
- chưa có chuỗi thực nghiệm đầy đủ `prune -> recover -> evaluate` cho kết quả tốt hơn baseline

### 4.4. Chưa hoàn thiện giai đoạn export và triển khai

Hệ thống hiện mới dừng ở mức:

- đo độ phức tạp mô hình
- đo latency / FPS

Chưa hoàn thiện:

- export sang định dạng triển khai cuối
- benchmark thực địa trên thiết bị mục tiêu
- kiểm thử end-to-end trong pipeline triển khai thực

## 5. Hướng phát triển tiếp theo

Hướng phát triển phù hợp nhất với trạng thái hiện tại của pipeline là:

### 5.1. Public face-domain pretraining

Thay vì tiếp tục tối ưu cục bộ trên bộ dữ liệu nhỏ, bước kế tiếp nên là:

- pretrain trực tiếp student backbone trên một public face dataset lớn trên Colab
- vẫn giữ kiến trúc `widened`
- dùng `AdaFace-only`
- không dùng KD ở giai đoạn này

Mục tiêu là giúp student học embedding khuôn mặt tổng quát trước khi quay về domain sinh viên nội bộ.

### 5.2. Clean-core finetuning

Sau khi có checkpoint public-pretrained, mô hình sẽ được finetune trên:

- `2_face_dataset_clean_core_split`

Giai đoạn này giúp mô hình học identity cục bộ từ các mẫu sạch nhất, giảm nhiễu và ổn định không gian embedding.

### 5.3. Full-dataset finetuning

Checkpoint tốt nhất từ clean-core sẽ tiếp tục được finetune trên:

- `2_face_dataset_split`

Giai đoạn này nhằm tăng robustness với:

- motion blur
- pose variation
- lighting variation
- các tình huống gần với môi trường triển khai thực tế

### 5.4. Chỉ thử lại KD sau khi đã có baseline mạnh hơn

KD hiện chưa phải là hướng ưu tiên. Chỉ nên thử lại khi:

- đã có teacher tương thích thật
- hoặc đã có một student pretrained đủ mạnh để làm baseline

### 5.5. Pruning và deployment sau cùng

Sau khi mô hình nội bộ đạt verification mạnh hơn, các bước tiếp theo sẽ là:

- pruning có kiểm soát
- fine-tuning recovery
- export mô hình
- benchmark triển khai trên thiết bị đích

## Kết luận chung

Hệ thống hiện tại đã hoàn thành tốt phần tái cấu trúc backbone, tổ chức lại dữ liệu, và xây dựng được một baseline AdaFace ổn định. Tuy nhiên, hiệu năng verification vẫn chưa đạt ngưỡng mục tiêu do mô hình còn thiếu face-domain pretraining đủ mạnh và dữ liệu nội bộ chưa đủ để bù đắp hoàn toàn cho khoảng trống đó.

Do đó, hướng đi hợp lý nhất ở giai đoạn kế tiếp là:

- `public pretraining -> clean-core finetuning -> full-dataset finetuning`

Đây là hướng vừa phù hợp với kiến trúc và codebase hiện tại, vừa có xác suất cải thiện verification cao hơn so với việc tiếp tục ưu tiên KD hoặc pruning ngay ở thời điểm này.
