- naive bayes
- preprocessing
- bag of words & matrix
- build model
- caculate accuracy
---
# Naive Bayes và bài toán phân loại văn bản

Với sự xuất hiện của những chiên gia trí tuệ nhân tạo chém gió như rồng hiện nay thì machine learning hẳn không còn xa lạ với chúng ta. Máy học đã và đang được dùng trong rất nhiều các lĩnh vực, gần gũi và dễ bắt gặp nhất là các bài toán xác định email rác, phân loại bài viết, hỗ trợ ra quyết định,...

Ở đây, mình sẽ sử dụng thuật toán đơn giản là Naive Bayes đế phân loại (gán nhãn) các bài báo tiếng Việt trong 5 lĩnh vực: Đời sống, Kinh doanh, Sức khỏe, Thể thao và Văn hóa.

# Suy luận Bayes

## Suy luận bằng xác suất

[cat-image]

Trước tiên mình cần bạn trả lời: làm sao bạn biết con vật phía trên là **mèo** mà không phải là chó? Ria trắng, tròng đen trong mắt nhỏ,...

Tất nhiên những đặc điểm kia chỉ là tương đối và chúng ta không phải dân Computer Vision nên mình sẽ biểu diễn dưới dạng xác suất thay vì mô tả. Dưới đây là tỉ lệ các đặc điểm thuộc về mỗi loài (theo mình đoán):

| Loài | Kêu "meo" | Mũi đen |
| ---- | --------- | ------- |
| Mèo  | 0.99      | 0.2     |
| Chó  | 0.0001    | 0.98    |

Giả sử xác suất ta gặp mèo và chó ngoài đường là ngang nhau và bằng 0.5. Lần này, ta ra đường và gặp một con vật kêu "meo" và có mũi đen thì khả năng nó là mèo là: 0.5 x 0.99 x 0.2 = 0.099; khả năng nó là chó là: 0.5 x 0.0001 x 0.98 = 0.000049.

OK. Dựa vào kết quả trên thì ta nói nó là mèo. Và vì đặc điểm quá rõ nên ta dễ dàng khẳng định. Nhưng đau lòng trong thực tế thì không phải lúc nào ta cũng có dữ liệu đẹp và dễ như thế, ví dụ như phân loại hàng của Nhật và hàng của China chẳng hạn.

## Công thức Bayes

Ở trên, xác suất ta gặp chó và mèo ngoài đường (0.5) được gọi là xác suất tiên nghiệm (đã biết trước), và kết quả ta tính là xác suất hậu nghiệm. Gọi C là tập các xác suất của nhãn (tiên nghiệm), X là tập xác suất của các đặc điểm (feature), công thức Bayes là:

| cha | tốt | con | chẳng | đói | meo |
| --- | --- | --- | ----- | --- | --- |
| 1   | 1   | 1   | 1     | 1   | 1   |
| 0   | 1   | 0   | 1     | 0   | 0   |
