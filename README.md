# Machine Learning là cái gì?

Để tránh làm như giáo viên dạy văn, tôi sẽ không nói tất tần tật về machine learning \(ML\), mà thay vào đó, ta sẽ bắt đầu bằng ví dụ.

Có bao giờ bạn tự nghĩ tại sao các cụ ngày xưa có mấy câu tục ngữ như _"trăng quầng thì hạn, trăng tán thì mưa"_ hay _"chuồn chuồn bay thấp thì mưa"_ không? Hay là tự hỏi tại sao Facebook biết trong ảnh có người yêu cũ \(nyc, xin lỗi vì chạm vào nỗi đau của bạn\) của bạn mà gợi ý bạn tag nó vào?

Trong câu hỏi đầu tiên có lẽ bạn đã biết được lý do rồi, là bởi vì người xưa quan sát thấy các hiện tượng đó lặp đi lặp lại, thế hệ này sang thế hệ khác dần dần đúc kết lại thành quy luật. Còn nhận dạng, bạn có thể hiểu theo cách tương tự là Facebook đã có nhiều ảnh sống ảo của nyc, việc nó cần làm là dùng những ảnh đó để rút ra đặc điểm và dựa trên đó để xác định có phải nyc hay không.

Bây giờ, ta sẽ phân tích ví dụ một chút, lấy luôn câu_"chuồn chuồn bay thấp thì mưa"_ . Ta thấy câu đó là một cặp **nguyên nhân - kết quả.** Nguyên nhân ở đây là _chuồn chuồn_, kết quả là _mưa_, và tất nhiên cần thêm điều kiện là _bay thấp_ nữa. Kể cả nhận dạng nyc cũng thuộc loại này. Trong toán học, chúng ta gọi nguyên nhân trên là điều kiện cần và điều kiện trên là điều kiện đủ. Nhưng khoan, làm sao Facebook có thể rút ra được những đặc điểm trên mặt nyc, khi nyc post ảnh nó có miêu tả mặt nó đâu? Tôi có thể trả lời ngắn gọn cho bạn là nếu Facebook thấy body ba vòng như một của nyc của bạn đến phát ngán thì nó sẽ ấn tượng đến mức biết đó là đặc trưng của nyc bạn :\) bạn cứ suy luận tương tự với các bộ phận khác.

Thế chốt lại ML là cái gì? Thật sự có rất nhiều định nghĩa, nhưng theo tôi, _**ML là tập hợp các kỹ thuật tính toán dựa trên kinh nghiệm để dự đoán hoặc cải thiện độ chính xác.**_ Ví dụ, những lần quan sát hiện tượng hay những bức ảnh của nyc chính là kinh nghiệm để máy móc có thể dự đoán trời mưa hoặc để càng ngày biết càng rõ nyc trông như thế thế nào. Và hai ví dụ nêu trên chính là đại diện của hai nhánh lớn trong ML mà ta sắp tìm hiểu dưới đây.

## Supervised learning \(học có hướng dẫn\) <a id="supervised-learning-hoc-co-huong-dan"></a>

Đây chính là ví dụ chuồn chuồn. Ta quay lại với bộ ba nguyên nhân - điều kiện - kết quả dưới hình thức phương trình toán học:

$$
ketqua = dieukien(nguyennhan)\\
hay \ \ Y = f(X)
$$

$$X $$là đầu vào, là dữ liệu, là kinh nghiệm để máy học; $$Y$$ là đầu ra, là kết quả đã được mặc định với đầu vào đó; hàm $$f$$ là hàm biểu diễn mối quan hệ giữa hai biến trên, giống như cái máy xay hạt cà phê thành bột ấy. Triết học một chút, nguyên nhân khi gặp một điều kiện sẽ sinh ra kết quả.

Nhiệm vụ của máy trong trường hợp này là tìm ra hàm $$f$$ đó, với độ chính xác cao nhất có thể \(bởi vì dữ liệu đời thật nó không có ngon như lý thuyết đâu\).

Tôi gọi đây là học có hướng dẫn thay vì gọi là học có giám sát như phần đông vì ta cho máy biết chính xác đầu ra và máy phải học dựa trên nó. Tất nhiên là cả hai cách gọi đều không sai.

## Unsupervised learning \(học không có hướng dẫn\) <a id="unsupervised-learning-hoc-khong-co-huong-dan"></a>

