# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Qua bốn phản hồi, có thể thấy khi temperature thấp (0.0), mô hình tạo ra câu trả lời ổn định, nhất quán và ít thay đổi giữa các lần gọi. Khi temperature tăng (0.5, 1.0 và 1.5), mô hình có xu hướng đa dạng hơn trong cách diễn đạt và lựa chọn thông tin, mặc dù trong trường hợp này các phản hồi vẫn xoay quanh những sự thật phổ biến về Việt Nam như Hang Sơn Đoòng hoặc ngành cà phê. Nói chung, temperature càng cao thì mức độ sáng tạo và ngẫu nhiên của câu trả lời càng lớn.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Tôi sẽ chọn temperature = 0.2 (hoặc trong khoảng 0.0–0.3) cho chatbot hỗ trợ khách hàng. Mức temperature thấp giúp câu trả lời nhất quán, chính xác và ít mang tính ngẫu nhiên, từ đó giảm nguy cơ cung cấp thông tin sai lệch hoặc trả lời không đúng trọng tâm. Điều này đặc biệt quan trọng đối với chatbot hỗ trợ khách hàng, nơi người dùng mong đợi các câu trả lời đáng tin cậy và thống nhất.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> *Với workload:

Số người dùng mỗi ngày: 10.000
Mỗi người gọi API: 3 lần/ngày
Tổng số lượt gọi:
10.000×3=30.000 requests/ngày
Mỗi request có trung bình 350 token output

Giả sử chỉ tính chi phí output token:

GPT-4o:

30.000×350=10.500.000 tokens
1000
10.500.000
	​

×0.010=105 USD/ngày

GPT-4o-mini:

1000
10.500.000
	​

×0.0006=6.3 USD/ngày
Tỷ lệ chi phí: 105 / 6.3 ≈ 16.7 lần

Như vậy, GPT-4o đắt hơn GPT-4o-mini khoảng 16–17 lần cho workload này (chưa tính chi phí input token).

GPT-4o xứng đáng sử dụng trong các trường hợp yêu cầu chất lượng cao như trợ lý chuyên gia, phân tích tài liệu phức tạp, hỗ trợ lập trình hoặc các tác vụ cần khả năng suy luận tốt và độ chính xác cao. Ngược lại, GPT-4o-mini phù hợp cho chatbot chăm sóc khách hàng phổ thông, trả lời câu hỏi thường gặp hoặc các tác vụ có số lượng request lớn vì chi phí thấp hơn nhiều nhưng vẫn đáp ứng tốt nhu cầu cơ bản.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Hai phản hồi có sự khác biệt rõ rệt về cách diễn đạt, độ phức tạp và lựa chọn từ vựng. Với persona giáo viên tiểu học, mô hình sử dụng ngôn ngữ đơn giản, ví dụ gần gũi như “chuỗi các hộp” hoặc “cuốn sổ lớn”, giúp trẻ em dễ hình dung khái niệm blockchain. Ngược lại, với persona chuyên gia tài chính, mô hình trả lời dài hơn, sử dụng nhiều thuật ngữ chuyên ngành như “cơ sở dữ liệu phân tán”, “hàm băm mật mã”, “phi tập trung” và “bất biến”. Điều này cho thấy system prompt có ảnh hưởng trực tiếp đến vai trò, phong cách, mức độ chi tiết và cách lựa chọn kiến thức của mô hình khi tạo câu trả lời.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> *148 so với ~134, chênh nhau 10%, do tiếng Việt có nhiều dấu thanh, cần nhiều hơn 1 token để biểu diễn*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming quan trọng nhất trong các trường hợp cần phản hồi nhanh và tạo trải nghiệm tương tác theo thời gian thực, ví dụ như chatbot, trợ lý ảo, hỗ trợ khách hàng hoặc các ứng dụng yêu cầu người dùng thấy kết quả ngay khi mô hình đang xử lý. Việc hiển thị từng phần nội dung giúp giảm cảm giác chờ đợi, đặc biệt với những câu trả lời dài. Tuy nhiên, non-streaming phù hợp hơn khi cần xử lý các tác vụ yêu cầu nhận toàn bộ kết quả trước khi sử dụng, chẳng hạn như tạo báo cáo, phân tích dữ liệu, xuất file hoặc các quy trình tự động cần kiểm tra toàn bộ nội dung đầu ra. Do đó, việc lựa chọn streaming hay non-streaming phụ thuộc vào yêu cầu về tốc độ phản hồi và cách ứng dụng sử dụng kết quả.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Exponential backoff giúp hệ thống xử lý lỗi tốt hơn bằng cách tăng dần thời gian chờ giữa các lần retry (ví dụ: 1s, 2s, 4s, 8s), từ đó giảm áp lực lên API khi đang bị quá tải và cho hệ thống có thời gian phục hồi. So với delay cố định, cách này hạn chế việc gửi quá nhiều request liên tục trong thời gian ngắn. Nếu hàng nghìn client cùng retry với delay cố định giống nhau, chúng có thể tạo ra hiện tượng thundering herd, khi tất cả client gửi lại request cùng một thời điểm, khiến API tiếp tục bị quá tải hoặc xảy ra lỗi lặp lại. Exponential backoff kết hợp thêm random jitter giúp phân tán các lần retry và cải thiện độ ổn định của hệ thống.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *Bạn là một trợ lý AI chuyên hỗ trợ lập trình và nghiên cứu công nghệ. 
Hãy trả lời bằng tiếng Việt, giải thích rõ ràng, có cấu trúc và ưu tiên các ví dụ thực tế. 
Khi giải thích vấn đề kỹ thuật, hãy trình bày từ kiến thức cơ bản đến nâng cao, tránh sử dụng thuật ngữ phức tạp nếu chưa giải thích. 
Đưa ra câu trả lời chính xác, ngắn gọn và tập trung vào vấn đề người dùng đang hỏi.
Tôi lựa chọn persona trợ lý AI hỗ trợ lập trình và nghiên cứu công nghệ vì đây là vai trò cần khả năng giải thích kiến thức kỹ thuật một cách dễ hiểu và có hệ thống. Cụm từ “trả lời bằng tiếng Việt” giúp đảm bảo mô hình sử dụng ngôn ngữ phù hợp với người dùng, trong khi yêu cầu “từ kiến thức cơ bản đến nâng cao” giúp câu trả lời có tính hướng dẫn, phù hợp cả với người mới bắt đầu và người đã có nền tảng. Cụm “tập trung vào vấn đề người dùng đang hỏi” giúp giảm các nội dung lan man và cải thiện trải nghiệm sử dụng.*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Hạn chế lớn nhất của trợ lý hiện tại là không có bộ nhớ dài hạn, nên mô hình chỉ có thể dựa vào nội dung trong phiên trò chuyện hiện tại và không ghi nhớ được sở thích, kiến thức nền hoặc các thông tin quan trọng của người dùng từ những lần sử dụng trước. Một cải thiện cụ thể là triển khai hệ thống memory kết hợp với cơ sở dữ liệu vector (Vector Database). Khi người dùng cung cấp thông tin quan trọng, hệ thống có thể lưu trữ và mã hóa dữ liệu thành vector bằng embedding model, sau đó truy xuất các thông tin liên quan ở những lần trò chuyện sau thông qua kỹ thuật RAG (Retrieval-Augmented Generation). Điều này giúp trợ lý đưa ra câu trả lời cá nhân hóa hơn mà vẫn kiểm soát được dữ liệu được lưu trữ.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
