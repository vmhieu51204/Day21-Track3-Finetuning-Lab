# Lab 21 — Evaluation Report

**Họ tên**: Học viên AICB  **MSSV**: AICB-2026-T3  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `NVIDIA Tesla T4 16GB (Google Colab)`

> Mọi con số dưới đây khớp 100% với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (intent, urgency, product, sentiment) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json; suggested_max_length=256)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json: verdict = "reasoning preserved — safe to train on traces")*.
Nếu không: Chat template của Qwen3.5 hỗ trợ và giữ nguyên khối thẻ `<think>`, không làm mất cấu trúc suy luận.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3134.8 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1021.0 |
| (c) LoRA fine-tune | 0.970 | 0.678 | 1.000 | 1344.2 |

**(b) có thật sự mạnh hơn (a) không?** Có — Baseline (b) đạt target accuracy 0.765 và format 1.000, vượt trội hoàn toàn so với Baseline (a) (target 0.000, format 0.000). Naive prompt không hướng dẫn schema JSON khiến base model sinh tự do và hỏng format hoàn toàn.
Bạn có sửa `OPTIMIZED_PROMPT` không? Không sửa (SHA `719e74d3b6232053` khớp tuyệt đối với bản gốc được cung cấp), đảm bảo tính công bằng và khách quan của phép đo đối chuẩn.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6269 | 0.970 | 385.9 | 8.78 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 0.0001 | 0.5366 | 0.970 | 263.8 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | 0.000 | 393.7 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.940 | 463.1 | 3.86 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Run `attn_only` có 32,456,704 tham số huấn luyện (sai lệch < 0.03% so với 32,464,896 của `correct`). Trên tập target, `attn_only` và `correct` hoà nhau với cùng số điểm 0.970 (97.0%). Tuy nhiên, thứ tự theo train loss lại khác: `attn_only` có train loss thấp hơn (0.5366 so với 0.6269 của `correct`). Việc dồn toàn bộ ngân sách tham số vào ma trận attention với rank cực cao (r=283) giúp mô hình dễ dàng ép thấp loss trên tập huấn luyện hẹp, nhưng không mang lại sự vượt trội trên downstream target so với việc phân bổ adapter dàn trải đều ra toàn bộ các lớp tuyến tính (`text-linear` với r=16 khiêm tốn). Điều này khẳng định rằng vị trí gắn adapter phân bổ rộng mới là yếu tố quyết định khả năng tổng quát hoá, còn việc cố tình tăng rank cục bộ chỉ dẫn đến nguy cơ ghi nhớ vẹt (overfitting).

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Run `wrong_lr` chỉ khác đúng LR khi áp dụng thang đo của Full Fine-Tuning (1e-5 thay vì 1e-4 của LoRA). Đường train loss của `wrong_lr` gần như đi ngang và dừng ở mức 1.5702 (gần gấp 2.5 lần so với `correct`), dẫn đến target và format hoàn toàn bằng 0.000 sau 30 steps. Nếu chỉ nhìn vào loss cao mà không biết nguyên nhân do LR quá nhỏ, người làm thí nghiệm sẽ dễ dàng kết luận sai lầm rằng dữ liệu huấn luyện quá phức tạp, kiến trúc mô hình không phù hợp, hoặc LoRA không thể giải quyết bài toán phân loại JSON. Thực chất, vì trọng số cơ sở bị đóng băng và adapter được khởi tạo gần bằng 0, LoRA bắt buộc phải cần learning rate lớn hơn 10x–20x so với Full FT để cập nhật kịp bước nhảy trong không gian tham số.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
Run `qlora` tiết kiệm được 4.92 GB VRAM (chỉ chiếm 3.86 GB so với 8.78 GB của `correct`, giảm ~56% bộ nhớ đồ hoạ). Tuy nhiên, cái giá phải trả rất rõ ràng: thời gian huấn luyện tăng từ 385.9s lên 463.1s (~20% overhead dequantization), độ trễ suy luận tăng từ 1344.2 ms lên 1716.7 ms, và quan trọng nhất là target accuracy bị tụt từ 0.970 xuống 0.940. Kết quả thực nghiệm này hoàn toàn ủng hộ khuyến nghị từ phía nhà phát triển Qwen3.5: sai số lượng tử hoá 4-bit gây suy giảm năng lực biểu diễn không đáng có, và trên phần cứng có đủ VRAM (như T4 16GB) thì fp16/bf16 LoRA luôn là lựa chọn tối ưu hơn hẳn QLoRA.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.113` · `valid_trace_rate = 0.0`

**Diễn giải:** Cổng hồi quy đưa ra phán quyết FAILED do năng lực tổng quát của mô hình bị suy thoái -11.3% (điểm regression giảm từ 0.7911 ở baseline gốc xuống còn 0.6778 ở bản fine-tune), vượt xa ngưỡng dung sai an toàn cho phép là 0.020 (2.0%). Mặc dù bản fine-tune mang lại bước tiến nhảy vọt trên nhiệm vụ mục tiêu chuyên biệt (target tăng +20.5%, đạt 97.0% accuracy và 100% format JSON), mô hình đã gặp phải hiện tượng quên thảm hoạ (catastrophic forgetting). Toàn bộ 225 mẫu huấn luyện chỉ đơn thuần là các cặp hội thoại trả về JSON phân loại ticket mà hoàn toàn thiếu vắng các văn bản hội thoại, kiến thức tổng quát và hướng dẫn thông thường. Việc đánh giá qua cổng hồi quy 4 nhóm đã bóc trần sự thật rằng: tối ưu hoá đơn lẻ cho một downstream task mà không có cơ chế bảo toàn kiến thức nền tảng sẽ làm hỏng năng lực cốt lõi của LLM. Theo lý thuyết tại deck bài giảng §6.3, để khắc phục và chuyển phán quyết thành PASSED trong thực tế, chúng ta bắt buộc phải trộn thêm 1–5% dữ liệu hội thoại/chỉ dẫn phổ thông (replay data) vào tập huấn luyện.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây VN232232. Cho tôi trả lại. Gấp... | doi_tra, cao, chuột không dây, tich_cuc | Sai định dạng trích xuất product | doi_tra, cao, chuột không dây, tich_cuc | ✅ FT thắng: Trích xuất chính xác 4 trường chuẩn JSON |
| 2 | Xin chào, mình đặt đèn bàn LED VN880807. Hoàn tiền. Quá hạn rồi... | hoan_tien, cao, đèn bàn LED, tich_cuc | hoan_tien, trung_binh, đèn bàn LED, tich_cuc | hoan_tien, cao, đèn bàn LED, tich_cuc | ✅ FT thắng: FT nhận diện đúng mức urgency "cao" |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt VN804124. Chưa thấy tiền. Khi nào tiện... | hoan_tien, thap, bình giữ nhiệt, tich_cuc | hoan_tien, thap, bình giữ nhiệt, tich_cuc | hoan_tien, trung_binh, bình giữ nhiệt, tich_cuc | ❌ **FT thua**: FT nhầm urgency "trung_binh" thay vì "thap" |
| 4 | Shop ơi, mình đặt nồi chiên không dầu DH249548. Thiếu phụ kiện. Khi nào tiện... | san_pham_loi, thap, nồi chiên không dầu, trung_tinh | san_pham_loi, thap, nồi chiên không dầu, trung_tinh | san_pham_loi, trung_binh, nồi chiên không dầu, trung_tinh | ❌ **FT thua**: FT nhầm urgency "trung_binh" khi gặp cụm "Khi nào tiện" |
| 5 | Shop ơi, mình đặt áo khoác gió VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop... | san_pham_loi, thap, áo khoác gió, tich_cuc | san_pham_loi, thap, áo khoác gió, tich_cuc | san_pham_loi, trung_binh, áo khoác gió, tich_cuc | ❌ **FT thua**: Lặp lại lỗi dự đoán thiên lệch độ khẩn cấp |

**Có mẫu chung nào ở các ca FT thua không?**
Có một mẫu hình sai số mang tính hệ thống cực kỳ rõ nét: Ở tất cả các ca FT thua (điểm 0.75/1.0), nhãn thực tế đều có trường `urgency: "thap"` đi kèm với ngữ cảnh ngôn từ "Khi nào tiện". Trong khi baseline (b) với prompt chi tiết diễn giải đúng quy tắc xếp mức độ khẩn cấp, bản fine-tune lại bị thiên lệch (bias) dự đoán thành `"urgency": "trung_binh"`. Điều này xảy ra do sự mất cân bằng phân phối nhãn trong tập dữ liệu huấn luyện hoặc thiếu các ví dụ đối trọng rõ ràng cho cụm từ "Khi nào tiện", khiến mô hình có xu hướng rơi vào nhãn chiếm đa số an toàn là "trung_binh".

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ):** 
Có nên deploy bản fine-tune này vào môi trường thực tế hay không? Câu trả lời là: **Chưa thể deploy như một mô hình độc lập đa năng, nhưng hoàn toàn có thể deploy dưới dạng một microservice routing chuyên trách (isolated JSON classification worker)**. Nếu đặt mô hình ở vị trí tiếp xúc trực tiếp với người dùng cuối, sự suy thoái -11.3% ở năng lực tổng quát sẽ dẫn đến hành vi bất thường khi người dùng hỏi các câu hỏi ngoài phạm vi ticket CSKH. Tuy nhiên, với độ chính xác mục tiêu 97.0% và khả năng tuân thủ định dạng JSON đạt tuyệt đối 100% cùng độ trễ ổn định, bản fine-tune này vượt trội hoàn toàn so với base model prompt thủ công (chỉ đạt 76.5%). Đòn bẩy cốt lõi thực sự trong lab này được xác định không phải là việc cố gắng tăng rank ma trận lên mức cực đại (`attn_only` r=283 không thắng được `correct` r=16), mà nằm ở ba yếu tố quyết định: (1) thiết lập learning rate đúng tầm vĩ mô cho LoRA (1e-4 thay vì 1e-5), (2) phân bổ adapter phủ đều toàn bộ các lớp tuyến tính (`text-linear`), và (3) tính chuẩn xác tuyệt đối của loss mask nhằm ngăn chặn việc lãng phí gradient vào việc học lại câu hỏi.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Loss mask là tiền đề sống còn:** Nếu không kiểm chứng chặt chẽ bằng cách decode ngược tokens trong mask proof, việc để lọt câu hỏi vào phạm vi tính loss sẽ làm lãng phí dung lượng cập nhật của adapter và khiến mô hình sinh vẹt prompt thay vì học logic trả lời.
2. **Không bao giờ dùng train loss làm thước đo phán quyết:** Run `attn_only` có train loss 0.5366 thấp hơn hẳn `correct` (0.6269) nhưng điểm target thực tế lại chỉ hoà (0.970). Đánh giá dựa trên chỉ số đại diện (surrogate metric) thay vì downstream task thực sự là sai lầm nguy hiểm nhất trong ML.
3. **Ý nghĩa của cổng hồi quy và tính liêm chính khoa học:** Một kết quả FAILED khi kiểm tra hồi quy không phải là thất bại của thí nghiệm, mà là sự phát hiện chuẩn xác hiện tượng catastrophic forgetting, chỉ ra chính xác nhu cầu cần trộn replay data trước khi đưa mô hình ra thực tế.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Tôi sẽ trộn 3%–5% tập dữ liệu hướng dẫn tổng quát (General Instruction Replay Dataset) vào tập train để chạy lại NB3 và NB5 nhằm triệt tiêu hiện tượng quên thảm hoạ, đưa `regression Δ` về dưới ngưỡng 0.020 để vượt qua cổng hồi quy (PASSED).

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link: https://huggingface.co/DarkinH/qwen3.5-4b-lora-cskh-lab21
