# Reflection — Day 18 Lab: Data Lakehouse Architecture

**Họ tên:** Hoàng Tuấn Anh
**MSSV:** 2A202600075

---

## Anti-pattern dễ mắc nhất: Small-File Problem

Trong thực tế, dự án RAG cá nhân mà em đang xây dựng cần liên tục nhận log từ API theo từng request nhỏ lẻ (event-driven ingestion). Đây chính là môi trường lý tưởng để mắc phải anti-pattern **Small-File Problem**.

Vấn đề xảy ra khi mỗi lần có một sự kiện mới (một tài liệu upload, một lần retrieval) là hệ thống lại ghi ra một file Parquet siêu nhỏ. Sau vài ngày vận hành, số lượng file có thể lên đến hàng nghìn, khiến mọi truy vấn phải mở từng file một để đọc — hiệu năng tụt thảm hại. Trong bài lab NB2, em đã tự tay tái hiện lại cảnh này bằng cách append 200 batch liên tục và đo được sự chênh lệch tốc độ truy vấn cực lớn trước và sau khi chạy `OPTIMIZE + ZORDER`.

Insight quan trọng nhất em mang về là: **hiệu năng không tự dưng có được**. Dữ liệu nếu không được `OPTIMIZE` và `Z-ORDER` theo đúng pattern truy vấn, sẽ trở thành "dữ liệu chết" dù về mặt kỹ thuật vẫn đang tồn tại trên ổ đĩa. Đây là bài học đắt giá mà em rút ra được để tránh hệ thống bị sập khi đưa lên môi trường thực tế. Để chứng minh việc đã làm chủ kiến thức này, trong phần **Bonus Challenge (Multimodal RAG PoC)**, em đã chủ động thiết kế lịch chạy `OPTIMIZE + ZORDER` tự động hàng ngày cho các layer Silver và Gold, giải quyết triệt để rủi ro nghẽn cổ chai ngay từ khâu thiết kế kiến trúc.

---

