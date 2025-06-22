import google.generativeai as genai

# ----- Config Google Gemini API -----
try:
    genai.configure(api_key="API-KEY-HERE")
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    print(f"Lỗi khởi tạo Gemini: {e}")
    model = None

# ----- Prompt Template -----
prompt_template = """
Bạn là một chuyên gia phân tích và tóm tắt văn bản. Tôi sẽ cung cấp cho bạn nội dung của một trang sách.

Nhiệm vụ của bạn là đọc kỹ và tạo ra một bản tóm tắt đáp ứng các yêu cầu sau:

1.  **Dễ hiểu:** Sử dụng ngôn ngữ đơn giản, rõ ràng, phù hợp với mọi đối tượng độc giả.
2.  **Đầy đủ:** Không bỏ sót bất kỳ ý chính, luận điểm quan trọng, sự kiện then chốt nào.
3.  **Súc tích:** Ngắn gọn nhưng vẫn truyền tải đủ thông điệp.
4.  **Trung thành:** Phản ánh chính xác nội dung gốc, không thêm suy diễn cá nhân.

Chú ý: đừng thêm ký tự đặc biệt

Bây giờ, hãy tóm tắt nội dung sau:

---
{text_content}
---
"""

def compress(content):
        
    full_prompt = prompt_template.format(text_content=content)
    print("Sending")
    try:
        response = model.generate_content(full_prompt)
        summary = response.text
        print(summary)
        return summary
    except Exception as e:
        print(f"Failed {e}")
        return None
