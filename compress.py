import google.generativeai as genai
import os

try:
    genai.configure(api_key="XXX")
except AttributeError:
    print("API FAILED.")
    exit()

model = genai.GenerativeModel('gemini-1.5-flash-latest')

prompt_template = """
Bạn là một chuyên gia phân tích và tóm tắt văn bản. Tôi sẽ cung cấp cho bạn nội dung của một trang sách.

Nhiệm vụ của bạn là đọc kỹ và tạo ra một bản tóm tắt đáp ứng các yêu cầu sau:

1.  **Dễ hiểu:** Sử dụng ngôn ngữ đơn giản, rõ ràng, phù hợp với mọi đối tượng độc giả, tránh các thuật ngữ quá chuyên môn hoặc diễn giải chúng nếu bắt buộc phải có.
2.  **Đầy đủ:** Không bỏ sót bất kỳ ý chính, luận điểm quan trọng, sự kiện then chốt hay thông tin cốt lõi nào có trong văn bản gốc.
3.  **Súc tích:** Tóm tắt cần ngắn gọn nhưng vẫn truyền tải đủ thông điệp. Giữ lại những chi tiết quan trọng nhất và loại bỏ những thông tin phụ, không cần thiết.
4.  **Trung thành:** Bản tóm tắt phải phản ánh chính xác nội dung và tinh thần của văn bản gốc, không thêm vào những suy diễn hay bình luận cá nhân không có trong văn bản.

Bây giờ, hãy tóm tắt nội dung sau:

---
{text_content}
---
"""

def compress(content) -> str:
    full_prompt = prompt_template.format(text_content=content)
    print("Sending...")
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        print(f"API failed {e}")
        return None
    
if __name__ == "__main__":
    print("lmao lmao")