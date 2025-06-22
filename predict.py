import os
import threading
import time
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import pygame
from google.cloud import vision
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer

# ----- Import hàm compress từ file gemini_handler.py -----
from compress import compress

# -------------------- Setup ----------------------
KEY_PATH = "key.json"
if not os.path.exists(KEY_PATH):
    raise FileNotFoundError(f"Google Vision API key not found at: {KEY_PATH}")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
vision_client = vision.ImageAnnotatorClient()
os.makedirs("./screenshot", exist_ok=True)

# -------------------- App State ----------------------
class AppState:
    def __init__(self):
        self.localIP = "192.168.1.50"
        self.video = cv2.VideoCapture(f"http://{self.localIP}:81/stream")
        self.latest_frame = None
        self.cached_data = []
        self.stop_event = threading.Event()
        self.isSpeaking = False
        self.last_text = ""
        self.isReading = False
        self.isOcr = False

# -------------------- TTS ----------------------
def speak_vietnamese(text, state):
    pygame.mixer.init()
    def run_tts():
        try:
            state.isSpeaking = True
            tts = gTTS(text=text, lang='vi')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
            pygame.mixer.music.load(fp.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.quit()
            state.isSpeaking = False
        except Exception as e:
            print("TTS Error:", e)
            state.isSpeaking = False
    threading.Thread(target=run_tts, daemon=True).start()

# Hàm TTS riêng để đọc bản tóm tắt
def speak_summary(summary_text, state):
    full_text_to_speak = "Đây là bản tóm tắt nội dung: " + summary_text
    speak_vietnamese(full_text_to_speak, state)

# -------------------- MediaPipe Hand Detection (optimized) ----------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils

def hand_handler(frame, state):
    h, w = frame.shape[:2]
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, landmarks, mpHands.HAND_CONNECTIONS)
            index_finger = landmarks.landmark[8]
            px, py = int(index_finger.x * w), int(index_finger.y * h)
            for box, text in state.cached_data:
                (x1, y1), (x2, y2) = box
                if x1 <= px <= x2 and y1 <= py <= y2 and text != state.last_text and not state.isSpeaking:
                    state.last_text = text
                    speak_vietnamese(text, state)
                    return # Thoát sớm khi tìm thấy

# -------------------- OCR via Google Vision ----------------------
def predict(state):
    def run_predict():
        state.isOcr = True
        state.isReading = True
        speak_vietnamese("Đang chạy máy đọc", state)
        time.sleep(1)
        frame = state.latest_frame
        if frame is None:
            state.isOcr = state.isReading = False
            return

        img_path = "./screenshot/cap.jpg"
        cv2.imwrite(img_path, frame)
        with open(img_path, "rb") as image_file:
            image = vision.Image(content=image_file.read())
        
        response = vision_client.document_text_detection(image=image)
        if not response.full_text_annotation or not response.full_text_annotation.text.strip():
            speak_vietnamese("Không tìm thấy chữ", state)
            state.isOcr = state.isReading = False
            return

        state.cached_data = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for para in block.paragraphs:
                    paragraph_text = ""
                    for word in para.words:
                        word_text = ''.join([symbol.text for symbol in word.symbols])
                        paragraph_text += word_text + (" " if word.symbols[-1].property.detected_break.type in [1,3,5] else "")
                    
                    paragraph_text = paragraph_text.strip()
                    if paragraph_text:
                        box = para.bounding_box.vertices
                        x_coords = [v.x for v in box]
                        y_coords = [v.y for v in box]
                        state.cached_data.append([((min(x_coords), min(y_coords)), (max(x_coords), max(y_coords))), paragraph_text])

        speak_vietnamese("Máy đọc đã chạy xong, bạn có thể chỉ tay để đọc", state)
        
        img = cv2.imread(img_path)
        for box, _ in state.cached_data:
            cv2.rectangle(img, box[0], box[1], (0,255,0), 2)
        cv2.imwrite("./screenshot/output.jpg", img)

        state.isOcr = state.isReading = False
    threading.Thread(target=run_predict, daemon=True).start()

# -------------------- Luồng xử lý tóm tắt (ĐÃ THÊM LẠI) ----------------------
def run_compress_workflow(state):
    def process():
        if state.isReading or state.isOcr:
            return
        state.isReading = True
        
        speak_vietnamese("Bắt đầu quét văn bản để tóm tắt", state)
        frame = state.latest_frame
        if frame is None:
            state.isReading = False
            return
        
        # Chạy OCR
        _, img_encoded = cv2.imencode('.jpg', frame)
        image = vision.Image(content=img_encoded.tobytes())
        response = vision_client.document_text_detection(image=image)

        if response.full_text_annotation and response.full_text_annotation.text.strip():
            full_text = response.full_text_annotation.text.strip()
            # Gọi hàm tóm tắt từ file đã import
            summary = compress(full_text)
            if summary:
                # Gọi hàm TTS riêng để đọc tóm tắt
                speak_summary(summary, state)
        else:
            speak_vietnamese("Không nhận dạng được văn bản để tóm tắt.", state)
            
        state.isReading = False
    threading.Thread(target=process, daemon=True).start()


# -------------------- HTTP Server Code ----------------------
# (Bỏ trống như file gốc của bạn)

# -------------------- Main Loop ----------------------
def displayProcess(state):
    while state.video.isOpened() and not state.stop_event.is_set():
        ret, frame = state.video.read()
        if not ret or frame is None:
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        state.latest_frame = frame.copy()
        
        edited = frame.copy()
        hand_frame = frame.copy() # Tách frame cho hand detection để không bị vẽ đè

        if state.isReading:
            cv2.putText(edited, "DANG XU LY...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            for box, _ in state.cached_data:
                cv2.rectangle(edited, box[0], box[1], (0,255,255), 2)
            
            # Chỉ chạy hand_handler trên frame riêng để không ảnh hưởng đến frame chính
            hand_handler(hand_frame, state)

        cv2.imshow("He thong ho tro doc", edited)
        cv2.imshow("Nhan dien ban tay", hand_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # Phím ESC
            state.stop_event.set()
            break
        if key == ord('r'): # Phím 'r' để chạy OCR cho chế độ chỉ tay
            if not state.isOcr and not state.isReading:
                predict(state)
        if key == ord('s'): # Phím 's' để chạy tóm tắt (ĐÃ THÊM LẠI)
            if not state.isOcr and not state.isReading:
                run_compress_workflow(state)

    state.video.release()
    cv2.destroyAllWindows()

# -------------------- Run ----------------------
if __name__ == "__main__":
    app_state = AppState()
    displayProcess(app_state)
