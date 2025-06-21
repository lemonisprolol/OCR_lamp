import os
import threading
import time
import cv2
import mediapipe as mp
from gtts import gTTS
import pygame
from google.cloud import vision
from ultralytics import YOLO
import tempfile
from compress import compress

# ----- Config Google Vision API -----
KEY_PATH = r"key.json"
if not os.path.exists(KEY_PATH):
    raise FileNotFoundError(f"Google Vision API key not found at: {KEY_PATH}")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
vision_client = vision.ImageAnnotatorClient()

# ----- Ensure screenshot folder exists -----
os.makedirs("./screenshot", exist_ok=True)

# ----- AppState Class to Replace Globals -----
class AppState:
    def __init__(self):
        self.localIP = "192.168.137.247"
        self.video = cv2.VideoCapture(f"http://{self.localIP}:81/stream")

        self.latest_frame = None   # Lưu frame mới nhất
        self.cached_data = []      # Danh sách các câu, mỗi câu là list các box-text
        self.stop_event = threading.Event()
        self.isSpeaking = False
        self.last_text = ""
        self.isReading = False
        self.isOcr = False
        self.lastTime = 0
        self.isDetected = False

# ----- TTS -----
def speak_vietnamese(text, state):
      pygame.mixer.init()
      def run_tts():
           try:
                state.isSpeaking = True
                tts = gTTS(text=text, lang='vi')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    temp_filename = fp.name
                    tts.save(temp_filename)

                pygame.mixer.music.load(temp_filename)
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)

                pygame.mixer.quit()
                state.isSpeaking = False
           except Exception as e:
                print("TTS Error:", e)
                state.isSpeaking = False

      threading.Thread(target=run_tts, daemon=True).start()

def speak_summary(summary_text, state):
    full_text_to_speak = "Đây là bản tóm tắt nội dung: " + summary_text
    speak_vietnamese(full_text_to_speak, state)

# ----- Hand Detection (MediaPipe) -----
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

def hand_handler(frame, state):
    height, width = frame.shape[:2]
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, landmarks, mpHands.HAND_CONNECTIONS)
            for index, lm in enumerate(landmarks.landmark):
                if index == 8:  # ngón trỏ
                    px = lm.x * width
                    py = lm.y * height
                    break
            for sentence in state.cached_data:
                # sentence: list các box-text trong 1 câu
                # check nếu trỏ nằm trong box nào của câu này
                for texts in sentence:
                    x1, y1 = texts[0][0][0], texts[0][0][1]
                    x2, y2 = texts[0][1][0], texts[0][1][1]
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        if state.last_text != texts[1] and not state.isSpeaking:
                            state.last_text = texts[1]
                            speak_vietnamese(texts[1], state)
                        break

# ----- Google Vision OCR -----
def predict(state):
      def run_predict():
           state.isOcr = True
           state.isReading = True
           print("Starting Google Vision OCR")
           speak_vietnamese("Đang chạy máy đọc", state)
           time.sleep(5)

           frame = state.latest_frame
           if frame is None:
                print("Can't get latest frame")
                state.isOcr = False
                state.isReading = False
                return

           img_path = "./screenshot/cap.jpg"
           cv2.imwrite(img_path, frame)

           with open(img_path, "rb") as image_file:
                content = image_file.read()
           image = vision.Image(content=content)

           response = vision_client.text_detection(image=image)
           texts = response.text_annotations
    
           data = []
           if texts:
                print("Full text:", texts[0].description.strip())
                for text in texts[1:]:
                    box = [(v.x, v.y) for v in text.bounding_poly.vertices]
                    if len(box) == 4:
                        data.append([[box[0], box[2]], text.description])

           # Không gộp box, mỗi box là 1 câu riêng lẻ
           state.cached_data = [[box] for box in data]

           speak_vietnamese("Máy đọc đã chạy xong", state)

           img = cv2.imread(img_path)
           colors = [(0,255,255), (255,0,255), (255,255,0), (0,128,255), (255,128,0)]
           for idx, sentence in enumerate(state.cached_data):
                color = colors[idx % len(colors)]
                for textBox in sentence:
                    x1, y1 = textBox[0][0]
                    x2, y2 = textBox[0][1]
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
           cv2.imwrite("./screenshot/output.jpg", img)

           print("OCR cached sentences:", [[t[1] for t in sent] for sent in state.cached_data])
           state.isReading = False
           state.isOcr = False
           state.isDetected = False

      threading.Thread(target=run_predict, daemon=True).start()

def run_compress_workflow(state):
    def process():
        if state.isReading or state.isOcr:
            return
        state.isReading = True
        
        speak_vietnamese("Bắt đầu quét văn bản để tóm tắt", state)
        
        frame = state.latest_frame
        if frame is None:
            print("Can't get latest frame")
            state.isReading = False
            return
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as fp:
            temp_img_path = fp.name
            cv2.imwrite(temp_img_path, frame)

        with open(temp_img_path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = vision_client.text_detection(image=image)
        texts = response.text_annotations
        
        os.remove(temp_img_path)

        if texts and texts[0].description:
            full_text = texts[0].description.strip()
            summary = compress(full_text)
            if summary:
                speak_summary(summary, state)
        else:
            speak_vietnamese("Không nhận dạng được văn bản.", state)
            
        state.isReading = False

    threading.Thread(target=process, daemon=True).start()

# ----- YOLO fist detection setup -----
yolo_model = YOLO("fist.pt")
class_names = yolo_model.names
target_object = "fist"

def detectFist(frame, state):
    yolo_results = yolo_model.predict(source=frame, conf=0.5, verbose=False)
    if yolo_results:
        for r in yolo_results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                name = class_names[cls_id]
                if name.lower() == target_object.lower() and conf >= 0.75:
                    if not state.isOcr:
                        predict(state)
                    return

# ----- Main camera & processing loop -----
def displayProcess(state):
    while state.video.isOpened() and not state.stop_event.is_set():
        ret, frame = state.video.read()
        if not ret or frame is None:
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        state.latest_frame = frame.copy()

        edited_frame = frame.copy()
        hand_frame = frame.copy()

        if not state.isReading:
            for idx, sentence in enumerate(state.cached_data):
                color = (0,255,255)
                for textBox in sentence:
                    edited_frame = cv2.rectangle(edited_frame, textBox[0][0], textBox[0][1], color, 2)

            detectFist(frame, state)
            hand_handler(hand_frame, state)

        cv2.imshow("Edited Frame", edited_frame)
        cv2.imshow("Hand Detection", hand_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            state.stop_event.set()
            break
        if key == ord('r'):
            if not state.isOcr and not state.isReading:
                predict(state)
        if key == ord('s'):
            if not state.isReading:
                run_compress_workflow(state)

    state.video.release()
    cv2.destroyAllWindows()

# ----- Run app -----
if __name__ == "__main__":
    app_state = AppState()
    displayProcess(app_state)