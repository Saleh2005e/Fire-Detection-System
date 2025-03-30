import cv2
import torch
import datetime
import os
import requests
from ultralytics import YOLO
import math
import cvzone
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import time
import threading  

bot_token = 'bot_token'
chat_id = 'chat_id'

def send_telegram_message(image_path, report_text):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    params = {"chat_id": chat_id, "text": report_text}
    requests.get(url, params=params)
    requests.get(url, params=params, timeout=40)


    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    with open(image_path, 'rb') as img:
        files = {'photo': img}
        params = {"chat_id": chat_id}
        requests.post(url, params=params, files=files)
        requests.get(url, params=params, timeout=40)


output_folder = "detected_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

root = Tk()
root.title("Fire Detection System")
root.geometry("1000x600")

frame_main = Frame(root)
frame_main.pack(fill=BOTH, expand=True)

frame_cam = Frame(frame_main, bg="black", width=700, height=500)
frame_cam.pack(side=LEFT, fill=BOTH, expand=True)

frame_log = Frame(frame_main, width=300, bg="lightgray")
frame_log.pack(side=RIGHT, fill=Y)

lmain = Label(frame_cam, bg="black")
lmain.pack(fill=BOTH, expand=True)

log_label = Label(frame_log, text="Activity log", bg="lightgray", font=("Arial", 12, "bold"))
log_label.pack(pady=5)
log_text = Text(frame_log, height=30, wrap=WORD, bg="white", fg="black")
log_text.pack(pady=5, padx=10, fill=BOTH, expand=True)

cap = cv2.VideoCapture(0)
model = YOLO('./fire.pt')
classnames = ['fire']
last_fire_detection_time = time.time() - 10

def show_frame():
    global last_fire_detection_time
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)

    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 75:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100], scale=1.5, thickness=2)

                if time.time() - last_fire_detection_time >= 10:
                    last_fire_detection_time = time.time()
                    image_filename = f"fire_detected_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                    image_path = os.path.join(output_folder, image_filename)
                    cv2.imwrite(image_path, frame)

                    log_text.insert(END, f"Fire Detected: {image_path}\nTimestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_text.see(END)
                    report_text = f"Fire Detected!\nTimestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    threading.Thread(target=send_telegram_message, args=(image_path, report_text)).start()  # استخدام الخيوط

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

show_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
