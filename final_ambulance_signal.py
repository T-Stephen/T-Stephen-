import cv2
from ultralytics import YOLO
from tkinter import Tk, Label
from PIL import Image, ImageTk
import threading
import pygame

# Initialize Pygame mixer
pygame.mixer.init()
siren = pygame.mixer.Sound("siren.mp3")  # Ensure siren.mp3 is in same folder

model = YOLO("best.pt")

red_img = Image.open("red.png").resize((200, 137))
green_img = Image.open("green.png").resize((200, 137))

root = Tk()
root.title("Traffic Signal - Ambulance Detection")

signal_label = Label(root)
signal_label.pack()

status_label = Label(root, text="Detecting...", font=("Arial", 20))
status_label.pack()

cap = cv2.VideoCapture(0)

# To prevent multiple siren plays
siren_playing = False

def play_siren_once():
    global siren_playing
    if not siren_playing:
        siren_playing = True
        siren.play(-1)  # Loop siren

def stop_siren():
    global siren_playing
    if siren_playing:
        siren.stop()
        siren_playing = False

def update_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    results = model.predict(frame, imgsz=416, verbose=False)
    detections = results[0].boxes.cls.cpu().numpy().astype(int)

    if 0 in detections:
        signal_img = green_img
        status_label.config(text="Ambulance Detected: GREEN LIGHT 🚑", fg="green")
        play_siren_once()
    else:
        signal_img = red_img
        status_label.config(text="No Ambulance: RED LIGHT 🚦", fg="red")
        stop_siren()

    cv2.imshow("Webcam View", frame)

    signal_photo = ImageTk.PhotoImage(signal_img)
    signal_label.config(image=signal_photo)
    signal_label.image = signal_photo

    if cv2.waitKey(1) & 0xFF == ord('q'):
        root.destroy()
        cap.release()
        cv2.destroyAllWindows()
        stop_siren()
    else:
        root.after(10, update_frame)

update_frame()
root.mainloop()
