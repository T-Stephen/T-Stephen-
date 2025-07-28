import cv2
import pygame
from ultralytics import YOLO
from datetime import datetime
from tkinter import Tk, Label, PhotoImage, Canvas
from PIL import Image, ImageTk

# Initialize YOLO model
model = YOLO("best.pt")

# Initialize Pygame Mixer for Siren Sound
pygame.mixer.init()
siren = pygame.mixer.Sound("siren.mp3")

# Initialize Tkinter GUI
root = Tk()
root.title("Ambulance Detection System with Traffic Signal and Clock")

# GUI Layout Setup
root.geometry("1000x700")

# Load and Resize Traffic Signal Images
green_img = Image.open("green.png").resize((200, 137))
red_img = Image.open("red.png").resize((200, 137))
green_light = ImageTk.PhotoImage(green_img)
red_light = ImageTk.PhotoImage(red_img)

# Labels
signal_label = Label(root, image=red_light)
signal_label.pack(pady=10)

clock_label = Label(root, font=('Helvetica', 20), fg="blue")
clock_label.pack()

video_label = Label(root)
video_label.pack()

# Log file
log_file = open("detection_log.txt", "a")

# Clock Function
def update_clock():
    current_time = datetime.now().strftime('%H:%M:%S')
    clock_label.config(text="Current Time: " + current_time)
    root.after(1000, update_clock)

# Webcam Capture
cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    results = model(frame)[0]
    detected_classes = [model.names[int(cls)] for cls in results.boxes.cls]

    if 'ambulance' in detected_classes:
        print("Ambulance Detected!")
        signal_label.config(image=green_light)
        if not pygame.mixer.get_busy():
            siren.play()
        log_file.write(f"Ambulance detected at {datetime.now()}\n")
        log_file.flush()
    else:
        signal_label.config(image=red_light)
        siren.stop()

    # Display Webcam Feed
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (800, 500))
    img = ImageTk.PhotoImage(Image.fromarray(frame))
    video_label.configure(image=img)
    video_label.image = img

    root.after(10, update_frame)

update_clock()
update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
log_file.close()
