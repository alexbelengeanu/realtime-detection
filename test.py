import cv2
import numpy as np
import torch

# loading all the class labels (objects)
labels = open("data/coco.names").read().strip().split("\n")
# generating colors for each object for later plotting
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")


# adresa RTSP sau WebRTC a camerei web
url = "http://86.127.212.219:80/cgi-bin/faststream.jpg?stream=half&fps=15&rand=COUNTER"

# incarca modelul YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# deschide conexiunea la camera web
cap = cv2.VideoCapture(url)

# verifică dacă conexiunea a fost deschisă cu succes
if not cap.isOpened():
    print("Nu se poate accesa camera web!")
    exit()

# bucla infinita pentru a afișa fluxul video într-un window OpenCV
while True:
    # citeste un frame din fluxul video
    ret, frame = cap.read()

    # verifica daca citirea frame-ului s-a realizat cu succes
    if not ret:
        print("Nu se poate citi frame-ul!")
        break

    # redimensioneaza frame-ul
    #frame = imutils.resize(frame, width=800)
    result = model(frame)
    results = results.pandas().xyxy[0]  # img1 predictions (pandas)


    # afișează frame-ul într-un window OpenCV
    cv2.imshow("Camera Web", frame)

    # asteapta apasarea tastei 'q' pentru a iesi din bucla infinita
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# elibereaza resursele
cap.release()
cv2.destroyAllWindows()
