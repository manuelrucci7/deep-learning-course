import numpy as np
import cv2
import requests
import tempfile
import time
from datetime import datetime

cap = cv2.VideoCapture(0)

t = datetime.now()

while True:

    c = datetime.now()
    el = c-t
    # if difference in time greater than 3 seconds
    ret, frame = cap.read()
    if ret:
        # Display the frame
        cv2.imshow("Camera", frame)
    
    if el.seconds > 5:
        t = datetime.now()

        # Save the frame to a temporary file
        is_success, im_buf_arr = cv2.imencode(".jpg", frame)
        byte_im = im_buf_arr.tobytes()

        # Send the frame to the server
        response = requests.post(
            "http://localhost:8000/predict",
            files={"file": ("frame.jpg", byte_im)},
        )
        
        # Print the server's response
        print(response.json())
        res = response.json()["prediction"]
        num = len(res["scores"])
        # Display the results
        for i in range(0, num):
            x1, y1, x2, y2 = res["boxes"][i]
            score    = res["scores"][i]
            name = res["names"][i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name}: {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Results", frame)

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
