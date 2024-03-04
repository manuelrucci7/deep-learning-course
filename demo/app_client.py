import numpy as np
import cv2
import requests
import tempfile
import time
from datetime import datetime
#import telegram_send
from telegram import Bot
import asyncio
import subprocess

# define a custom coroutine
#async def send_to_telegram(bot, chat_id, text):
#    await bot.send_message(chat_id=chat_id, text=text)

# Connect to telegram
# https://medium.com/@robertbracco1/how-to-write-a-telegram-bot-to-send-messages-with-python-bcdf45d0a580
# pip install telegram-send telegram-send configure
# configure telegram
# Write on the terminate $ telegram-send --configure 
# pip install python-telegram-bot


def send_msg(msg):
    # The command to run
    command = ["telegram-send", f"{msg}"]
    # Run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Get the output and error messages
    stdout, stderr = process.communicate()

if __name__ == "__main__":
    #bot = Bot(token=API_TOKEN)
    
    # Create a VideoCapture object
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
        
        if el.seconds > 10:

            #asyncio.run(send_to_telegram(bot, CHAT_ID, "Hello, I am a bot"))
            
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
            det = 0
            for i in range(0, num):
                x1, y1, x2, y2 = res["boxes"][i]
                score    = res["scores"][i]
                name = res["names"][i]
                if name=="person" and score>0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name}: {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    det = det +1
            cv2.imshow("Results", frame)

            if det>0:
                # Send message to telegram
                send_msg("Person detected!")
            

        # Check if the 'q' key is pressed to exit the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
