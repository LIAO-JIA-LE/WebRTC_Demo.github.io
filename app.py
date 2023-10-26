import asyncio
import base64
import websockets
import cv2
import numpy as np

from ultralytics import YOLO
# # Load the YOLOv8 model
model = YOLO('yolov8n.pt')

async def image_processing(websocket, path):
    while True:
        image_data = await websocket.recv()

        # 解码接收到的图像数据
        img_bytes = image_data.split(',')[1].encode()
        nparr = np.frombuffer(base64.b64decode(img_bytes), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 在此处进行图像处理
        # 例如，可以使用OpenCV进行实时处理
        image = model(image)[0].plot()
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 将处理后的图像发送回前端
        retval, buffer = cv2.imencode('.jpg', image)
        await websocket.send(buffer.tobytes())
        

start_server = websockets.serve(image_processing, "localhost", 8080)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
