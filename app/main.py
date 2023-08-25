import cv2
import numpy as np
from typing import Union # เพื่อใช้ในการประกาศ type hint สำหรับ parameter ของ function
from fastapi import FastAPI # import FastAPI เพื่อใช้ในการสร้าง API
from pydantic import BaseModel # เพื่อไว้สร้าง model ที่เอาไว้รับข้อมูลที่ส่งมาจาก body
import base64

app = FastAPI()

# สร้าง class จาก BaseModel เพื่อไว้เก็บข้อมูล JSON ที่ถูกส่งเข้ามาทาง body
class Item(BaseModel):
    image_base64: str

# ฟังก์ชั่นในการเเปลงรูปภาพเเบบ base64 ไปเป็นไฟล์รูปภาพ
def readb64(uri):
   encoded_data = uri.split(',')[1] # ทำการเเยกข้อความ URL ด้วยเครื่องหมาย , เลือก [1] หมายความว่าตัดส่วน data:image/jpeg;base64, ออกไปจาก URL
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8) # แปลงข้อมูลที่ถูกเข้ารหัสด้วย Base64 กลับมาเป็นอาร์เรย์ของตัวเลขจำนวนเต็ม 8 บิต
   img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) # เเปลงเป็นรูปภาพ, ใช้ cv2.IMREAD_GRAYSCALE เพื่อโหลดภาพเป็นโทนสีเทา
   return img_gray


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None): 
    return {"item_id": item_id, "q": q}

# ส่วนของการประมวลผลเพื่อหาเอกลักษณ์ของภาพด้วย HOG
@app.post("/api/genhog")
def Image_Features(data: Item): # data: Item คือการนำข้อมูลที่ส่งมาจาก body มาเก็บไว้ใน data ซึ่งเป็นรูปภาพในรูปเเบบ base64
    
    img_gray = readb64(data.image_base64) # ทำการเรียกใช้ฟังก์ชั่นในการเเปลงรูปภาพเเบบ base64 ไปเป็นไฟล์รูปภาพ

    img_new = cv2.resize(img_gray, (128,128), cv2.INTER_AREA) # ปรับขนาดของภาพ grayscale เป็นขนาด 128x128 pixels โดยใช้วิธีการรับรู้ขนาดรูปที่ดีที่สุดเมื่อปรับขนาดด้วย cv2.INTER_AREA.
    win_size = img_new.shape # เก็บขนาดของภาพ img_new ในตัวแปร win_size เพื่อใช้ในการกำหนดขนาดของหน้าต่างที่ใช้ในการคำนวณ HOG.

    # กำหนดพารามิเตอร์ที่ใช้ในการคำนวณ HOG descriptor โดยแต่ละพารามิเตอร์จะกำหนดขนาดของเซลล์ (cell) ขนาดของบล็อก (block) และการเลื่อนบล็อก (block stride) รวมถึงจำนวนของ bin ใน histogram.
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8) 
    num_bins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins) # สร้างอ็อบเจกต์ HOG descriptor โดยกำหนดพารามิเตอร์ที่ตั้งค่าไว้ก่อนหน้า.
    hog_descriptor = hog.compute(img_new) # คำนวณ HOG descriptor สำหรับ img_new ที่เราปรับขนาดไว้ ผลลัพธ์จะเป็น numpy array ที่เก็บค่าของ HOG descriptor สำหรับแต่ละเซลล์และบล็อกในภาพ.
    
    return {"vector": hog_descriptor.tolist()}

