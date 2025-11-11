from ultralytics import YOLO
from cap_from_youtube import cap_from_youtube
import cv2

# 🔗 ลิงก์วิดีโอ YouTube
youtube_url = "https://youtu.be/1sLcwlvM28M"

# โหลดวิดีโอจาก YouTube (เลือกความละเอียดที่ดีที่สุด)
cap = cap_from_youtube(youtube_url, resolution="best")

# โหลดโมเดล YOLOv8 ที่เทรนไว้ล่วงหน้า
model = YOLO('yolov8n.pt')

# ตั้งค่าการบันทึกวิดีโอผลลัพธ์
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None  # จะกำหนดขนาดหลังจากอ่านเฟรมแรกได้

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ วิดีโอสิ้นสุด หรือไม่สามารถอ่านเฟรมได้")
        break

    # ถ้ายังไม่กำหนด out ให้กำหนดเมื่อรู้ขนาดภาพ
    if out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter('output_car_detection.mp4', fourcc, 20.0, (w, h))

    # ตรวจจับวัตถุด้วย YOLO
    results = model(frame, verbose=False)

    # นับจำนวนรถ
    car_count = 0
    for box in results[0].boxes:
        cls = int(box.cls[0])
        if model.names[cls] in ["car", "truck", "bus", "motorbike"]:
            car_count += 1

    # วาดกรอบและข้อความ
    annotated = results[0].plot()
    cv2.putText(annotated, f'Cars detected: {car_count}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # แสดงผลแบบเรียลไทม์
    cv2.imshow("Real-Time Car Detection (YouTube)", annotated)

    # บันทึกวิดีโอผลลัพธ์
    out.write(annotated)

    # กด 'q' เพื่อหยุด
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดทั้งหมด
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
print("✅ วิดีโอบันทึกไว้ที่: output_car_detection.mp4")
