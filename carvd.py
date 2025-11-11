from ultralytics import YOLO
import cv2

# โหลดโมเดล YOLOv8 ที่เทรนไว้ล่วงหน้า
model = YOLO('yolov8n.pt')

# เปิดกล้องเว็บแคมหรือวิดีโอ (ใส่ path วิดีโอได้ เช่น "traffic.mp4")
cap = cv2.VideoCapture(0)  # 0 คือกล้องเว็บแคมหลัก

if not cap.isOpened():
    print("❌ ไม่สามารถเปิดกล้องหรือวิดีโอได้")
    exit()

# ตั้งค่าการบันทึกวิดีโอผลลัพธ์
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_car_detection.mp4', fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ ไม่สามารถอ่านเฟรมได้ หรือวิดีโอจบแล้ว")
        break

    # ตรวจจับวัตถุ
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

    # แสดงผลและบันทึกลงไฟล์
    cv2.imshow("Real-Time Car Detection", annotated)
    out.write(annotated)

    # กด 'q' เพื่อหยุดการทำงาน
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการทำงานทั้งหมด
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ วิดีโอบันทึกไว้ที่: output_car_detection.mp4")
