from ultralytics import YOLO
import cv2

# โหลดโมเดล YOLOv8 ที่เทรนไว้ล่วงหน้า
model = YOLO('yolov8n.pt')

# โหลดรูปภาพ
image_path = "car_pj.jpg"  # 
image = cv2.imread(image_path)

# ใช้โมเดลตรวจจับวัตถุ
results = model(image)

# นับจำนวนรถยนต์ (class id ของ "car" คือ 2)
car_count = 0
for box in results[0].boxes:
    cls = int(box.cls[0])  # ดึง class id
    if model.names[cls] in ["car", "truck", "bus", "motorbike"]: 
        car_count += 1


annotated = results[0].plot()
cv2.putText(annotated, f'Cars detected: {car_count}', (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

cv2.imshow("Detected Cars", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"จำนวนรถยนต์ที่ตรวจพบ: {car_count} คัน")
