import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# 初始化摄像头 - 使用V4L2驱动
cap = cv2.VideoCapture(0)  # Linux通常使用0表示默认摄像头
if not cap.isOpened():
    # 尝试使用V4L2后端
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("错误：无法访问摄像头，请检查权限和设备")
        exit(1)

# 加载YOLO模型
model = YOLO('best_more.pt')

# 获取类别名称
try:
    class_names = model.names
except AttributeError:
    class_names = {}

# 中文字体配置 - 常见Linux字体路径
linux_font_paths = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # 文泉驿正黑
    "/usr/share/fonts/truetype/arphic/uming.ttc",  # AR PL UMing
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # 备用字体
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf"  # 备用字体
]

font = None
for font_path in linux_font_paths:
    if os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, 30)
            print(f"使用字体: {font_path}")
            break
        except:
            continue

if font is None:
    print("警告：无法加载中文字体，将使用默认字体")
    font = ImageFont.load_default()

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        print("无法获取视频帧")
        continue  # 跳过当前帧继续循环

    # 将OpenCV图像转换为PIL图像
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 使用YOLO模型进行目标检测
    results = model.predict(
        source=frame,
        conf=0.5,
        iou=0.5,
        show=False,
        verbose=False
    )

    result = results[0]
    for box in result.boxes:
        # 获取边界框坐标
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # 获取置信度和类别
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = class_names.get(class_id, f"{class_id}")

        # 绘制边界框
        color = (0, 255, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # 准备标签文本
        label = f"{class_name} {confidence:.2f}"

        # 计算文本尺寸
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # 绘制文本背景
        draw.rectangle(
            [x1, y1 - text_height - 5, x1 + text_width, y1],
            fill=color
        )

        # 绘制文本
        draw.text(
            (x1, y1 - text_height - 5),
            label,
            font=font,
            fill=(0, 0, 0)
        )

    # 转换回OpenCV格式
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 显示处理后的帧
    cv2.imshow('YOLO Object Detection', frame)

    # 按ESC或q退出
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()