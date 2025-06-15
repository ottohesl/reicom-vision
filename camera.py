import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 初始化摄像头
cap = cv2.VideoCapture(1)  # 0表示默认摄像头

# 加载YOLO模型
model = YOLO('best_more.pt')

# 获取类别名称
try:
    class_names = model.names
except AttributeError:
    class_names = {}

# 加载中文字体 - 替换为您系统中的中文字体文件路径
try:
    # 常见中文字体路径（Windows系统）
    font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
    font = ImageFont.truetype(font_path, 30)
except:
    print("警告：无法加载中文字体，将使用默认字体")
    font = ImageFont.load_default()

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        print("无法获取视频帧")
        break

    # 将OpenCV图像转换为PIL图像（以便处理中文）
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 使用YOLO模型进行目标检测
    results = model.predict(
        source=frame,
        conf=0.5,  # 置信度阈值
        iou=0.5,  # IoU阈值
        show=False,
        verbose=False  # 关闭详细输出以提升性能
    )

    result = results[0]
    for box in result.boxes:
        # 获取边界框坐标（左上角x, y, 右下角x, y）
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # 获取置信度
        confidence = float(box.conf[0])

        # 获取类别ID和名称
        class_id = int(box.cls[0])
        class_name = class_names.get(class_id, f"{class_id}")

        # 绘制边界框
        color = (0, 255, 0)  # 绿色框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # 准备标签文本
        label = f"{class_name} {confidence:.2f}"

        # 计算文本尺寸
        text_size = draw.textbbox((0, 0), label, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]

        # 绘制文本背景
        draw.rectangle([x1, y1 - text_height - 5, x1 + text_width, y1], fill=color)

        # 绘制文本
        draw.text((x1, y1 - text_height - 5), label, font=font, fill=(0, 0, 0))

    # 将PIL图像转换回OpenCV格式
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 显示处理后的帧
    cv2.imshow('YOLO目标检测', frame)


    key = cv2.waitKey(1)
    if key != -1:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()