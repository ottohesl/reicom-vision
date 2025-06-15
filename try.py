from ultralytics import YOLO
import os
import glob
import contextlib
import sys

# 加载训练好的模型
model = YOLO('best_more.pt')

# 图片目录路径
image_dir = r"F:\YOLOv8\recon\YOLO训练集\images\val"
# 输出目录路径
output_dir = r"F:\YOLOv8\recon\YOLO训练集\results"


@contextlib.contextmanager
def suppress_output():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout



os.makedirs(output_dir, exist_ok=True)

# 获取目录下所有jpg图片
image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))

# 处理每张图片
for img_path in image_paths:
    with suppress_output():

        results = model.predict(
            source=img_path,
            save=True,
            save_dir=output_dir,
            conf=0.7,
            show=True,
            verbose=False
        )

    # 提取文件名
    filename = os.path.basename(img_path)

    # 初始化计数器
    class_counts = {
        '白色-人质': 0,
        '红色-敌人': 0,
        '蓝色-敌人': 0,
        '迷彩-友军': 0
    }

    # 处理每个检测结果
    for result in results:
        # 获取检测信息
        boxes = result.boxes
        cls_ids = boxes.cls.tolist()
        class_names = result.names

        # 统计每个类别的数量
        for cls_id in cls_ids:
            class_name = class_names[int(cls_id)]
            if class_name in class_counts:
                class_counts[class_name] += 1

    # 计算总人数
    total_count = sum(class_counts.values())

    # 准备输出部分
    output_parts = []
    for class_name, count in class_counts.items():
        if count > 0:
            output_parts.append(f"{class_name}x{count}人")

    # 打印结果
    if output_parts:
        print(f"检测到总人数: {total_count}人 - 图片({filename})检测到的: {', '.join(output_parts)}")
    else:
        print(f"检测到总人数: 0人 - 图片({filename})未检测到目标")