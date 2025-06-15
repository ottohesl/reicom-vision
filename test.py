from ultralytics import YOLO
import multiprocessing


def train_model():
    # 加载模型
    model = YOLO('yolov8n.pt')

    # 训练参数配置
    results = model.train(
        data='兵人.yaml',
        epochs=500,
        imgsz=640,
        batch=8,
        device='0',
        workers=0,
        verbose=True
    )


if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_model()
    print("训练完成!")