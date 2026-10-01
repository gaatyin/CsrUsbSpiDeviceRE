# YOLO 摄像头实时检测（用于 D:\mnist_torch）

把本目录全部文件复制到 `D:\mnist_torch`，在已有 PyTorch 虚拟环境中执行：

```bat
cd /d D:\mnist_torch
setup_yolo.bat
python test_detect_camera.py
```

常用参数：

```bat
python test_detect_camera.py --camera 0
python test_detect_camera.py --model yolo11n.pt --conf 0.4
```

预览窗口按 `q` 退出。默认使用 COCO 预训练检测模型 `yolo11n.pt`（画框 + 类别）。
