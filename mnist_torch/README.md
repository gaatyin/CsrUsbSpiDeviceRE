# YOLO 分类测试（用于 D:\mnist_torch）

把本目录全部文件复制到 `D:\mnist_torch`，在已有 PyTorch 虚拟环境中执行：

```bat
cd /d D:\mnist_torch
setup_yolo.bat
python test_classify.py
```

指定本地图片：

```bat
python test_classify.py path\to\your.jpg
```

默认使用 ImageNet 预训练的 `yolo11n-cls.pt`（图像分类，不是目标检测框）。
