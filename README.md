# Spectral-Spatial Edge Enhanced SAR Detection

Official implementation of 《Spectral-Spatial Synergistic Learning and Context-Aware Edge Enhancement for Efficient Oriented Object Detection in SAR Images》.

Based on [YOLOv13](https://github.com/iMoonLab/yolov13) and [Ultralytics](https://github.com/ultralytics/ultralytics).

## Dataset

The default configuration targets the [**RSAR**](https://github.com/zhasion/RSAR) dataset with 6 categories: `aircraft`, `bridge`, `car`, `harbor`, `ship`, `tank`.

The dataset can be obtained from the [RSAR official repository](https://github.com/zhasion/RSAR). Organize your dataset as follows and update the `path` field in `ultralytics/cfg/datasets/rsar.yaml`:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Installation

```bash
conda create -n yolov13-sar python=3.11
conda activate yolov13-sar
pip install -r requirements.txt
pip install -e .
```

Optional: Install [Flash Attention](https://github.com/Dao-AILab/flash-attention) for acceleration:
```bash
pip install flash-attn
```

## Training

Our experiments were conducted on NVIDIA RTX 3090 GPUs with a per-GPU batch size of 12.

```bash
python train.py

# Custom settings
python train.py --model ultralytics/cfg/models/v13/yolov13_sar.yaml \
                --data ultralytics/cfg/datasets/rsar.yaml \
                --imgsz 800 --batch 12 --epochs 200

# Multi-GPU
python train.py --device 0,1,2,3 --batch 48

# Resume
python train.py --resume runs/train/exp/weights/last.pt
```

## Evaluation

```bash
python val.py --weights runs/train/exp/weights/best.pt

# Custom thresholds
python val.py --weights runs/train/exp/weights/best.pt \
              --iou 0.5 --conf 0.05 --split test
```

## Inference

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
results = model.predict(source="path/to/image.jpg", imgsz=800, conf=0.25)
results[0].show()
```

## Citation

```bibtex
@article{yolov13,
  title={YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception},
  author={Lei, Mengqi and Li, Siqi and Wu, Yihong and et al.},
  journal={arXiv preprint arXiv:2506.17733},
  year={2025}
}

@inproceedings{zhang2025rsar,
  title={Rsar: Restricted State Angle Resolver and Rotated SAR Benchmark},
  author={Zhang, Xin and Yang, Xue and Li, Yuxuan and Yang, Jian and Cheng, Ming-Ming and Li, Xiang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={7416--7426},
  year={2025}
}
```

## License

This project is licensed under the [AGPL-3.0 License](LICENSE).
