# Explainability for end-users in object detection
## Visualizing YOLOv3 using Grad-CAM

### YOLOv3 Implemented in TensorFlow 2.0
This repo uses an implementation of YOLOv3 in TensorFlow 2.0
Please visite the source [repository](https://github.com/zzh8829/yolov3-tf2) of this implementation. You can find model visualizations in `./0_model`

### Explainability
You can find examples of the generated visualizations in `1_results`.
![Example results](https://raw.githubusercontent.com/marvinKrae/yolo-gradcam/public/1_results/nyc_guy/advanced/aggregated/person_small.jpg?token=ALXAHS5FKNXDMOQZWWTGHRLATPMP2)

### Usage

You can use `detect.py` to start an object detection and explain the results, after you installed all dependencies.
For example you could use the following command to detect objects in `image.jpg` and explain the detection for "person" (class id: 0) and "cat" (class id: 15)
```
  python detect.py --image ./data/image.jpg  --explain 0,15
```
All classes are listed in `./data/coco.names`.

### Installation

#### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov3-tf2-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov3-tf2-gpu
```

#### Pip

```bash
pip install -r requirements.txt
```

### Nvidia Driver (For GPU)

```bash
# Ubuntu 18.04
sudo apt-add-repository -r ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-430
# Windows/Other
https://www.nvidia.com/Download/index.aspx
```

### Convert pre-trained Darknet weights

```bash
# yolov3
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf

# yolov3-tiny
wget https://pjreddie.com/media/files/yolov3-tiny.weights -O data/yolov3-tiny.weights
python convert.py --weights ./data/yolov3-tiny.weights --output ./checkpoints/yolov3-tiny.tf --tiny
```

## Command Line Args Reference

```bash
convert.py:
  --output: path to output
    (default: './checkpoints/yolov3.tf')
  --[no]tiny: yolov3 or yolov3-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './data/yolov3.weights')
  --num_classes: number of classes in the model
    (default: '80')
    (an integer)

detect.py:
  --classes: path to classes file
    (default: './data/coco.names')
  --image: path to input image
    (default: './data/girl.png')
  --output: path to output image
    (default: './output.jpg')
  --[no]tiny: yolov3 or yolov3-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './checkpoints/yolov3.tf')
  --num_classes: number of classes in the model
    (default: '80')
    (an integer)
  --explain: comma seperated list of class ids to explain
    (default: '0,1')
```