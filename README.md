# Real-time Hand Gesture Recognition with 3D CNNs
PyTorch implementation of the article [Real-time Hand Gesture Detection and Classification Using Convolutional Neural Networks](https://arxiv.org/abs/1901.10323) and [Resource Efficient 3D Convolutional Neural Networks](https://arxiv.org/pdf/1904.02422.pdf), codes and pretrained models.


<div align="center" style="width:image width px;">
  <img  src="https://media.giphy.com/media/9M3aPvPOVxSQmYGv8p/giphy.gif" width=500 alt="simulation results">
</div>

Figure: A real-time simulation of the architecture with input video from EgoGesture dataset (on left side) and real-time (online) classification scores of each gesture (on right side) are shown, where each class is annotated with different color. 


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ahmetgunduz/Real-time-GesRec&type=Date)](https://star-history.com/#ahmetgunduz/Real-time-GesRec&Date)


This code includes training, fine-tuning and testing on EgoGesture and nvGesture datasets.  
Note that the code only includes ResNet-10, ResNetL-10, ResneXt-101, C3D v1, whose other versions can be easily added.

## Abstract
Real-time recognition of dynamic hand gestures from video streams is a challenging task since (i) 
there is no indication when a gesture starts and ends in the video, (ii) performed gestures should 
only be recognized once, and (iii) the entire architecture should be designed considering the memory 
and power budget. In this work, we address these challenges by proposing a hierarchical structure 
enabling offline-working convolutional neural network (CNN) architectures to operate online efficiently
by using sliding window approach. The proposed architecture consists of two models: (1) A detector which 
is a lightweight CNN architecture to detect gestures and (2) a classifier which is a deep CNN to classify 
the detected gestures. In order to evaluate the single-time activations of the detected gestures, we propose
to use the Levenshtein distance as an evaluation metric since it can measure misclassifications, multiple detections,
and missing detections at the same time. We evaluate our architecture on two publicly available datasets - EgoGesture
and NVIDIA Dynamic Hand Gesture Datasets - which require temporal detection and classification of the performed hand gestures.
ResNeXt-101 model, which is used as a classifier, achieves the state-of-the-art offline classification accuracy of 94.04% and 
83.82% for depth modality on EgoGesture and NVIDIA benchmarks, respectively. In real-time detection and classification,
we obtain considerable early detections while achieving performances close to offline operation. The codes and pretrained models used in this work are publicly available. 



## Requirements

* [PyTorch](http://pytorch.org/)

```bash
conda install pytorch torchvision cuda80 -c soumith
```

* Python 3

### Pretrained models
[Pretrained_models_v1 (1.08GB)](https://drive.google.com/file/d/11MJWXmFnx9shbVtsaP1V8ak_kADg0r7D/view?usp=sharing): The best performing models in [paper](https://arxiv.org/abs/1901.10323)

[Pretrained_RGB_models_for_det_and_clf (371MB)(Google Drive)](https://drive.google.com/file/d/1V23zvjAKZr7FUOBLpgPZkpHGv8_D-cOs/view?usp=sharing)
[Pretrained_RGB_models_for_det_and_clf (371MB)(Baidu Netdisk)](https://pan.baidu.com/s/114WKw0lxLfWMZA6SYSSJlw) -code:p1va

[Pretrained_models_v2 (15.2GB)](https://drive.google.com/file/d/1rSWnzlOwGXjO_6C7U8eE6V43MlcnN6J_/view?usp=sharing): All models in [paper](https://ieeexplore.ieee.org/document/8982092) with efficient 3D-CNN Models
## Preparation

### EgoGesture

* Download videos by following [the official site](http://www.nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html).
* We will use extracted images that is also provided by the owners

* Generate n_frames files using ```utils/ego_prepare.py``` 

N frames format is as following: "path to the folder" "class index" "start frame" "end frame"

```bash
mkdir annotation_EgoGesture
python utils/ego_prepare.py training trainlistall.txt all
python utils/ego_prepare.py training trainlistall_but_None.txt all_but_None
python utils/ego_prepare.py training trainlistbinary.txt binary
python utils/ego_prepare.py validation vallistall.txt all
python utils/ego_prepare.py validation vallistall_but_None.txt all_but_None
python utils/ego_prepare.py validation vallistbinary.txt binary
python utils/ego_prepare.py testing testlistall.txt all
python utils/ego_prepare.py testing testlistall_but_None.txt all_but_None
python utils/ego_prepare.py testing testlistbinary.txt binary
```

* Generate annotation file in json format similar to ActivityNet using ```utils/egogesture_json.py```

```bash
python utils/egogesture_json.py 'annotation_EgoGesture' all
python utils/egogesture_json.py 'annotation_EgoGesture' all_but_None
python utils/egogesture_json.py 'annotation_EgoGesture' binary
```

### nvGesture

* Download videos by following [the official site](https://research.nvidia.com/publication/online-detection-and-classification-dynamic-hand-gestures-recurrent-3d-convolutional).

* Generate n_frames files using ```utils/nv_prepare.py``` 

N frames format is as following: "path to the folder" "class index" "start frame" "end frame"

```bash
mkdir annotation_nvGesture
python utils/nv_prepare.py training trainlistall.txt all
python utils/nv_prepare.py training trainlistall_but_None.txt all_but_None
python utils/nv_prepare.py training trainlistbinary.txt binary
python utils/nv_prepare.py validation vallistall.txt all
python utils/nv_prepare.py validation vallistall_but_None.txt all_but_None
python utils/nv_prepare.py validation vallistbinary.txt binary
```

* Generate annotation file in json format similar to ActivityNet using ```utils/nv_json.py```

```bash
python utils/nv_json.py 'annotation_nvGesture' all
python utils/nv_json.py 'annotation_nvGesture' all_but_None
python utils/nv_json.py 'annotation_nvGesture' binary
```

### Jester

* Download videos by following [the official site](https://20bn.com/datasets/jester).

* N frames and class index  file is already provided annotation_Jester/{'classInd.txt', 'trainlist01.txt', 'vallist01.txt'}

N frames format is as following: "path to the folder" "class index" "start frame" "end frame"

* Generate annotation file in json format similar to ActivityNet using ```utils/jester_json.py```

```bash
python utils/jester_json.py 'annotation_Jester'
```


## Running the code
* Offline testing (offline_test.py) and training (main.py)
```bash
bash run_offline.sh
```

* Online testing
```bash
bash run_online.sh
```

## Raspberry Pi 4 (CPU-only deployment)

The Pi deployment is a **classifier-only** architecture. It is designed for one
available checkpoint and does not require CUDA or a second computer at runtime:

```
USB / Pi Camera  ->  OpenCV capture  ->  8-frame RGB buffer  ->  3D CNN on CPU
                                                                  |
                                                                  v
                                                        class + confidence output
```

The repository contains the model definitions, not trained model weights. Place
a trained `.pth` checkpoint in `models/` (or another location) before running.
The model settings must exactly match the checkpoint. Existing ResNeXt-101
weights will run but are generally too slow for real-time use on Pi 4; a trained
MobileNetV2 or ShuffleNetV2 width `0.5` model is the intended Pi configuration.

If you downloaded the README's **Pretrained RGB detector + classifier models**
archive, use its `egogesture_resnext_101_RGB_32.pth` classifier with the exact
settings in `deploy/egogesture-rgb-pretrained.env.example`. It will load in the
Pi runner, but it is a ResNeXt-101 / 32-frame model and is expected to be slow
on a Pi 4. The companion `egogesture_resnetl_10_RGB_8.pth` is the detector; the
Pi runner deliberately uses the classifier alone because only one model is
needed for the simple camera architecture.

### Hardware

* Raspberry Pi 4 (64-bit Raspberry Pi OS; 4 GB RAM minimum, 8 GB preferred)
* Official USB-C supply and active cooling
* USB webcam or a configured Pi Camera
* MicroSD card or SSD with at least 8 GB free

### Install and run

On the Pi, clone the repository, create a virtual environment, install an ARM64
PyTorch build compatible with its Python version, then install the remaining
dependencies:

```bash
sudo apt update
sudo apt install -y git python3-venv libopenblas-dev
git clone https://github.com/YOUR-USER/YOUR-REPO.git /opt/gesture-recognition
cd /opt/gesture-recognition
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch
pip install -r requirements-pi.txt
mkdir -p models
```

Use `pi_inference.py` for a USB camera or video file. It loads both regular and
DataParallel (`module.`-prefixed) checkpoints. A lightweight checkpoint can run
with this command:

```bash
python3 pi_inference.py --checkpoint /home/pi/models/gesture.pth --source 0 \
  --model mobilenetv2 --width-mult 0.5 --classes 83 --duration 8 --size 80 \
  --stride 2 --threads 4
```

`--source 0` selects the first USB camera; replace it with a video filename to
test a file. Add `--labels labels.txt` for one label per line, or `--display` to
open a preview window. The script prints a prediction only after it has received
the requested number of frames. It is intentionally CPU-only, so do not use the
old `online_test_video.py` for Raspberry Pi deployment: that script calls CUDA
unconditionally.

### Start automatically on boot

Configure the supplied service once the direct command works. Copy the example
configuration, update every architecture value to match the checkpoint, and
enable the service:

```bash
sudo cp deploy/gesture-recognition.env.example /etc/gesture-recognition.env
sudo nano /etc/gesture-recognition.env
chmod +x deploy/run-pi.sh
sudo cp deploy/gesture-recognition.service.example /etc/systemd/system/gesture-recognition.service
sudo systemctl daemon-reload
sudo systemctl enable --now gesture-recognition
sudo journalctl -u gesture-recognition -f
```

## Citation

Please cite the following articles if you use this code or pre-trained models:

```bibtex
@article{kopuklu_real-time_2019,
	title = {Real-time Hand Gesture Detection and Classification Using Convolutional Neural Networks},
	url = {http://arxiv.org/abs/1901.10323},
	author = {Köpüklü, Okan and Gunduz, Ahmet and Kose, Neslihan and Rigoll, Gerhard},
  year={2019}
}
```

```bibtex
@article{kopuklu2020online,
  title={Online Dynamic Hand Gesture Recognition Including Efficiency Analysis},
  author={K{\"o}p{\"u}kl{\"u}, Okan and Gunduz, Ahmet and Kose, Neslihan and Rigoll, Gerhard},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  volume={2},
  number={2},
  pages={85--97},
  year={2020},
  publisher={IEEE}
}
```

## Acknowledgement
We thank Kensho Hara for releasing his [codebase](https://github.com/kenshohara/3D-ResNets-PyTorch), which we build our work on top.
#   v i s o n h a n d  
 