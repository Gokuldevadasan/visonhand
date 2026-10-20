"""CPU-only gesture recognition for a Raspberry Pi camera or video file.

This does not use CUDA, DataParallel, datasets, or a display by default.  The
architecture arguments must match the checkpoint that is supplied.
"""
import argparse
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mean import get_mean, get_std
from model import generate_model


def parse_args():
    parser = argparse.ArgumentParser(description='Raspberry Pi CPU gesture inference')
    parser.add_argument('--checkpoint', required=True, help='Classifier .pth checkpoint')
    parser.add_argument('--source', default='0', help='Camera number (for example 0) or video path')
    parser.add_argument('--labels', default='', help='Optional text file: one class label per line')
    parser.add_argument('--model', default='mobilenetv2',
                        choices=['c3d', 'squeezenet', 'mobilenet', 'mobilenetv2',
                                 'shufflenet', 'shufflenetv2', 'resnet', 'resnetl', 'resnext'])
    parser.add_argument('--model-depth', type=int, default=10)
    parser.add_argument('--width-mult', type=float, default=0.5)
    parser.add_argument('--groups', type=int, default=3)
    parser.add_argument('--resnet-shortcut', default='A', choices=['A', 'B'])
    parser.add_argument('--classes', type=int, required=True)
    parser.add_argument('--duration', type=int, default=8, help='Frames per model input')
    parser.add_argument('--size', type=int, default=80, help='Square input resolution')
    parser.add_argument('--stride', type=int, default=2, help='Run model once every N frames')
    parser.add_argument('--threads', type=int, default=4, help='PyTorch CPU threads')
    parser.add_argument('--mean-dataset', choices=['activitynet', 'kinetics'], default='activitynet')
    parser.add_argument('--norm-value', type=int, choices=[1, 255], default=1)
    parser.add_argument('--std-norm', action='store_true')
    parser.add_argument('--display', action='store_true', help='Show preview; press q to quit')
    return parser.parse_args()


def build_model(args):
    # generate_model only needs this small compatibility subset on CPU.
    opt = SimpleNamespace(model=args.model, model_depth=args.model_depth,
                          width_mult=args.width_mult, groups=args.groups,
                          resnet_shortcut=args.resnet_shortcut,
                          resnext_cardinality=32, version=1.1,
                          n_classes=args.classes, n_finetune_classes=args.classes,
                          sample_size=args.size, sample_duration=args.duration,
                          no_cuda=True, pretrain_path='', modality='RGB',
                          pretrain_modality='RGB', ft_portion='complete')
    model, _ = generate_model(opt)
    return model


def load_checkpoint(model, checkpoint_path):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError('Checkpoint not found: {}'.format(path))
    # The original project checkpoints contain training metadata that PyTorch
    # 2.6+ does not allow under its new weights_only default. Only load a file
    # obtained from this project's official pretrained download or one you trust.
    try:
        checkpoint = torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument
        checkpoint = torch.load(str(path), map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    # Training used DataParallel, whose keys begin with "module.".  Pi models do not.
    state_dict = {key.removeprefix('module.'): value for key, value in state_dict.items()}
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            'Checkpoint and command-line architecture do not match. Check --model, '
            '--model-depth, --width-mult, --classes, --duration, and --size.\n' + str(error))


def read_labels(path, classes):
    if not path:
        return [str(index) for index in range(classes)]
    labels = Path(path).read_text(encoding='utf-8').splitlines()
    if len(labels) != classes:
        raise ValueError('--labels contains {} labels but --classes is {}'.format(len(labels), classes))
    return labels


def preprocess(frame_bgr, size, mean, std, norm_value):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # The legacy pipeline scales the short edge then center-crops.  Square camera
    # input is predictable and prevents a crop from failing on small frames.
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()
    tensor.div_(norm_value)
    tensor.sub_(torch.tensor(mean).view(3, 1, 1))
    tensor.div_(torch.tensor(std).view(3, 1, 1))
    return tensor


def main():
    args = parse_args()
    if args.duration < 1 or args.size < 16 or args.stride < 1 or args.threads < 1:
        raise ValueError('--duration, --stride and --threads must be positive; --size must be at least 16')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = build_model(args)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    labels = read_labels(args.labels, args.classes)
    mean = get_mean(args.norm_value, args.mean_dataset)
    std = get_std(args.norm_value) if args.std_norm else [1, 1, 1]
    source = int(args.source) if args.source.isdigit() else args.source
    camera = cv2.VideoCapture(source)
    if not camera.isOpened():
        raise RuntimeError('Unable to open camera/video source: {}'.format(args.source))

    frames = deque(maxlen=args.duration)
    frame_number = 0
    last_time = time.perf_counter()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frames.append(preprocess(frame, args.size, mean, std, args.norm_value))
            frame_number += 1
            if len(frames) == args.duration and frame_number % args.stride == 0:
                clip = torch.stack(tuple(frames), dim=1).unsqueeze(0)  # N,C,T,H,W
                with torch.inference_mode():
                    probabilities = F.softmax(model(clip), dim=1)[0]
                score, index = torch.max(probabilities, dim=0)
                now = time.perf_counter()
                fps = args.stride / max(now - last_time, 1e-9)
                last_time = now
                print('frame={} class={} score={:.3f} inference_fps={:.2f}'.format(
                    frame_number, labels[index.item()], score.item(), fps), flush=True)
            if args.display:
                cv2.imshow('Gesture recognition (q to quit)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        camera.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
