# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
MOT dataset which returns image_id for evaluation.
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw
import copy
import datasets.transforms as T
from models.structures import Instances

from random import choice, randint


def is_crowd(ann):
    return 'extra' in ann and 'ignore' in ann['extra'] and ann['extra']['ignore'] == 1


class DetMOTDetection:
    def __init__(self, args, data_txt_path: str, seqs_folder, transform):
        self.args = args
        self.transform = transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.video_dict = {}
        self.mot_path = args.mot_path

        self.labels_full = defaultdict(lambda : defaultdict(list))
        def add_mot_folder(split_dir):
            print("Adding", split_dir)
            for vid in os.listdir(os.path.join(self.mot_path, split_dir)):
                if 'seqmap' == vid:
                    continue
                vid = os.path.join(split_dir, vid)
                if 'DPM' in vid or 'FRCNN' in vid:
                    print(f'filter {vid}')
                    continue
                gt_path = os.path.join(self.mot_path, vid, 'gt', 'gt.txt')
                for l in open(gt_path):
                    t, i, *xywh, mark, label = l.strip().split(',')[:8]
                    t, i, mark, label = map(int, (t, i, mark, label))
                    if mark == 0:
                        continue
                    if label in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
                        continue
                    else:
                        crowd = False
                    x, y, w, h = map(float, (xywh))
                    self.labels_full[vid][t].append([x, y, w, h, i, crowd])

        if args.train_dataset == 'train':
            add_mot_folder("DanceTrack/train")
        elif args.train_dataset == 'trainval':
            add_mot_folder("DanceTrack/train")
            add_mot_folder("DanceTrack/val")
        else:
            print("Warning, please check which dataset you want to train with. We use only train dataset in this time.")
            add_mot_folder("DanceTrack/train")

        vid_files = list(self.labels_full.keys())

        self.indices = []
        self.vid_tmax = {}
        for vid in vid_files:
            self.video_dict[vid] = len(self.video_dict)
            t_min = min(self.labels_full[vid].keys())
            t_max = max(self.labels_full[vid].keys()) + 1
            self.vid_tmax[vid] = t_max - 1
            for t in range(t_min, t_max - self.num_frames_per_batch):
                self.indices.append((vid, t))
        print(f"Found DanceTrack {len(vid_files)} videos, {len(self.indices)} frames")

        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths
        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths))
        self.period_idx = 0

        # crowdhuman
        self.ch_dir = Path(args.mot_path) / 'crowdhuman'
        self.ch_indices = []
        if args.append_crowd:
            for line in open(self.ch_dir / f"annotation_trainval.odgt"):
                datum = json.loads(line)
                boxes = [ann['fbox'] for ann in datum['gtboxes'] if not is_crowd(ann)]
                self.ch_indices.append((datum['ID'], boxes))
        # self.ch_indices = self.ch_indices + self.ch_indices
        print(f"Found CrowdHuman {len(self.ch_indices)} images")

        if args.det_db:
            with open(os.path.join(args.mot_path, args.det_db)) as f:
                self.det_db = json.load(f)
        else:
            self.det_db = defaultdict(list)

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print("set epoch: epoch {} period_idx={}".format(epoch, self.period_idx))
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        n_gt = len(targets['labels'])
        gt_instances.boxes = targets['boxes'][:n_gt]
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        return gt_instances

    def load_crowd(self, index):
        ID, boxes = self.ch_indices[index]
        boxes = copy.deepcopy(boxes)
        img = Image.open(self.ch_dir / 'Images' / f'{ID}.jpg')

        w, h = img._size
        n_gts = len(boxes)
        scores = [0. for _ in range(len(boxes))]
        for line in self.det_db[f'crowdhuman/train_image/{ID}.txt']:
            *box, s = map(float, line.split(','))
            boxes.append(box)
            scores.append(s)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        areas = boxes[..., 2:].prod(-1)
        boxes[:, 2:] += boxes[:, :2]

        target = {
            'boxes': boxes,
            'scores': torch.as_tensor(scores),
            'labels': torch.zeros((n_gts, ), dtype=torch.long),
            'iscrowd': torch.zeros((n_gts, ), dtype=torch.bool),
            'image_id': torch.tensor([0]),
            'area': areas,
            'obj_ids': torch.arange(n_gts),
            'size': torch.as_tensor([h, w]),
            'orig_size': torch.as_tensor([h, w]),
            'dataset': "CrowdHuman",
        }
        rs = T.FixedMotRandomShift(self.num_frames_per_batch)
        return rs([img], [target])

    def _pre_single_frame(self, vid, idx: int):
        img_path = os.path.join(self.mot_path, vid, 'img1', f'{idx:08d}.jpg')
        img = Image.open(img_path)
        targets = {}
        w, h = img._size
        assert w > 0 and h > 0, "invalid image {} with shape {} {}".format(img_path, w, h)
        obj_idx_offset = self.video_dict[vid] * 100000  # 100000 unique ids is enough for a video.

        targets['dataset'] = 'MOT17'
        targets['boxes'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['obj_ids'] = []
        targets['scores'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])
        for *xywh, id, crowd in self.labels_full[vid][idx]:
            targets['boxes'].append(xywh)
            assert not crowd
            targets['iscrowd'].append(crowd)
            targets['labels'].append(0)
            targets['obj_ids'].append(id + obj_idx_offset)
            targets['scores'].append(1.)
        txt_key = os.path.join(vid, 'img1', f'{idx:08d}.txt')
        for line in self.det_db[txt_key]:
            *box, s = map(float, line.split(','))
            targets['boxes'].append(box)
            targets['scores'].append(s)

        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'])
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'], dtype=torch.float64)
        targets['scores'] = torch.as_tensor(targets['scores'])
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        targets['boxes'][:, 2:] += targets['boxes'][:, :2]
        return img, targets

    def _get_sample_range(self, start_idx):

        # take default sampling method for normal dataset.
        assert self.sample_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = start_idx, start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1, sample_interval
        return default_range

    def pre_continuous_frames(self, vid, indices):
        return zip(*[self._pre_single_frame(vid, i) for i in indices])

    def sample_indices(self, vid, f_index):
        assert self.sample_mode == 'random_interval'
        rate = randint(1, self.sample_interval + 1)
        tmax = self.vid_tmax[vid]
        ids = [f_index + rate * i for i in range(self.num_frames_per_batch)]
        return [min(i, tmax) for i in ids]

    def __getitem__(self, idx):
        if idx < len(self.indices):
            vid, f_index = self.indices[idx]
            indices = self.sample_indices(vid, f_index)
            images, targets = self.pre_continuous_frames(vid, indices)
        else:
            images, targets = self.load_crowd(idx - len(self.indices))
        # images[0].save('output_image_0.jpg', 'JPEG')
        if self.transform is not None:
            images, targets = self.transform(images, targets)
        # import torchvision.transforms as transforms
        # transforms.ToPILImage()(images[4]).save('transform_output_image_4.jpg', 'JPEG')
        gt_instances, proposals = [], []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
            n_gt = len(targets_i['labels'])
            proposals.append(torch.cat([
                targets_i['boxes'][n_gt:],
                targets_i['scores'][n_gt:, None],
            ], dim=1))
        return {
            'imgs': images,
            'gt_instances': gt_instances,
            'proposals': proposals,
        }

    def __len__(self):
        return len(self.indices) + len(self.ch_indices)


class DetMOTDetectionValidation(DetMOTDetection):
    def __init__(self, args, seqs_folder, transform):
        args.data_txt_path = args.val_data_txt_path
        super().__init__(args, seqs_folder, transform)


def make_transforms_for_mot17(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if args.resize_augment_scale == 6:
        scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992] 
    elif args.resize_augment_scale == 8:
        scales = [544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056] 
    elif args.resize_augment_scale == 4:
        scales = [544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896]
    elif args.resize_augment_scale == 2:
        scales = [448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896]
    scales = [224]
    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomHorizontalFlip(),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                # T.MotRandomResize([800], max_size=1333),
                # T.MotRandomResize([(args.input_width, args.input_height)]),
                T.MotCompose([
                    T.MotRandomResize([800, 1000, 1200]),
                    T.FixedMotRandomCrop(800, 1200),
                    # T.MotRandomResize([(args.input_width, args.input_height)]),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            T.MOTHSV(),
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_transform(args, image_set):
    mot17_train = make_transforms_for_mot17('train', args)
    mot17_test = make_transforms_for_mot17('val', args)

    if image_set == 'train':
        return mot17_train
    elif image_set == 'val':
        return mot17_test
    else:
        raise NotImplementedError()


def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'provided MOT path {root} does not exist'
    transform = build_transform(args, image_set)
    if image_set == 'train':
        data_txt_path = args.data_txt_path_train
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, transform=transform)
    if image_set == 'val':
        data_txt_path = args.data_txt_path_val
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, transform=transform)
    return dataset
