# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------



import argparse
import datetime
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from util.tool import load_model
from util.plot_lr import draw_lr
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
from engine import train_one_epoch_mot
from models import build_model

from submit_dance import Detector, RuntimeTrackerBase

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_scheduler', default="OneCycle", type=str, choices=('StepLR', 'CosineAnneal', 'OneCycle'))
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=5e-3, type=float)
    parser.add_argument('--lr_div_fact', default=25, type=float)
    parser.add_argument('--lr_gamma', default=0.5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets',], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--save_period', default=50, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--meta_arch', default='deformable_detr', type=str)

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--accurate_ratio', default=False, action='store_true')


    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--num_anchors', default=1, type=int)

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--pretrained_backbone', default=False, action='store_true')
    parser.add_argument('--stem_norm_type', default="FrozenBatchNorm2d", type=str, choices=('BatchNorm2d', 'FrozenBatchNorm2d'))
    parser.add_argument('--output_norm_type', default="LayerNorm", type=str, choices=('BatchNorm2d', 'FrozenBatchNorm2d', 'LayerNorm'))
    
    parser.add_argument('--pretrained_dir', default='/tmp', type=str,)
    parser.add_argument('--enable_fpn', action='store_true')
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
    parser.add_argument('--input_width', default=1344, type=int, help='input image width')
    parser.add_argument('--input_height', default=896, type=int, help='input image height')
    
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--decoder_cross_self', default=False, action='store_true')
    parser.add_argument('--sigmoid_attn', default=False, action='store_true')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--cj', action='store_true')
    parser.add_argument('--extra_track_attn', action='store_true')
    parser.add_argument('--loss_normalizer', action='store_true')
    parser.add_argument('--max_size', default=1333, type=int)
    parser.add_argument('--val_width', default=800, type=int)
    parser.add_argument('--filter_ignore', action='store_true')
    parser.add_argument('--append_crowd', default=False, action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--mix_match', action='store_true',)
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--train_dataset', default='train', choices=('train', 'trainval'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--gt_file_train', type=str)
    parser.add_argument('--gt_file_val', type=str)
    parser.add_argument('--coco_path', default='/data/workspace/detectron2/datasets/coco/', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pretrained', default=None, help='resume from checkpoint')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--testset_inference_mode', default=False, action='store_true', help='after train, inference testset and save results')

    # end-to-end mot settings.
    parser.add_argument('--mot_path', default='/data/Dataset/mot', type=str)
    parser.add_argument('--det_db', default='', type=str)
    parser.add_argument('--input_video', default='figs/demo.mp4', type=str)
    parser.add_argument('--data_txt_path_train',
                        default='./datasets/data_path/detmot17.train', type=str,
                        help="path to dataset txt split")
    parser.add_argument('--data_txt_path_val',
                        default='./datasets/data_path/detmot17.train', type=str,
                        help="path to dataset txt split")
    parser.add_argument('--img_path', default='data/valid/JPEGImages/')

    parser.add_argument('--query_interaction_layer', default='QIM', type=str,
                        help="")
    parser.add_argument('--sample_mode', type=str, default='fixed_interval')
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--random_drop', type=float, default=0)
    parser.add_argument('--fp_ratio', type=float, default=0)
    parser.add_argument('--merger_dropout', type=float, default=0.1)
    parser.add_argument('--update_query_pos', action='store_true')

    parser.add_argument('--sampler_steps', type=int, nargs='*')
    parser.add_argument('--sampler_lengths', type=int, nargs='*')
    parser.add_argument('--exp_name', default='submit', type=str)
    parser.add_argument('--memory_bank_score_thresh', type=float, default=0.)
    parser.add_argument('--memory_bank_len', type=int, default=4)
    parser.add_argument('--memory_bank_type', type=str, default=None)
    parser.add_argument('--memory_bank_with_self_attn', action='store_true', default=False)

    parser.add_argument('--use_checkpoint', action='store_true', default=False)
    parser.add_argument('--query_denoise', type=float, default=0.)

    parser.add_argument('--resize_augment_scale', type=int, default=6)
    parser.add_argument('--infer_valid', type=int, default=6)
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--resize_type', default="original_ratio", type=str, choices=('original_ratio', 'fixed_size'))
    parser.add_argument('--update_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance', default=20, type=int)
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    collate_fn = utils.mot_collate_fn
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)

    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'CosineAnneal':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                              T_max = args.epochs, # Maximum number of iterations.
                             eta_min = 2e-5) # Minimum learning rate.
    elif args.lr_scheduler == 'OneCycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                       max_lr = [args.lr, args.lr_backbone, args.lr * args.lr_linear_proj_mult], # Upper learning rate boundaries in the cycle for each parameter group
                       steps_per_epoch = 1, # The number of steps per epoch to train for.
                       epochs = args.epochs, # The number of epochs to train for.
                       anneal_strategy = 'cos',  # Specifies the annealing strategy
                       div_factor = args.lr_div_fact) # Determines the initial learning rate via
    else:
        raise ValueError("Wrong lr scheduler arguments")
    ## draw lr scheduler
    # draw_lr(args, optimizer, lr_scheduler, 'lr_OneCycle.png')

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    if args.pretrained is not None:
        print("Try to load pretrained model")
        model_without_ddp = load_model(model_without_ddp, args.pretrained)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            print("start from ", checkpoint['epoch'] + 1, "epoch which resume from ", args.resume)
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            # print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()

    dataset_train.set_epoch(args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):
        torch.cuda.empty_cache()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch_mot(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_period == 0 or (((args.epochs >= 100 and (epoch + 1) > 100) or args.epochs < 100) and (epoch + 1) % 5 == 0):
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        dataset_train.step_epoch()

        import os

        model_without_ddp.track_embed.score_thr = args.update_score_threshold
        model_without_ddp.track_base = RuntimeTrackerBase(args.score_threshold, args.score_threshold, args.miss_tolerance)
        model_without_ddp.eval()

        sub_dir = 'DanceTrack/val'
        seq_nums = os.listdir(os.path.join(args.mot_path, sub_dir))
        if 'seqmap' in seq_nums:
            seq_nums.remove('seqmap')
        vids = [os.path.join(sub_dir, seq) for seq in seq_nums]

        rank = args.gpu
        ws = args.world_size

        vids = vids[rank::ws]

        out_fname = f'checkpoint{epoch:04}.pth'
        for vid in vids:
            det = Detector(args, model=model_without_ddp, vid=vid, out_fname=out_fname)
            det.detect(args.score_threshold)

        out_fname_base = os.path.basename(out_fname).split(".")[0]
        out_dir_abspath = os.path.abspath(args.output_dir)
        predict_path = os.path.abspath(os.path.join(args.output_dir, out_fname_base))
        out_txt_path = os.path.join(predict_path, 'results.txt')
        terminal_command = f"python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL val  --METRICS HOTA CLEAR Identity  --GT_FOLDER {args.mot_path}/DanceTrack/val --SEQMAP_FILE dancetrack/val_seqmap.txt --SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL {out_dir_abspath} --TRACKER_SUB_FOLDER {out_fname_base} --USE_PARALLEL True --NUM_PARALLEL_CORES 8 --PLOT_CURVES False --OUTPUT_SUMMARY True > {out_txt_path}"

        os.system(f"/bin/bash -c '{terminal_command}'")

        if args.testset_inference_mode:
            sub_dir = 'DanceTrack/test'
            seq_nums = os.listdir(os.path.join(args.mot_path, sub_dir))
            if 'seqmap' in seq_nums:
                seq_nums.remove('seqmap')
            vids = [os.path.join(sub_dir, seq) for seq in seq_nums]

            rank = args.gpu
            ws = args.world_size

            vids = vids[rank::ws]

            out_fname = f'testset-checkpoint{epoch:04}.pth'
            for vid in vids:
                det = Detector(args, model=model_without_ddp, vid=vid, out_fname=out_fname)
                det.detect(args.score_threshold)

        model_without_ddp.train()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
