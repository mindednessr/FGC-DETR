"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import sys
import math
import time
from typing import Iterable
from pathlib import Path

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_train_batch_sample, save_pred_batch_sample

def _compute_encoder_transformer_grad_percentage(model: torch.nn.Module) -> float:
    """Compute percentage of gradients attributed to encoder transformer only.
    This avoids collecting/printing any other stats for speed.
    """
    total_l1 = 0.0
    enc_l1 = 0.0
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        val = grad.detach().abs().sum().item()
        total_l1 += val
        # Support both DDP ('module.') and non-DDP naming
        if name.startswith('module.encoder.encoder'):
            enc_l1 += val
    if total_l1 <= 0.0 or not math.isfinite(total_l1):
        return 0.0
    return 100.0 * enc_l1 / total_l1


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    # Gradient Analysis
    encoder_grad_percentages = []
    cur_iters = epoch * len(data_loader)

    teacher_model = kwargs.get('teacher_model', None)
    vis_interval = kwargs.get('vis_interval', 0)
    vis_dir = kwargs.get('vis_dir', None)
    postprocessor = kwargs.get('postprocessor', None)
    vis_saved_this_epoch = False

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        need_vis = (
            not vis_saved_this_epoch and vis_interval and vis_dir and i == 0
            and epoch % int(vis_interval) == 0 and dist_utils.is_main_process()
        )
        vis_label_path = None
        vis_pred_path = None
        if (
            need_vis
        ):
            vis_label_path = Path(vis_dir) / 'mosaic_vis' / f'epoch_{epoch:04d}_label.jpg'
            vis_pred_path = Path(vis_dir) / 'mosaic_vis' / f'epoch_{epoch:04d}_pred.jpg'
            try:
                save_train_batch_sample(samples, targets, vis_label_path)
            except Exception as e:
                print(f'save_train_batch_sample failed: {e}')
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        teacher_encoder_output_for_distillation = None
        if teacher_model is not None:
            with torch.no_grad():
                teacher_encoder_output_for_distillation = teacher_model(samples).detach()

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets,
                                teacher_encoder_output=teacher_encoder_output_for_distillation)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    new_key = key.replace('module.', '')
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # Collect gradient
            if dist_utils.is_main_process() and hasattr(criterion, 'distill_adaptive_params') and \
               getattr(criterion, 'distill_adaptive_params') and \
               criterion.distill_adaptive_params.get('enabled', False):
                pct = _compute_encoder_transformer_grad_percentage(model)
                encoder_grad_percentages.append(pct)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets,
                            teacher_encoder_output=teacher_encoder_output_for_distillation) # NEW kwarg
            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            # Collect gradient
            if dist_utils.is_main_process() and hasattr(criterion, 'distill_adaptive_params') and \
               getattr(criterion, 'distill_adaptive_params') and \
               criterion.distill_adaptive_params.get('enabled', False):
                pct = _compute_encoder_transformer_grad_percentage(model)
                encoder_grad_percentages.append(pct)

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        if need_vis and postprocessor is not None and vis_pred_path is not None:
            try:
                with torch.no_grad():
                    vis_outputs = {
                        'pred_logits': outputs['pred_logits'].detach(),
                        'pred_boxes': outputs['pred_boxes'].detach(),
                    }
                    h, w = int(samples.shape[-2]), int(samples.shape[-1])
                    orig_target_sizes = torch.tensor([[w, h]] * samples.shape[0], device=device)
                    vis_results = postprocessor(vis_outputs, orig_target_sizes)
                save_pred_batch_sample(samples, vis_results, vis_pred_path, score_thr=0.25)
                vis_saved_this_epoch = True
            except Exception as e:
                print(f'save_pred_batch_sample failed: {e}')

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = float(sum(loss_dict_reduced.values()).detach().item())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # Keep console output focused on key losses only.
        display_losses = {}
        for k in ('loss_distill', 'loss_bbox', 'loss_giou', 'loss_mal'):
            if k in loss_dict_reduced:
                display_losses[k] = float(loss_dict_reduced[k].detach().item())

        metric_logger.update(loss=loss_value, **display_losses)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value, global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', float(v.detach().item()), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, encoder_grad_percentages


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    total_infer_time = 0.0
    total_images = 0

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        t0 = time.time()
        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)
        total_infer_time += time.time() - t0
        total_images += int(samples.shape[0])

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    if total_images > 0:
        stats['inference_time_ms'] = (total_infer_time / total_images) * 1000.0

    return stats, coco_evaluator
