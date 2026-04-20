"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import time
import json
import datetime
import math
import re
from pathlib import Path

import torch

from ..misc import dist_utils, stats, plot_training_curves

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


class DetSolver(BaseSolver):

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        stats_str = next(iter(model_stats.keys()), '') if isinstance(model_stats, dict) else str(model_stats)
        gflops = None
        m = re.search(r'Model FLOPs:([0-9.]+)\s*GFLOPS', stats_str, flags=re.IGNORECASE)
        if m:
            gflops = float(m.group(1))

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches,
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        top1 = 0
        best_stat = {'epoch': -1, }
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            for k in test_stats:
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        best_vis_epochs = []  # [(precision, epoch, path), ...]
        best_metrics = {
            'P': {'value': -1.0, 'epoch': -1},
            'R': {'value': -1.0, 'epoch': -1},
            'mAP50': {'value': -1.0, 'epoch': -1},
            'mAP50-95': {'value': -1.0, 'epoch': -1},
        }
        infer_time_ms_history = []
        vis_interval = int(self.cfg.yaml_cfg.get('mosaic_vis_interval', 1))
        vis_interval = max(0, vis_interval)
        if dist_utils.is_main_process():
            if vis_interval > 0:
                print(f'mosaic_vis save interval: every {vis_interval} epoch(s)')
            else:
                print('mosaic_vis saving disabled (mosaic_vis_interval=0)')
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats, grad_percentages = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                teacher_model=self.teacher_model, # NEW: Pass teacher model to train_one_epoch
                vis_interval=vis_interval,
                vis_dir=str(self.output_dir),
                postprocessor=self.postprocessor,
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1
            if dist_utils.is_main_process() and hasattr(self.criterion, 'distill_adaptive_params') and \
                self.criterion.distill_adaptive_params and self.criterion.distill_adaptive_params.get('enabled', False):

                params = self.criterion.distill_adaptive_params
                default_weight = params.get('default_weight')

                avg_percentage = sum(grad_percentages) / len(grad_percentages) if grad_percentages else 0.0

                current_weight = self.criterion.weight_dict.get('loss_distill', 0.0)
                new_weight = current_weight
                reason = 'unchanged'

                if avg_percentage < 1e-6:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'reset_to_default_zero_grad'
                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'ema_phase_default'
                else:
                    rho = params['rho']
                    delta = params['delta']
                    lower_bound = rho - delta
                    upper_bound = rho + delta
                    if not (lower_bound <= avg_percentage <= upper_bound):
                        target_percentage = upper_bound if avg_percentage < lower_bound else lower_bound
                        if current_weight > 1e-6:
                            p_current = avg_percentage / 100.0
                            p_target = target_percentage / 100.0
                            numerator = p_target * (1.0 - p_current)
                            denominator = p_current * (1.0 - p_target)
                            if abs(denominator) >= 1e-9:
                                ratio = numerator / denominator
                                ratio = max(ratio, 0.1)  # clamp non-positive to 0.1
                                new_weight = current_weight * ratio
                                new_weight = min(max(new_weight, current_weight / 10.0), current_weight * 10.0)
                                reason = f'adjusted_to_{target_percentage:.2f}%'

                if abs(new_weight - current_weight) > 0:
                    self.criterion.weight_dict['loss_distill'] = new_weight
                print(f"Epoch {epoch}: avg encoder grad {avg_percentage:.2f}% | distill {current_weight:.6f} -> {new_weight:.6f} ({reason})")

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # Disable periodic snapshot checkpoints (checkpointXXXX.pth).
                # Keep only last.pth for resume and best_stg*.pth for best model tracking.
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            if self.output_dir and dist_utils.is_main_process():
                vis_label_path = Path(self.output_dir) / 'mosaic_vis' / f'epoch_{epoch:04d}_label.jpg'
                vis_pred_path = Path(self.output_dir) / 'mosaic_vis' / f'epoch_{epoch:04d}_pred.jpg'
                precision = None
                if 'coco_eval_bbox' in test_stats and len(test_stats['coco_eval_bbox']) > 1:
                    precision = float(test_stats['coco_eval_bbox'][1])

                if vis_label_path.exists() and vis_pred_path.exists():
                    if precision is not None and math.isfinite(precision) and precision >= 0:
                        best_vis_epochs.append((precision, epoch, (vis_label_path, vis_pred_path)))
                        best_vis_epochs.sort(key=lambda x: (x[0], x[1]), reverse=True)
                        while len(best_vis_epochs) > 2:
                            _, _, rm_paths = best_vis_epochs.pop()
                            for rm_path in rm_paths:
                                if rm_path.exists():
                                    rm_path.unlink()
                    else:
                        vis_label_path.unlink()
                        vis_pred_path.unlink()

            # TODO
            for k in test_stats:
                if not isinstance(test_stats[k], (list, tuple)):
                    continue
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print['epoch'] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')  # global best

                if best_stat['epoch'] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {'epoch': -1, }
                    self.ema.decay -= 0.0001
                    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            # Print key eval metrics every epoch.
            if 'coco_eval_bbox' in test_stats and len(test_stats['coco_eval_bbox']) > 8:
                map5095 = float(test_stats['coco_eval_bbox'][0])
                map50 = float(test_stats['coco_eval_bbox'][1])
                p = map50  # COCO summary proxy for precision
                r = float(test_stats['coco_eval_bbox'][8])

                if p > best_metrics['P']['value']:
                    best_metrics['P'] = {'value': p, 'epoch': epoch}
                if r > best_metrics['R']['value']:
                    best_metrics['R'] = {'value': r, 'epoch': epoch}
                if map50 > best_metrics['mAP50']['value']:
                    best_metrics['mAP50'] = {'value': map50, 'epoch': epoch}
                if map5095 > best_metrics['mAP50-95']['value']:
                    best_metrics['mAP50-95'] = {'value': map5095, 'epoch': epoch}

                infer_ms = float(test_stats.get('inference_time_ms', -1.0))
                if infer_ms >= 0:
                    infer_time_ms_history.append(infer_ms)

                infer_msg = f' | infer {infer_ms:.2f} ms/img' if infer_ms >= 0 else ''
                print(
                    f"Epoch {epoch}: P {p:.4f} | R {r:.4f} | "
                    f"mAP50 {map50:.4f} | mAP50-95 {map5095:.4f}{infer_msg}"
                )


            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            # Expose explicit P/R metrics for plotting convenience.
            if 'coco_eval_bbox' in test_stats and len(test_stats['coco_eval_bbox']) > 8:
                log_stats['test_precision'] = test_stats['coco_eval_bbox'][1]  # AP50 as precision proxy
                log_stats['test_recall'] = test_stats['coco_eval_bbox'][8]     # AR@100 as recall proxy

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        avg_infer_ms = sum(infer_time_ms_history) / len(infer_time_ms_history) if infer_time_ms_history else -1.0
        print('Final Summary:')
        print(f"Best P: {best_metrics['P']['value']:.4f} @ epoch {best_metrics['P']['epoch']}")
        print(f"Best R: {best_metrics['R']['value']:.4f} @ epoch {best_metrics['R']['epoch']}")
        print(f"Best mAP50: {best_metrics['mAP50']['value']:.4f} @ epoch {best_metrics['mAP50']['epoch']}")
        print(f"Best mAP50-95: {best_metrics['mAP50-95']['value']:.4f} @ epoch {best_metrics['mAP50-95']['epoch']}")
        if gflops is not None:
            print(f'Params: {n_parameters} | GFLOPs: {gflops:.4f}')
        else:
            print(f'Params: {n_parameters}')
        if avg_infer_ms >= 0:
            print(f'Avg inference time: {avg_infer_ms:.2f} ms/img')

        if self.output_dir and dist_utils.is_main_process():
            try:
                plot_training_curves(self.output_dir)
            except Exception as e:
                print(f'plot_training_curves failed: {e}')


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return


    def state_dict(self):
        """State dict, train/eval"""
        state = {}
        state['date'] = datetime.datetime.now().isoformat()

        # For resume
        state['last_epoch'] = self.last_epoch

        for k, v in self.__dict__.items():
            if k == 'teacher_model':
                continue
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict()

        return state