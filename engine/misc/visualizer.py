""""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import PIL
import torch
import torch.utils.data
import torchvision
from pathlib import Path
torchvision.disable_beta_transforms_warning()

__all__ = ['show_sample', 'save_train_batch_sample', 'save_pred_batch_sample']

DEFAULT_BOX_WIDTH = 4
DEFAULT_FONT_SIZE = 24


def _class_name_from_label(label_id: int) -> str:
    # Wafer dataset is single-class crack; keep fallback for non-zero labels.
    return 'crack' if int(label_id) == 0 else f'class{int(label_id)}'


def _draw_boxes(image_u8, boxes, labels, color, width, font_size):
    from torchvision.utils import draw_bounding_boxes

    font_path = None
    for p in (
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/calibri.ttf',
        'C:/Windows/Fonts/msyh.ttc',
    ):
        if Path(p).exists():
            font_path = p
            break

    try:
        kwargs = dict(
            labels=labels,
            colors=color,
            width=width,
        )
        if font_path is not None:
            kwargs['font'] = font_path
            kwargs['font_size'] = font_size
        return draw_bounding_boxes(
            image_u8,
            boxes,
            **kwargs,
        )
    except TypeError:
        return draw_bounding_boxes(image_u8, boxes, labels=labels, colors=color, width=width)

def show_sample(sample):
    """for coco dataset/dataloader
    """
    import matplotlib.pyplot as plt
    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)

    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    fig.show()
    plt.show()


def save_train_batch_sample(samples, targets, save_path, color='blue', width=DEFAULT_BOX_WIDTH, font_size=DEFAULT_FONT_SIZE):
    """Save a 3x3 grid from a training batch with GT boxes drawn."""
    from pathlib import Path
    from torchvision.utils import make_grid

    if samples is None or len(samples) == 0 or targets is None or len(targets) == 0:
        return

    max_items = min(9, len(samples), len(targets))
    vis_images = []

    for idx in range(max_items):
        image = samples[idx].detach().cpu().clamp(0, 1)
        target = targets[idx]
        boxes = target.get('boxes', None)
        labels = target.get('labels', None)

        image_u8 = (image * 255.0).to(torch.uint8)

        if boxes is not None and boxes.numel() > 0:
            boxes = boxes.detach().cpu().float()
            h, w = int(image.shape[1]), int(image.shape[2])

            # Dataset pipeline typically uses normalized cxcywh boxes.
            if boxes.max() <= 1.5 and boxes.min() >= -0.5:
                cx, cy, bw, bh = boxes.unbind(-1)
                x1 = (cx - bw / 2.0) * w
                y1 = (cy - bh / 2.0) * h
                x2 = (cx + bw / 2.0) * w
                y2 = (cy + bh / 2.0) * h
                boxes = torch.stack([x1, y1, x2, y2], dim=-1)

            boxes[:, 0::2].clamp_(0, w - 1)
            boxes[:, 1::2].clamp_(0, h - 1)
            boxes = boxes.to(torch.int64)

            text = None
            if labels is not None:
                labels = labels.detach().cpu().to(torch.int64)
                text = [_class_name_from_label(int(x.item())) for x in labels]

            vis = _draw_boxes(image_u8, boxes, labels=text, color=color, width=width, font_size=font_size)
        else:
            vis = image_u8

        vis_images.append(vis)

    if not vis_images:
        return

    grid = make_grid(vis_images, nrow=3, padding=4)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(grid.permute(1, 2, 0).numpy()).save(str(save_path))


def save_pred_batch_sample(samples, results, save_path, score_thr=0.25, color='blue', width=DEFAULT_BOX_WIDTH, font_size=DEFAULT_FONT_SIZE):
    """Save a 3x3 prediction visualization grid from a batch.

    Args:
        samples: Tensor[B, C, H, W] in [0, 1]
        results: list[dict] from postprocessor with keys boxes/scores/labels
    """
    from pathlib import Path
    from torchvision.utils import make_grid

    if samples is None or len(samples) == 0 or results is None or len(results) == 0:
        return

    max_items = min(9, len(samples), len(results))
    vis_images = []

    for idx in range(max_items):
        image = samples[idx].detach().cpu().clamp(0, 1)
        pred = results[idx]
        boxes = pred.get('boxes', None)
        scores = pred.get('scores', None)
        labels = pred.get('labels', None)

        image_u8 = (image * 255.0).to(torch.uint8)

        if boxes is not None and boxes.numel() > 0:
            boxes = boxes.detach().cpu().float()
            h, w = int(image.shape[1]), int(image.shape[2])
            boxes[:, 0::2].clamp_(0, w - 1)
            boxes[:, 1::2].clamp_(0, h - 1)

            if scores is not None:
                scores = scores.detach().cpu().float()
                keep = scores >= float(score_thr)
                boxes = boxes[keep]
                labels = labels[keep] if labels is not None else None
                scores = scores[keep]
            else:
                scores = None

            if boxes.numel() > 0:
                text = None
                if labels is not None:
                    labels = labels.detach().cpu().to(torch.int64)
                    if scores is not None:
                        text = [f"{_class_name_from_label(int(c.item()))} {float(s):.1f}" for c, s in zip(labels, scores)]
                    else:
                        text = [_class_name_from_label(int(c.item())) for c in labels]

                boxes = boxes.to(torch.int64)
                vis = _draw_boxes(image_u8, boxes, labels=text, color=color, width=width, font_size=font_size)
            else:
                vis = image_u8
        else:
            vis = image_u8

        vis_images.append(vis)

    if not vis_images:
        return

    grid = make_grid(vis_images, nrow=3, padding=4)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(grid.permute(1, 2, 0).numpy()).save(str(save_path))
