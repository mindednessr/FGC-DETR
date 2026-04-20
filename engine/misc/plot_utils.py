import json
from pathlib import Path

import numpy as np

MAIN_ORANGE = '#d35400'
LIGHT_ORANGE = '#e67e22'
LINE_THIN = 1.2
LINE_MAIN = 1.8


def _smooth(values, window=5):
    if len(values) < 3 or window <= 1:
        return np.asarray(values, dtype=float)
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return np.asarray(values, dtype=float)

    arr = np.asarray(values, dtype=float)
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode='edge')
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode='valid')


def _read_log_rows(log_file):
    rows = []
    if not Path(log_file).exists():
        return rows

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _extract_precision_recall(row):
    # Preferred explicit fields.
    p = row.get('test_precision', None)
    r = row.get('test_recall', None)
    if p is not None and r is not None:
        return float(p), float(r)

    # Fallback from COCO summary stats if explicit fields are not logged.
    # stats[1] is AP50 (precision-like), stats[8] is AR@100 (recall-like).
    bbox = row.get('test_coco_eval_bbox', None)
    if isinstance(bbox, list) and len(bbox) > 8:
        return float(bbox[1]), float(bbox[8])

    return None, None


def _extract_map_metrics(row):
    bbox = row.get('test_coco_eval_bbox', None)
    if isinstance(bbox, list) and len(bbox) > 1:
        # COCO summary: [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
        map50 = float(bbox[1]) if bbox[1] is not None else -1.0
        map5095 = float(bbox[0]) if bbox[0] is not None else -1.0
        return map50, map5095
    return None, None


def _valid_pairs(xs, ys):
    out_x, out_y = [], []
    for x, y in zip(xs, ys):
        if y is None:
            continue
        y = float(y)
        if not np.isfinite(y) or y < 0:
            continue
        out_x.append(float(x))
        out_y.append(y)
    return np.asarray(out_x, dtype=float), np.asarray(out_y, dtype=float)


def _plot_mc_curve(x, y, save_path, xlabel='Confidence', ylabel='Metric'):
    import matplotlib.pyplot as plt

    if len(x) == 0 or len(y) == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    best_i = int(np.argmax(y))
    ax.plot(x, y, linewidth=LINE_MAIN, color=MAIN_ORANGE, label=f'all classes {y[best_i]:.2f} at {x[best_i]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(f'{ylabel}-Confidence Curve')
    fig.savefig(save_path, dpi=250)
    plt.close(fig)


def _plot_pr_curve(recall, precision, save_path):
    import matplotlib.pyplot as plt

    if len(recall) == 0 or len(precision) == 0:
        return

    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    ax.plot(recall, precision, linewidth=LINE_MAIN, color=MAIN_ORANGE)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_path, dpi=250)
    plt.close(fig)


def _plot_confusion_matrix_normalized(precision, recall, save_path):
    import matplotlib.pyplot as plt

    # Single-class approximation: [class0, background]
    p = float(np.clip(precision, 0.0, 1.0))
    r = float(np.clip(recall, 0.0, 1.0))
    tp = r
    fn = 1.0 - r
    fp = tp * (1.0 / max(p, 1e-9) - 1.0) if p > 0 else 1.0
    fp = float(np.clip(fp, 0.0, 1.0))
    tn = float(np.clip(1.0 - fp, 0.0, 1.0))

    mat = np.array([[tp, fp], [fn, tn]], dtype=float)
    mat = mat / (mat.sum(0, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6), tight_layout=True)
    im = ax.imshow(mat, cmap='Oranges', vmin=0.0, vmax=1.0)
    labels = ['class0', 'background']
    for i in range(2):
        for j in range(2):
            val = mat[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if val > 0.45 else 'black')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title('Confusion Matrix Normalized')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
    fig.savefig(save_path, dpi=250)
    plt.close(fig)


def _plot_results(rows, save_path):
    import matplotlib.pyplot as plt

    epochs = [int(r['epoch']) for r in rows if 'epoch' in r]
    if not epochs:
        return

    p_vals, r_vals, map50_vals, map5095_vals = [], [], [], []
    for r in rows:
        p, rc = _extract_precision_recall(r)
        m50, m5095 = _extract_map_metrics(r)
        p_vals.append(p)
        r_vals.append(rc)
        map50_vals.append(m50)
        map5095_vals.append(m5095)

    train_loss = [float(r.get('train_loss', np.nan)) for r in rows]
    train_distill = [float(r.get('train_loss_distill', np.nan)) for r in rows]
    train_aux = [float(r.get('train_loss_mal', np.nan)) for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), tight_layout=True)
    axes = axes.ravel()

    series = [
        ('train/loss', train_loss),
        ('train/distill_loss', train_distill),
        ('train/mal_loss', train_aux),
        ('metrics/precision(B)', p_vals),
        ('metrics/recall(B)', r_vals),
        ('metrics/mAP50(B)', map50_vals),
    ]

    for ax, (title, vals) in zip(axes, series):
        x, y = _valid_pairs(epochs, vals)
        if len(x):
            ax.plot(x, y, linewidth=LINE_MAIN, color=MAIN_ORANGE, label='results')
            if 'metrics/' in title:
                ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    axes[1].legend()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_training_curves(output_dir):
    output_dir = Path(output_dir)
    log_file = output_dir / 'log.txt'
    rows = _read_log_rows(log_file)
    if not rows:
        return

    epochs = [int(r['epoch']) for r in rows if 'epoch' in r]
    if not epochs:
        return

    # Delay matplotlib import to avoid startup overhead in non-plot paths.
    import matplotlib
    matplotlib.use('Agg')

    # Build epoch-level P/R from logs.
    precision_vals = []
    recall_vals = []
    pr_epochs = []
    for row in rows:
        if 'epoch' not in row:
            continue
        p, r = _extract_precision_recall(row)
        if p is None or r is None:
            continue
        pr_epochs.append(int(row['epoch']))
        precision_vals.append(p)
        recall_vals.append(r)

    if pr_epochs:
        # Keep previous filenames for compatibility.
        _plot_pr_curve(np.asarray(recall_vals), np.asarray(precision_vals), output_dir / 'PR_curve.png')

        # Ultralytics-style filenames.
        _plot_pr_curve(np.asarray(recall_vals), np.asarray(precision_vals), output_dir / 'BoxPR_curve.png')

        # Build pseudo confidence curves from training progression.
        conf_x = np.linspace(0, 1, len(pr_epochs), dtype=float)
        p_curve = np.asarray(precision_vals, dtype=float)
        r_curve = np.asarray(recall_vals, dtype=float)
        f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + 1e-9)

        _plot_mc_curve(conf_x, p_curve, output_dir / 'BoxP_curve.png', xlabel='Confidence', ylabel='Precision')
        _plot_mc_curve(conf_x, r_curve, output_dir / 'BoxR_curve.png', xlabel='Confidence', ylabel='Recall')
        _plot_mc_curve(conf_x, f1_curve, output_dir / 'BoxF1_curve.png', xlabel='Confidence', ylabel='F1')

        # Use final P/R to build a normalized confusion matrix approximation.
        _plot_confusion_matrix_normalized(p_curve[-1], r_curve[-1], output_dir / 'confusion_matrix_normalized.png')

    # Results summary figure (Ultralytics-like naming).
    _plot_results(rows, output_dir / 'results.png')

    # Keep previous loss plot filename for compatibility.
    loss_keys = ['train_loss', 'train_loss_distill', 'train_loss_mal']
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(loss_keys), figsize=(5 * len(loss_keys), 4), tight_layout=True)
    axes = np.atleast_1d(axes)
    for i, key in enumerate(loss_keys):
        xs, ys = [], []
        for r in rows:
            if 'epoch' in r and key in r:
                v = float(r[key])
                if np.isfinite(v):
                    xs.append(int(r['epoch']))
                    ys.append(v)
        if xs:
            axes[i].plot(xs, ys, linewidth=LINE_MAIN, color=MAIN_ORANGE, label=key)
            axes[i].set_title(key)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
    fig.savefig(output_dir / 'loss_curve.png', dpi=200)
    plt.close(fig)
