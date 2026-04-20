import argparse
import json
from pathlib import Path


def _norm_rel_path(p: str) -> str:
    return p.replace('\\', '/').lstrip('./')


def _resolve_split(image_rel: str, train_dir: Path, val_dir: Path):
    rel = _norm_rel_path(image_rel)
    name = Path(rel).name

    # Try exact relative path under split folder (e.g. train/xxx.png).
    train_exact = train_dir.parent / rel
    val_exact = val_dir.parent / rel
    if train_exact.exists():
        return 'train', str(Path('train') / Path(rel).name)
    if val_exact.exists():
        return 'val', str(Path('val') / Path(rel).name)

    # Fallback by file name under train/val.
    train_by_name = train_dir / name
    val_by_name = val_dir / name
    in_train = train_by_name.exists()
    in_val = val_by_name.exists()

    if in_train and not in_val:
        return 'train', str(Path('train') / name)
    if in_val and not in_train:
        return 'val', str(Path('val') / name)
    if in_train and in_val:
        # Prefer train if duplicated in both folders.
        return 'train', str(Path('train') / name)

    return None, None


def _new_coco_dict(categories):
    return {
        'images': [],
        'annotations': [],
        'categories': categories,
    }


def convert(input_json: Path, images_root: Path, out_train: Path, out_val: Path):
    train_dir = images_root / 'train'
    val_dir = images_root / 'val'
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f'Missing split dirs: {train_dir} or {val_dir}')

    data = json.loads(input_json.read_text(encoding='utf-8'))
    records = data.get('Dataset', [])
    if not isinstance(records, list):
        raise ValueError('Input json does not contain list field "Dataset"')

    # Build categories from ObjectCategory values.
    cat_name_to_id = {}
    for rec in records:
        for ann in rec.get('Annotations', []) or []:
            name = ann.get('ObjectCategory', 'object')
            if name not in cat_name_to_id:
                cat_name_to_id[name] = len(cat_name_to_id) + 1

    categories = [
        {'id': cid, 'name': name, 'supercategory': 'none'}
        for name, cid in sorted(cat_name_to_id.items(), key=lambda x: x[1])
    ]

    coco_train = _new_coco_dict(categories)
    coco_val = _new_coco_dict(categories)

    img_id_map = {'train': {}, 'val': {}}
    ann_id = 1
    skipped = 0

    for rec in records:
        image_rel = rec.get('Image')
        if not image_rel:
            skipped += 1
            continue

        split, split_file_name = _resolve_split(image_rel, train_dir, val_dir)
        if split is None:
            skipped += 1
            continue

        target = coco_train if split == 'train' else coco_val
        target_map = img_id_map[split]

        src_img_id = int(rec.get('ID', 0))
        if src_img_id not in target_map:
            # Infer image size from first annotation segmentation size if present.
            width = None
            height = None
            anns = rec.get('Annotations', []) or []
            if anns:
                seg = anns[0].get('segmentation')
                if isinstance(seg, dict) and isinstance(seg.get('size'), list) and len(seg['size']) == 2:
                    height, width = int(seg['size'][0]), int(seg['size'][1])

            if width is None or height is None:
                width = 0
                height = 0

            target_map[src_img_id] = src_img_id
            target['images'].append({
                'id': src_img_id,
                'file_name': split_file_name.replace('\\', '/'),
                'width': width,
                'height': height,
            })

        for ann in rec.get('Annotations', []) or []:
            bbox = ann.get('bbox')
            if not bbox or len(bbox) != 4:
                continue

            cat_name = ann.get('ObjectCategory', 'object')
            cat_id = cat_name_to_id[cat_name]
            seg = ann.get('segmentation', [])
            iscrowd = int(ann.get('iscrowd', 0))

            target['annotations'].append({
                'id': ann_id,
                'image_id': src_img_id,
                'category_id': cat_id,
                'bbox': [float(v) for v in bbox],
                'area': float(bbox[2]) * float(bbox[3]),
                'iscrowd': iscrowd,
                'segmentation': seg,
            })
            ann_id += 1

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)
    out_train.write_text(json.dumps(coco_train, ensure_ascii=False), encoding='utf-8')
    out_val.write_text(json.dumps(coco_val, ensure_ascii=False), encoding='utf-8')

    print(f'train images={len(coco_train["images"])} anns={len(coco_train["annotations"])}')
    print(f'val   images={len(coco_val["images"])} anns={len(coco_val["annotations"])}')
    print(f'categories={len(categories)} skipped_records={skipped}')
    print(f'output train: {out_train}')
    print(f'output val  : {out_val}')


def main():
    parser = argparse.ArgumentParser(description='Convert wafer Dataset JSON to COCO train/val json.')
    parser.add_argument('--input-json', required=True, help='Path to source json with top-level Dataset list.')
    parser.add_argument('--images-root', required=True, help='Root path containing images/train and images/val.')
    parser.add_argument('--out-train', required=True, help='Output COCO train json path.')
    parser.add_argument('--out-val', required=True, help='Output COCO val json path.')
    args = parser.parse_args()

    convert(
        input_json=Path(args.input_json),
        images_root=Path(args.images_root),
        out_train=Path(args.out_train),
        out_val=Path(args.out_val),
    )


if __name__ == '__main__':
    main()
