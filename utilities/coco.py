import base64
import json
import os
from pathlib import Path

from pycocotools import mask as pycocotools_mask


def to_serialized_rle(coco, anno):
    image = coco.imgs.get((anno.get('image_id')))
    w, h = image.get('width'), image.get('height')
    polygon = anno['segmentation']  # Assuming there is only one polygon per annotation
    rle_mask = pycocotools_mask.frPyObjects(polygon, h, w)
    anno['segmentation'] = rle_mask[0]
    # make JSON serializable
    anno['segmentation'].update(
        {'counts': base64.b64encode(anno['segmentation']['counts']).decode('utf-8')}
    )
    return anno


def deserialize_rle(anno, mask_format='bitmask'):
    if mask_format == 'bitmask':
        anno['segmentation'].update(
            {'counts': base64.b64decode(anno['segmentation']['counts']).decode('utf-8')}
        )
    if mask_format == 'polygon':
        from utilities.image import deserialize_contour_points
        anno.update(
            {'segmentation': [deserialize_contour_points(anno['segmentation']).tolist()]}
        )
    return anno


def dump_coco_file(args, coco, json_file_path, suffix=None):
    coco_file_name = f'{Path(json_file_path).stem}'
    if suffix:
        coco_file_name += f'__{suffix}'
    modified_json_file = os.path.join(args.output_dir,
                                      str(Path(json_file_path).parent.stem),
                                      f'{coco_file_name}.json')
    os.makedirs(str(Path(modified_json_file).parent), exist_ok=True)
    # Save the modified annotations to the specified file
    with open(modified_json_file, 'w') as output_file:
        json.dump(coco.dataset, output_file)
