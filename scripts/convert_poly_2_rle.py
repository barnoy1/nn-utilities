import argparse
import base64
import concurrent.futures
import json
import os.path
from pathlib import Path

import cv2
from pycocotools import mask as pycocotools_mask
import sys

from pycocotools.coco import COCO
from tqdm import tqdm

import utilities.json_utils
from utilities import file_utils, image_utils
from utilities.log.logger import setup_logger, logger

from utilities.image_utils import bin_to_color

sys.path.append('../utilities')


def to_rle(args, json_file_path, coco, anno):
    image = coco.imgs.get((anno.get('image_id')))
    w, h = image.get('width'), image.get('height')
    polygon = anno['segmentation']  # Assuming there is only one polygon per annotation
    rle_mask = pycocotools_mask.frPyObjects(polygon, h, w)
    anno['segmentation'] = rle_mask[0]
    binary_mask = pycocotools_mask.decode(rle_mask)
    # make JSON serializable
    anno['segmentation'].update(
        {'counts': base64.b64encode(anno['segmentation']['counts']).decode('utf-8')}
    )
    '''
    #1. encode image from mask
    encoded_mask = pycocotools_mask.encode(np.asfortanarray(binary_mask))
    #2. encoded b'string
    encoded_mask.update(
        {'counts': base64.b64encode(anno['segmentation']['counts']).decode('utf-8')})
    #3. load from b'string
    encoded_mask.update(
        {'counts': base64.b64decode(encoded_mask['counts'])})
    #4. load decoded image from mask
    pycocotools_mask.decode(encoded_mask)
    '''

    if args.debug_mask:
        mask_debug_dir = os.path.join(args.output_dir, str(Path(json_file_path).parent.stem))
        file_utils.generate_directory_if_not_exists(mask_debug_dir)
        is_empty = image_utils.is_empty(image=binary_mask)
        assert is_empty == False, f"annotation id {anno.get('image_id')} is empty"
        mask_file_name = f"{os.path.join(mask_debug_dir, 'ann_' + str(anno.get('id')))}.png"
        cv2.imwrite(mask_file_name, bin_to_color(binary_mask))
    return anno


def process_json_annotations_mp(args, json_file_path):
    coco = COCO(json_file_path)
    annotations = coco.dataset['annotations']
    with tqdm(total=len(annotations), desc="Processing annotations") as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit file processing tasks to the executor
            rle_annotations = {executor.submit(to_rle, args, json_file_path, coco, anno): anno for anno in annotations}

            # Retrieve results as they become available
            for future in concurrent.futures.as_completed(rle_annotations):
                rle_annotation = rle_annotations[future]
                try:
                    result = future.result()
                    # print(result)
                except Exception as e:
                    logger.error(f"Error processing file {rle_annotation}: {e}")
                finally:
                    pbar.update(1)

    from utilities.coco_utils import dump_coco_file
    dump_coco_file(args, coco, json_file_path, suffix='bitmask')


def process_json_annotations(args, json_file_path):
    coco = COCO(json_file_path)
    annotations = coco.dataset['annotations']
    rle_annotations = []
    with tqdm(total=len(annotations), desc="Processing annotations") as pbar:

        for anno in annotations:
            try:
                rle_anno = to_rle(args, json_file_path, coco, anno)
                rle_annotations.append(rle_anno)
            except Exception as e:
                logger.error(f"Error processing annotation {anno.get('id')}: {e}")
            finally:
                pbar.update(1)

    from utilities.coco_utils import dump_coco_file
    dump_coco_file(args, coco, json_file_path, suffix='bitmask')


def convert_rle(args):
    json_files_list = file_utils.find_files_with_suffix(args.input_dir, suffix='json')
    if args.multi_processing:
        [process_json_annotations_mp(args, json_file_path) for json_file_path in json_files_list]
    else:
        [process_json_annotations(args, json_file_path) for json_file_path in json_files_list]


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse arguments and save to JSON file')

    parser.add_argument('-i', '--input_dir', type=str, help='Path to the input root dir')
    parser.add_argument('-o', '--output_dir', default='./out', type=str, help='Path to the output root dir')
    parser.add_argument('-d', '--debug_mask', action='store_true', help='Save debug mask')
    parser.add_argument('-mp', '--multi_processing', action='store_true', help='Save debug mask')
    return parser.parse_args()


def main():
    args = parse_arguments()
    json_args = (utilities.json_utils
                 .dump_json_exec_args(args=args, output_json_path=os.path.join(args.output_dir, 'parsed_args.json')))
    args_msg = utilities.json_utils.pretty_print_dict('parsed_args', json_args)
    logger = setup_logger(args)
    from utilities.log import color
    logger.info(args_msg, color.BLUE)
    convert_rle(args)


if __name__ == '__main__':
    main()
