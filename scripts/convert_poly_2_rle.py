import argparse
import base64
import concurrent.futures
import json
import os.path
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pycocotools import mask as pycocotools_mask
import sys
from datetime import datetime

from pycocotools.coco import COCO
from tqdm import tqdm

import utilities.filesystem.json_utils
from utilities.filesystem import json_utils, file_utils
from utilities.log.logger import setup_logger, logger
from PIL import Image
import matplotlib

sys.path.append('../utilities')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse arguments and save to JSON file')

    parser.add_argument('-i', '--input_dir', type=str, help='Path to the input root dir')
    parser.add_argument('-o', '--output_dir', default='./out', type=str, help='Path to the output root dir')
    parser.add_argument('-d', '--debug_mask', action='store_true', help='Save debug mask')
    return parser.parse_args()


def process_annotation(args, json_file, coco, anno):
    image = coco.imgs.get((anno.get('image_id')))
    w, h = image.get('width'), image.get('height')

    binary_mask = np.zeros((h, w), dtype=np.uint8)
    polygon = anno['segmentation'][0]  # Assuming there is only one polygon per annotation
    rle_mask = pycocotools_mask.frPyObjects([polygon], h, w)

    binary_mask = pycocotools_mask.decode(rle_mask)
    if args.debug_mask:
        mask_debug_dir = os.path.join(args.output_dir, 'ann_mask', str(Path(json_file).parent.stem))
        file_utils.generate_directory_if_not_exists(mask_debug_dir)
        plt.imshow(binary_mask)
        plt.savefig(f"{os.path.join(mask_debug_dir, 'ann_' + str(anno.get('id')))}.png")
    return anno


def process_json_annotations_mp(args, json_file, coco):
    annotations = coco.anns

    with tqdm(total=len(annotations), desc="Processing files") as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit file processing tasks to the executor
            rle_annotations = {executor.submit(process_annotation, args, json_file, coco, anno): anno for id, anno in
                               annotations.items()}

            # Retrieve results as they become available
            for future in concurrent.futures.as_completed(rle_annotations):
                rle_annotation = rle_annotations[future]
                try:
                    result = future.result()
                    # print(result)
                except Exception as e:
                    print(f"Error processing file {rle_annotation}: {e}")
                finally:
                    pbar.update(1)


def convert_rle(args):
    json_files_list = file_utils.find_files_with_suffix(args.input_dir, suffix='json')
    for json_file in json_files_list:
        coco = COCO(json_file)
        process_json_annotations_mp(args, json_file, coco)
    pass


def main():
    args = parse_arguments()
    json_args = (utilities.filesystem.json_utils
                 .dump_json_exec_args(args=args, output_json_path=os.path.join(args.output_dir, 'parsed_args.json')))
    args_msg = utilities.filesystem.json_utils.pretty_print_dict('parsed_args', json_args)
    logger = setup_logger(args)
    from utilities.log import color
    logger.info(args_msg, color.BLUE)
    convert_rle(args)


if __name__ == '__main__':
    main()
