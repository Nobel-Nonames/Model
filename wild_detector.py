import sys
import argparse
import os
import os.path as osp
from pathlib import Path
import json
import time

import traceback
import logging
from logging.handlers import RotatingFileHandler

import torch
from ultralytics import YOLO # cls-infer v8
#from mmpretrain import inference_model # VIG
import cv2
import numpy as np

DEFAULT_DETECTOR_LABEL_MAP = {
    # '1': 'Animal',
    '1': 'Etc',
    '2': 'Person',
    '3': 'Vehicle'
}

# ###################################################################################################

def set_logger():
    logger = logging.getLogger('Wild_detection')
    logger.setLevel(logging.INFO)

    if not osp.exists('logs'):
        os.makedirs('logs', exist_ok=True)

    file_logger = RotatingFileHandler('logs/Wild_detection_status.log', maxBytes=10240, backupCount=10)
    file_logger.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)s: %(message)s'))
    file_logger.setLevel(logging.INFO)

    logger.addHandler(file_logger)

    return logger

def parse_args():
    parser = argparse.ArgumentParser(description='WKIT')
    parser.add_argument('--input_dir', default = 'input')
    parser.add_argument('--mv_dir', default = 'mv')
    parser.add_argument('--rst_dir', default = 'rst')
    parser.add_argument('--crop_dir')
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--clss_threshold', type=float, default=0.25)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--det_model', default = 'v5a', help = 'Version : v4, v5a, v5b')
    parser.add_argument('--clss_model', default='yolov8', help='Version : yolov7, yolov8')
    # parser.add_argument('--clss_class', type=int, default=10)
    parser.add_argument('--work_path', default = '/home/youngwoong/ppj/project')

    args = parser.parse_args()
    return args

# ###################################################################################################

def main(logger):
    logger.info('Wild detection start')
    args = parse_args()

    WORK_PATH = args.work_path
    sys.path.append(WORK_PATH)
    sys.path.append(osp.join(WORK_PATH, "detection"))

    import detection.visualization_utils as viz_utils
    from detection.pytorch_detector import PTDetector

    clss_threshold = args.clss_threshold
    confidence_threshold = args.threshold
    assert 0.0 < clss_threshold <= 1.0, f'Classifier confidence threshold needs to be between 0 and 1 :{clss_threshold}'
    assert 0.0 < confidence_threshold <= 1.0, f'Detection confidence threshold needs to be between 0 and 1 :{confidence_threshold}'

    detector_file = osp.join(WORK_PATH, 'model', 'md_v5a.0.0.pt')
    detector = PTDetector(detector_file, False)

    clss_model = args.clss_model
    if clss_model == 'yolov7':
        classify_model = torch.hub.load(osp.join(WORK_PATH, 'yolov7'), 'custom', osp.join(WORK_PATH, 'model', 'v7_animal_last.pt'), source='local', force_reload=False)
        cls_names = [ 'badger', 'bird', 'boar', 'cat', 'dog', 'goat', 'leopard_cat', 'marten', 'rabbit', 'raccoon', 'roe_deer', 'water_deer', 'weasel' ] # v7
    elif clss_model == 'yolov8':
        classify_model = YOLO(osp.join(WORK_PATH, 'model', 'G_v8x_merge_10_17_best_map89.pt'))
        cls_names = [ 'badger', 'bird', 'boar', 'cat', 'dog', 'leopard_cat', 'marten', 'rabbit', 'raccoon', 'roe_deer', 'water_deer', 'weasel' ] # G_v8

    input_dir = args.input_dir
    if not osp.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)

    mv_dir = args.mv_dir

    rst_dir = args.rst_dir
    if not osp.exists(rst_dir):
        os.makedirs(rst_dir, exist_ok=True)

    crop = False
    if args.crop_dir:
        crop = True
        crop_dir = args.crop_dir
        if not osp.exists(crop_dir):
            os.makedirs(crop_dir, exist_ok=True)

    logger.info(f'input_dir : {input_dir}, mv_dir : {mv_dir}, rst_dir : {rst_dir}, '
                f'det threshold : {confidence_threshold}, det model : {detector_file}, crop : {crop}, '
                f'clss threshold : {clss_threshold}, clss model : {clss_model}')

    print('START!')
    sys.stdout.flush()

    badger_count = 0
    bird_count = 0
    boar_count = 0
    cat_count = 0
    dog_count = 0
    leopard_cat_count = 0
    marten_count = 0
    rabbit_count = 0
    raccoon_count = 0
    roe_deer_count = 0
    water_deer_count = 0
    weasel_count = 0

    while True:
        iter_dir = (entry for entry in Path(input_dir).iterdir() if entry.is_file())

        for img_file in iter_dir:results
            img_file = str(img_file)

            if osp.getsize(img_file) > 0:
                try:
                    image = viz_utils.load_image(img_file)
                except Exception as e:
                    logger.error(f'Image {img_file} cannot be loaded. Exception: {e}')

                    mv_filename = viz_utils.move_save_img(img_file, mv_dir)
                    results = {
                        "file": img_file,
                        "annotated_filename": None,
                        "mv_filename": mv_filename,
                        "failure": "Failure image access"
                    }
                    print(json.dumps(results))
                    sys.stdout.flush()

                    continue

                try:
                    results = detector.generate_detections_one_image(image, img_file,
                                                                    detection_threshold=confidence_threshold,
                                                                    label_map=DEFAULT_DETECTOR_LABEL_MAP) # Megadetection model
                    if not results["prediction"]:
                        mv_filename = viz_utils.move_save_img(img_file, mv_dir)
                        results["annotated_filename"] = None
                        results["mv_filename"] = mv_filename
                        print(json.dumps(results))
                        sys.stdout.flush()

                        continue

                except Exception as e:
                    logger.error(f'An error occurred while running the detector on image {img_file}. Exception: {e}')

                    mv_filename = viz_utils.move_save_img(img_file, mv_dir)
                    results["annotated_filename"] = None
                    results["mv_filename"] = mv_filename
                    print(json.dumps(results))
                    sys.stdout.flush()

                    continue
                try:
                    images_cropped, categories = viz_utils.crop_image(results["prediction"], image,
                                                                      confidence_threshold=confidence_threshold)
                    obj_flag = False
                    for idx, cropped_image in enumerate(images_cropped):
                        if categories[idx] == "1":
                            if clss_model == 'yolov7':
                                result = classify_model(cropped_image) # by v7
                                clss_result = result.result() # by v7
                                if len(clss_result) != 0:                                
                                    clss_cls = clss_result[0]["cls"] # yolov7
                                    obj_flag = True
                            
                            elif clss_model == 'yolov8':
                                result = classify_model(cropped_image) # by any v8
                                if len(result) != 0:
                                    obj_flag = True
                                    clss_conf = 0
                                    clss_cls = 0
                                    for r in result:
                                        clss_conf = r.probs.top1conf.item() # G_v8 and G_v8_merge
                                        clss_cls = r.probs.top1 # G_v8 and G_v8_merge

                            if obj_flag:
                                if cls_names[int(clss_cls)] == "badger":
                                    badger_count += 1
                                if cls_names[int(clss_cls)] == "bird":
                                    bird_count += 1
                                if cls_names[int(clss_cls)] == "boar":
                                    boar_count += 1
                                if cls_names[int(clss_cls)] == "cat":
                                    cat_count += 1
                                if cls_names[int(clss_cls)] == "dog":
                                    dog_count += 1
                                if cls_names[int(clss_cls)] == "leopard_cat":
                                    leopard_cat_count += 1
                                if cls_names[int(clss_cls)] == "marten":
                                    marten_count += 1
                                if cls_names[int(clss_cls)] == "rabbit":
                                    rabbit_count += 1
                                if cls_names[int(clss_cls)] == "raccoon":
                                    raccoon_count += 1
                                if cls_names[int(clss_cls)] == "roe_deer":
                                    roe_deer_count += 1
                                if cls_names[int(clss_cls)] == "water_deer":
                                    water_deer_count += 1
                                if cls_names[int(clss_cls)] == "weasel":
                                    weasel_count += 1

                                if clss_model == 'yolov7':
                                    # result by yolo v7 
                                    if float(clss_result[0]["conf"]) >= clss_threshold:
                                        results["prediction"][idx]["best_class"] = str(int(clss_result[0]["cls"]))
                                        results["prediction"][idx]["best_probability"] = round(float(clss_result[0]["conf"]), 3)
                                        results["prediction"][idx][
                                            "name"] = f"{clss_result[0]['label']} {round(float(clss_result[0]['conf']) * 100)}%"
                                    else:
                                        # ETC
                                        results["prediction"][idx]["best_class"] = "15"
                                
                                elif clss_model == 'yolov8':
                                    #""" # result by yolo v8
                                    if float(clss_conf) >= clss_threshold:
                                        results["prediction"][idx]["best_class"] = str(int(clss_cls))
                                        results["prediction"][idx]["best_probability"] = round(float(clss_conf), 3)
                                        results["prediction"][idx][
                                            "name"] = f"{cls_names[int(clss_cls)]} {round(float(clss_conf) * 100)}%"
                                    else:
                                        # ETC
                                        results["prediction"][idx]["best_class"] = "15"
                                    #"""

                                if crop == True:
                                    clss_class_num = results["prediction"][idx]["best_class"]
                                    crop_file_name = f"{osp.splitext(osp.basename(img_file))[0]}_idx_{idx}_class_{clss_class_num}.jpg"
                                    cropped_image.save(osp.join(crop_dir, crop_file_name))
                            else:
                                # ETC
                                    results["prediction"][idx]["best_class"] = "15"

                        elif categories[idx] == "2":
                            # Person
                            results["prediction"][idx]["best_class"] = "13"

                        elif categories[idx] == "3":
                            # vehicle
                            results["prediction"][idx]["best_class"] = "14"
                                                
                except Exception as e:
                    logger.error(f'An error occurred while running the classifier on image {img_file}. Exception: {e}')
                    results["failure"] = "Failure classifier or Failure crop"

                try:
                    viz_utils.render_detection_bounding_boxes(results["prediction"], image,
                                                              confidence_threshold=confidence_threshold)

                    rst_file_name = osp.join(rst_dir, osp.basename(img_file))
                    image.save(rst_file_name)
                    results["annotated_filename"] = rst_file_name

                except Exception as e:
                    logger.error(f'Visualizing results on the image {img_file} failed. Exception: {e}')
                    results["failure"] = "Failure visualizing results"
                    results["annotated_filename"] = None

                mv_filename = viz_utils.move_save_img(img_file, mv_dir)
                results["mv_filename"] = mv_filename
                print(json.dumps(results))
                #print("badger", badger_count, "bird", bird_count, "boar", boar_count, "cat", cat_count, "dog", dog_count, "leopard_cat", leopard_cat_count, "marten", marten_count, "rabbit", rabbit_count, "raccoon", raccoon_count, "roe_deer", roe_deer_count, "water_deer", water_deer_count, "weasel", weasel_count)
                sys.stdout.flush()

            else:
                logger.error(f'Img File size 0 : {img_file}')
                time.sleep(0.1)

if __name__ == '__main__':
    try:
        logger = set_logger()
        main(logger)
    except Exception:
        logger.error(f'{traceback.format_exc()}')
    finally:
        logger.info('Wild detection exit')
