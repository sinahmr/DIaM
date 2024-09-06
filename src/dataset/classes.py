import argparse
from collections import defaultdict
from typing import Dict, Any
import os
import json

classId2className = {'coco': {
                         1: 'person',
                         2: 'bicycle',
                         3: 'car',
                         4: 'motorcycle',
                         5: 'airplane',
                         6: 'bus',
                         7: 'train',
                         8: 'truck',
                         9: 'boat',
                         10: 'traffic light',
                         11: 'fire hydrant',
                         12: 'stop sign',
                         13: 'parking meter',
                         14: 'bench',
                         15: 'bird',
                         16: 'cat',
                         17: 'dog',
                         18: 'horse',
                         19: 'sheep',
                         20: 'cow',
                         21: 'elephant',
                         22: 'bear',
                         23: 'zebra',
                         24: 'giraffe',
                         25: 'backpack',
                         26: 'umbrella',
                         27: 'handbag',
                         28: 'tie',
                         29: 'suitcase',
                         30: 'frisbee',
                         31: 'skis',
                         32: 'snowboard',
                         33: 'sports ball',
                         34: 'kite',
                         35: 'baseball bat',
                         36: 'baseball glove',
                         37: 'skateboard',
                         38: 'surfboard',
                         39: 'tennis racket',
                         40: 'bottle',
                         41: 'wine glass',
                         42: 'cup',
                         43: 'fork',
                         44: 'knife',
                         45: 'spoon',
                         46: 'bowl',
                         47: 'banana',
                         48: 'apple',
                         49: 'sandwich',
                         50: 'orange',
                         51: 'broccoli',
                         52: 'carrot',
                         53: 'hot dog',
                         54: 'pizza',
                         55: 'donut',
                         56: 'cake',
                         57: 'chair',
                         58: 'sofa',
                         59: 'pottedplant',
                         60: 'bed',
                         61: 'diningtable',
                         62: 'toilet',
                         63: 'tv',
                         64: 'laptop',
                         65: 'mouse',
                         66: 'remote',
                         67: 'keyboard',
                         68: 'cell phone',
                         69: 'microwave',
                         70: 'oven',
                         71: 'toaster',
                         72: 'sink',
                         73: 'refrigerator',
                         74: 'book',
                         75: 'clock',
                         76: 'vase',
                         77: 'scissors',
                         78: 'teddy bear',
                         79: 'hair drier',
                         80: 'toothbrush'
                         },

                     'pascal': {
                        1: 'airplane',
                        2: 'bicycle',
                        3: 'bird',
                        4: 'boat',
                        5: 'bottle',
                        6: 'bus',
                        7: 'car',
                        8: 'cat',
                        9: 'chair',
                        10: 'cow',
                        11: 'diningtable',
                        12: 'dog',
                        13: 'horse',
                        14: 'motorcycle',
                        15: 'person',
                        16: 'pottedplant',
                        17: 'sheep',
                        18: 'sofa',
                        19: 'train',
                        20: 'tv'
                        }
                     }

className2classId = defaultdict(dict)
for dataset in classId2className:
    for id in classId2className[dataset]:
        className2classId[dataset][classId2className[dataset][id]] = id


def get_split_classes(cfg: dict, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Returns the split of classes for Pascal-5i, Pascal-10i and Coco-20i
    inputs:
        args

    returns :
         split_classes : Dict.
                         split_classes['coco'][0]['train'] = training classes in fold 0 of Coco-20i
    """

    def save_splits_to_files(split_classes):
        for dataset_name, splits in split_classes.items():
            dataset_dir = f"{dataset_name}_splits"
            os.makedirs(dataset_dir, exist_ok=True)
            for split_key, split in splits.items():
                file_path = os.path.join(dataset_dir, f"split_{split_key}.json")
                with open(file_path, 'w') as f:
                    json.dump(split, f)
                print(f"Saved split {split_key} for {dataset_name} to {file_path}")

    def load_splits_from_files():
        split_classes = {'coco': defaultdict(dict), 'pascal': defaultdict(dict)}
        for dataset_name in split_classes.keys():
            dataset_dir = f"{dataset_name}_splits"
            if not os.path.exists(dataset_dir):
                continue
            for split_file in os.listdir(dataset_dir):
                split_key = int(split_file.split('_')[1].split('.')[0])
                file_path = os.path.join(dataset_dir, split_file)
                with open(file_path, 'r') as f:
                    try:
                        split_classes[dataset_name][split_key] = json.load(f)
                        print(f"Loaded split {split_key} for {dataset_name} from {file_path}")
                    except json.JSONDecodeError as e:
                        print(f"Error loading {file_path}: {e}")
                        continue
        return split_classes

    # Check if splits are already saved
    if os.path.exists("coco_splits") or os.path.exists("pascal_splits"):
        split_classes = load_splits_from_files()
    else:
        split_classes = {'coco': defaultdict(dict), 'pascal': defaultdict(dict)}


    split_classes = {'coco': defaultdict(dict), 'pascal': defaultdict(dict)}

    # =============== COCO ===================
    name = 'coco'
    class_list = list(range(1, 81))
    # class_list = list(range(1, 8))
    split_classes[name][-1]['val'] = class_list
    if cfg['DATA']['use_split_coco']:
        vals_lists = [list(range(1, 78, 4)), list(range(2, 79, 4)),list(range(3, 80, 4)), list(range(4, 81, 4))]
        # vals_lists = [list(range(1, 7, 4)), list(range(2, 8, 4)),list(range(3, 8, 4)), list(range(4, 8, 4))]
        for i, val_list in enumerate(vals_lists):
            split_classes[name][i]['val'] = val_list
            split_classes[name][i]['train'] = sorted(list(set(class_list) - set(val_list)))

    else:
        class_list = list(range(1, 81))
        vals_lists = [list(range(1, 21)), list(range(21, 41)),
                      list(range(41, 61)), list(range(61, 81))]
        for i, val_list in enumerate(vals_lists):
            split_classes[name][i]['val'] = val_list
            split_classes[name][i]['train'] = sorted(list(set(class_list) - set(val_list)))
    
    print(f'split classes is here.. {split_classes}')

    # =============== Pascal ===================
    name = 'pascal'
    class_list = list(range(1, 21))
    vals_lists = [
        (0, list(range(1, 6))), (1, list(range(6, 11))),
        (2, list(range(11, 16))), (3, list(range(16, 21))),
        (10, list(range(1, 11))), (11, list(range(11, 21))),
    ]
    split_classes[name][-1]['val'] = class_list
    for i, val_list in vals_lists:
        split_classes[name][i]['val'] = val_list
        split_classes[name][i]['train'] = sorted(list(set(class_list) - set(val_list)))
    

    save_splits_to_files(split_classes)
    # print("CLASSES FROM....", split_classes)
    # Now, let's create folders for each split
    

    return split_classes