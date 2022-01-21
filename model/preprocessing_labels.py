import imp
import json
import numpy as np
from PIL import Image
import os

def get_labels_maps():
    cwd = os.getcwd()
    with open(f'{cwd}\\model\\mapillary_config.json', 'r') as json_file:
        mapillary_labels_info = json.loads(json_file.read())['labels']

    with open(f'{cwd}\\model\\mapping_classes.json', 'r') as json_file:
        json_file_data = json_file.read()
        mapping_mapillary_classes = json.loads(json_file_data)['mapillary_classes_to_mainClassses']
        classes_ids = json.loads(json_file_data)['classes_ids']
        mapping_cityscapes_mainClasses_ids = json.loads(json_file_data)['cityscapes_classes_to_mainClasses_ids']

    # mapping mapillary classes to main classes ids
    mapillary_labels = [label_info['readable'].lower() for label_info in mapillary_labels_info]
    mapillary_classes = {class_name: class_id  for class_id, class_name in enumerate(mapillary_labels)}
    mapping_mapillary_classes = {mapillary_class.lower(): main_class.lower() for mapillary_class, main_class in mapping_mapillary_classes.items()}
    mapillary_to_mainClasses = np.zeros((max(mapillary_classes.values()) + 1,), dtype=np.uint8)
    for mapillary_class_name, idx in mapillary_classes.items():
        mapillary_to_mainClasses[idx] = classes_ids[mapping_mapillary_classes[mapillary_class_name]]

    # mapping cityscapes classes to main classes ids
    cityscapes_to_mainClasses = np.array([main_class_id for cityscapes_class_id, main_class_id in mapping_cityscapes_mainClasses_ids], dtype=np.uint8)

    return mapillary_to_mainClasses, cityscapes_to_mainClasses

def prepare_label(label_path, labels_maps):
    label = Image.open(label_path)
    # label = label.resize((640, 320))
    label_arr = np.array(label)
    if max(np.unique(label_arr)) <= 35:
        dataset_label_map = labels_maps['cityscapes']
    else:
        dataset_label_map = labels_maps['mapillary']
    return dataset_label_map[label_arr]

def get_classes():
    cwd = os.getcwd()
    with open(f'{cwd}\\model\\mapping_classes.json', 'r') as json_file:
        json_file_data = json_file.read()
        classes = json.loads(json_file_data)['labels']
    return classes

def get_colormap():
    colormap = np.array([
                [128,  64, 128],       # 0
                [70,  70,  70],       # 1
                [153, 153, 153],       # 2
                [107, 142,  35],       # 3
                [70, 130, 180],       # 4
                [220,  20,  60],       # 5
                [0,   0, 142],       # 6
                [0,   0,   0],       # 7
            ], dtype=np.uint8)
    return colormap