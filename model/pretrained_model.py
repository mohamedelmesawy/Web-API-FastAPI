import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import os
from model.deeplapv2_model import get_deeplab_v2, get_deeplab_v2_mtkt
affine_par = True

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ValidationModel():
    
    def __init__(self, model_type='baseline'):
        self.model_type = model_type


    def preprocess_cityscapes_vid(self, img_path):
        img = Image.open(img_path)
        img_orginal_size = np.array(img).shape
        mean = np.array((104.00698793, 116.66876762,
                        122.67891434), dtype=np.float32)
        img = img.convert('RGB')
        img = img.resize((640, 320), Image.BICUBIC)  # if
        image = np.asarray(img, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= mean
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(np.flip(image, axis=0).copy()).unsqueeze(0)
        return image,img_orginal_size


    def load_checkpoint_for_evaluation(self):
        if self.model_type=='baseline':
            checkpoint = os.path.join('model','gta2cityscapes_mapillary_baseline.pth')
            model = get_deeplab_v2()
        elif self.model_type=='mtkt':
            checkpoint = os.path.join('model','gta2cityscapes_mapillary_mtkt.pth')
            model = get_deeplab_v2_mtkt()
            
        saved_state_dict = torch.load(
            checkpoint, map_location=DEVICE)

        model.load_state_dict(saved_state_dict)
        model.eval()
        model.cuda(DEVICE)

        return model


    def predict(self, img):
        
        img, img_org_size = self.preprocess_cityscapes_vid(img)
        model = self.load_checkpoint_for_evaluation() 
        
        if self.model_type=='baseline':
            _, pred_main = model(img.cuda(DEVICE))
        elif self.model_type=='mtkt':
            _, pred_main_list= model(img.cuda(DEVICE))
            pred_main = pred_main_list[0]
        
        interp = nn.Upsample(size=(img_org_size[0], img_org_size[1]), mode='bilinear', align_corners=True)
        output = interp(pred_main).cpu().data[0].numpy()

        output = output.transpose(1, 2, 0)
        seg_map = np.argmax(output, axis=2)

        seg_map = cv2.resize(seg_map.astype('uint8'), dsize=(img_org_size[1], img_org_size[0]))
        

        return seg_map


def create_label_colormap(no_class=7, dataset=None):
    """Creates a label colormap used in Cityscapes segmentation benchmark.
    Returns:
        A Colormap for visualizing segmentation results.
    """
    # GTA 19 Classes
    if (dataset == 'GTA') or (no_class == 19):
        colormap = np.array([
            # COLOR           # Index in the Original Cityscape
            [128,  64,  128],   # 7,    # road
            [244,  35,  232],   # 8,    # sidewalk
            [70,   70,   70],   # 11,   # building
            [102,  102, 156],   # 12,   # wall
            [190,  153, 153],   # 13,   # fence
            [153,  153, 153],   # 17,   # pole
            [250,  170,  30],   # 19,   # traffic light
            [220,  220,   0],   # 20,   # traffic sign
            [107,  142,  35],   # 21,   # vegetation
            [152,  251, 152],   # 22,   # terrain
            [70,   130, 180],   # 23,   # sky - RAM Modified
            [220,   20,  60],   # 24,   # person
            [255,    0,   0],   # 25,   # rider
            [0,      0, 142],   # 26,   # car
            [0,      0,  70],   # 27,   # truck
            [0,     60, 100],   # 28,   # bus
            [0,     80, 100],   # 31,   # train
            [0,      0, 230],   # 32,   # motorcycle
            [119,   11,  32],   # 33,   # bicycle
            # 0,    # void [Background] is the last 20th (counting from 1) class.
            [0,      0,   0],
        ], dtype=np.uint8)
    elif (dataset == 'CITYSCAPES') or (no_class == 35):
        # Cityscape   35 Classes
        colormap = np.array([
            #  Color 35 Class    # Id/Index in Cityscape
            [0,   0,   0],    # 0
            [0,   0,   0],    # 1
            [0,   0,   0],    # 2
            [0,   0,   0],    # 3
            [0,   0,   0],    # 4
            [111,  74,   0],    # 5
            [81,   0,  81],    # 6
            [128,  64, 128],    # 7
            [244,  35, 232],    # 8
            [250, 170, 160],    # 9
            [230, 150, 140],    # 10
            [70,  70,  70],    # 11
            [102, 102, 156],    # 12
            [190, 153, 153],    # 13
            [180, 165, 180],    # 14
            [150, 100, 100],    # 15
            [150, 120,  90],    # 16
            [153, 153, 153],    # 17
            [153, 153, 153],    # 18
            [250, 170,  30],    # 19
            [220, 220,   0],    # 20
            [107, 142,  35],    # 21
            [152, 251, 152],    # 22
            [70, 130, 180],    # 23
            [220,  20,  60],    # 24
            [255,   0,   0],    # 25
            [0,   0, 142],    # 26
            [0,   0,  70],    # 27
            [0,  60, 100],    # 28
            [0,   0,  90],    # 29
            [0,   0, 110],    # 30
            [0,  80, 100],    # 31
            [0,   0, 230],    # 32
            [119,  11,  32],    # 33
            [0,   0, 142],    # 34
        ], dtype=np.uint8)
    elif no_class == 7:
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


def label_to_color_image(label):
    label = np.where(label == 255, 7, label)
    colormap = create_label_colormap()  # By default no_class = 7
    return colormap[label]



def vis_segmentation(image_path, prediction, file_name=''):
    # image
    image = cv2.imread(image_path)
    all_images = image

    # Prediction
    prediction = label_to_color_image(prediction)
    prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
    # prediction = cv2.resize(prediction.astype(
    #     'uint8'), dsize=(image.shape[1], image.shape[0]))
    all_images = np.hstack((all_images, prediction))

    background = image.copy()
    overlay = prediction.copy()
    added_image = cv2.addWeighted(background, 0.4, overlay, 0.75, 0)
    all_images = np.hstack((all_images, added_image))

    padding_length = all_images.shape[0]//8
    padding = all_images.copy()[:padding_length, :]
    padding[:, :] = 0

    # Rescaling *****************
    ratio = image.shape[1]/image.shape[0]
    h = 800
    w = int(ratio * h)
    all_images = cv2.resize(all_images.astype('uint8'), dsize=(w * 3, h))

    # Add Padding
    padding_length = all_images.shape[0]//8
    padding = all_images.copy()[:padding_length, :]
    padding[:, :] = 0

    all_images = np.vstack((padding, all_images))

    x_padding = w // 3
    x = w
    y = int(padding.shape[0] / 1.5)
    font = cv2.FONT_HERSHEY_TRIPLEX
    color = (255, 255, 255)
    scale = 2
    stroke = 3

    cv2.putText(all_images, "Image", (x_padding, y),
                font, scale, color, scale, stroke)
    cv2.putText(all_images, "Prediction",
                (x + x_padding, y), font, scale, color, scale, stroke)
    cv2.putText(all_images, "Overlay", (2*x + x_padding, y),
                font, scale, color, scale, stroke)

    pred_path = os.path.join('predictions', file_name)
    # print(pred_path)
    cv2.imwrite(pred_path, all_images)
  