import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import os
affine_par = True

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        # change
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        # change
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ClassifierModule(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes, multi_level):
        self.multi_level = multi_level
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=1, dilation=4)
        if self.multi_level:
            self.layer5 = ClassifierModule(1024, [6, 12, 18, 24], [
                                           6, 12, 18, 24], num_classes)
        self.layer6 = ClassifierModule(2048, [6, 12, 18, 24], [
                                       6, 12, 18, 24], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (stride != 1
                or self.inplanes != planes * block.expansion
                or dilation == 2
                or dilation == 4):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.multi_level:
            x1 = self.layer5(x)  # produce segmap 1
        else:
            x1 = None
        x2 = self.layer4(x)
        x2 = self.layer6(x2)  # produce segmap 2
        return x1, x2

    def get_1x_lr_params_no_scale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        if self.multi_level:
            b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_no_scale(), 'lr': lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * lr}]


def get_deeplab_v2(num_classes=7, multi_level=True):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes, multi_level)
    return model


def preprocess_cityscapes_vid(img_path):
    img = Image.open(img_path)
    mean = np.array((104.00698793, 116.66876762,
                     122.67891434), dtype=np.float32)
    img = img.convert('RGB')
    img = img.resize((640, 320), Image.BICUBIC)  # if
    image = np.asarray(img, np.float32)
    image = image[:, :, ::-1]  # change to BGR
    image -= mean
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(np.flip(image, axis=0).copy()).unsqueeze(0)
    return image

# move the function to main


def load_checkpoint_for_evaluation():
    checkpoint = os.path.join('model', 'gta2cityscapes_mapillary_baseline.pth')
    model = get_deeplab_v2()

    saved_state_dict = torch.load(
        checkpoint, map_location=DEVICE)

    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(DEVICE)

    return model


def predict(img):
    img = preprocess_cityscapes_vid(img)
    model = load_checkpoint_for_evaluation()
    _, pred_main = model(img.cuda(DEVICE))
    # pred_main = pred_main[0]

    interp = nn.Upsample(size=(640, 320), mode='bilinear', align_corners=True)
    output = interp(pred_main).cpu().data[0].numpy()

    output = output.transpose(1, 2, 0)
    seg_map = np.argmax(output, axis=2)

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
    prediction = cv2.resize(prediction.astype(
        'uint8'), dsize=(image.shape[1], image.shape[0]))
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

    # pred_path = os.path.join('predictions', file_name)
    # cv2.imwrite(pred_path, all_images)
