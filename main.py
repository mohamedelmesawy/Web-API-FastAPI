import imp
from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
import model.pretrained_model as pretrained_model
from model.pretrained_model import vis_segmentation
from model import preprocessing_labels
from model.metrics import compute_IOU
from starlette.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import torch.nn as nn
import numpy as np
import base64
import cv2
import io
import os


model_basline = pretrained_model.ValidationModel(model_type='baseline') 
model_mtkt    = pretrained_model.ValidationModel(model_type='mtkt')
mapillary_labels, cityscapes_labels = preprocessing_labels.get_labels_maps()
labels_maps = {
    'mapillary'  : mapillary_labels,
    'cityscapes' : cityscapes_labels,
} 

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return({"message": "It works!"})


@app.post("/upload/{model_type}")
async def upload_file(model_type, file: UploadFile = File(...), isByte=False):
    image_path = os.path.join('images', file.filename)

    with open(image_path, 'wb') as image:
        content = await file.read()
        image.write(content)
        image.close()

    if model_type == 'baseline':
        pred_main = model_basline.predict(image_path)
    elif model_type == 'mtkt':
        pred_main = model_mtkt.predict(image_path)
    
    vis_segmentation(image_path, pred_main, file.filename)
  
    pred_path = os.path.join('predictions', file.filename)
    # pred_path = os.path.join('predictions', "hamburg_000000_106102_leftImg8bit.png")
    im_png = cv2.imread(pred_path)
    res, im_png = cv2.imencode(".png", im_png)
    
    if isByte:
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
    return  base64.b64encode(im_png.tobytes())    
##########################

@app.post("/uploadfiles/{model_type}")
async def upload_files(model_type, file: List[UploadFile] = File(...), isByte=False):
    img = file[0]
    image_path = os.path.join('images', img.filename)
    ground_truth_img = file[1]
    ground_truth_path = os.path.join('images', ground_truth_img.filename)

    with open(image_path, 'wb') as image:
        content = await img.read()
        image.write(content)
        image.close()
        
    with open(ground_truth_path, 'wb') as image:
        content = await ground_truth_img.read()
        image.write(content)
        image.close()
    
    
    label = preprocessing_labels.prepare_label(ground_truth_path, labels_maps)

    if model_type == 'baseline':
        pred_main = model_basline.predict(image_path)
    elif model_type == 'mtkt':
        pred_main = model_mtkt.predict(image_path)
    
    vis_segmentation(image_path, pred_main, file[0].filename)

    

    ########## CHECK ME #############
    # org_img  = cv2.imread(image_path)
    # pred_main = cv2.resize(pred_main.astype(
    #                             'uint8'), dsize=(org_img.shape[1], org_img.shape[0]))

    print("---------------> Image size ", pred_main.shape, " <----------------------")
    print("---------------> label size ", label.shape, " <----------------------")
    print(np.unique(label.flatten()))

    # def fast_hist(a, b, n=7):
    #     k = (a >= 0) & (a < n)
    #     return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


    # def per_class_iu(hist):
    #     return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    # hist = fast_hist(label.flatten(), pred_main.flatten())
    # IoU_list = per_class_iu(hist)
    # mIoU = np.nanmean(IoU_list)
    # print("hist IOU -----------> ", single_score)
    # print("hist MIOU ----------> ", )
    
    mIoU, IoU_list = compute_IOU(pred_main, label)
    print("MIoU :", mIoU)
    print("IoU list:")
    print(IoU_list)
        
    pred_path = os.path.join('predictions', file[0].filename)
    # pred_path = os.path.join('predictions', "hamburg_000000_106102_leftImg8bit.png")
    im_png = cv2.imread(pred_path)

    # # Add v_Padding
    # h_padding_length = pred_main.shape[1]//7
    # h_padding = im_png.copy()[:, :h_padding_length]
    # h_padding[:, :] = 0

    # Add h_Padding
    h_padding_length = im_png.shape[1]//6
    h_padding = im_png.copy()[:, :h_padding_length]
    h_padding[:, :] = 255

    text_h_start = im_png.shape[1] + 20
    im_png = np.hstack((im_png, h_padding))

    font = cv2.FONT_HERSHEY_TRIPLEX
    color = (0, 0, 0)
    scale = 1.25
    stroke = 3
    im_png = cv2.putText(im_png, "mean IoU = {:.2f}%".format(round((mIoU * 100), 2)), (text_h_start, 60), font, (scale + 0.20), color, stroke)
    classes = preprocessing_labels.get_classes()
    color_map = list(preprocessing_labels.get_colormap())

    # convert colormap from RGB to BGR
    for i in range(len(color_map)):
        color_map[i] = color_map[i][::-1].tolist() 

    iou_list_start = 175
    for i, class_name in enumerate(classes):
        offset = i * 110
        print(color_map[i])
        color = tuple(color_map[i])
        im_png = cv2.putText(im_png, "{}= {:.2f}%".format(class_name, IoU_list[i] * 100), (text_h_start, (iou_list_start + offset)), font, scale, color, stroke)

    pred_path = os.path.join('predictions', img.filename)
    print(pred_path)
    cv2.imwrite(pred_path, im_png)

    res, im_png = cv2.imencode(".png", im_png)
    
    if isByte:
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
    return  base64.b64encode(im_png.tobytes())   

@app.get("/test2")
async def main():
    content = """
<body>
<form action="/upload/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)
##########################

# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...), isByte=False):
#     image_path = os.path.join('images', file.filename)

#     with open(image_path, 'wb') as image:
#         content = await file.read()
#         image.write(content)
#         image.close()

#     pred_main = model_basline.predict(image_path)
#     model_basline.vis_segmentation(image_path, pred_main, file.filename)

#     pred_path = os.path.join('predictions', file.filename)
#     im_png = cv2.imread(pred_path)
#     res, im_png = cv2.imencode(".png", im_png)
    
#     if isByte:
#         return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
#     return  base64.b64encode(im_png.tobytes())    





# @app.get("/")
# async def root():
#     content = """
#         <body>
#             <form action="/upload/" enctype="multipart/form-data" method="post">
#                 <input name="files" type="file">
#                 <input type="submit">
#             </form>
#         </body>
#     """
#     return HTMLResponse(content=content)
