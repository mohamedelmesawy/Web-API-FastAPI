from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
# import model.pretrained_model as pretrained_model
from starlette.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import base64
import cv2
import io
import os


# model_basline = pretrained_model.ValidationModel(model_type='baseline') 
# model_mtkt    = pretrained_model.ValidationModel(model_type='mtkt') 

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

    # if model_type == 'baseline':
    #     pred_main = model_basline.predict(image_path)
    #     model_basline.vis_segmentation(image_path, pred_main, file.filename)
    # elif model_type == 'mtkt':
    #     pred_main = model_mtkt.predict(image_path)
    #     model_mtkt.vis_segmentation(image_path, pred_main, file.filename)
        
    # pred_path = os.path.join('predictions', file.filename)
    pred_path = os.path.join('predictions', "hamburg_000000_106102_leftImg8bit.png")
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

    # if model_type == 'baseline':
    #     pred_main = model_basline.predict(image_path)
    #     model_basline.vis_segmentation(image_path, pred_main, file.filename)
    # elif model_type == 'mtkt':
    #     pred_main = model_mtkt.predict(image_path)
    #     model_mtkt.vis_segmentation(image_path, pred_main, file.filename)
        
    # pred_path = os.path.join('predictions', file.filename)
    pred_path = os.path.join('predictions', "hamburg_000000_106102_leftImg8bit.png")
    im_png = cv2.imread(pred_path)
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
