from fastapi import FastAPI
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
import model.baslinemodel as baslinemodel
from starlette.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import io


import os

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


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    image_path = os.path.join('images', file.filename)

    with open(image_path, 'wb') as image:
        content = await file.read()
        image.write(content)
        image.close()

    pred_main = baslinemodel.predict(image_path)
    baslinemodel.vis_segmentation(image_path, pred_main, file.filename)

    # pred_path = 'filename2.png'
    pred_path = os.path.join('predictions', file.filename)
    im_png = cv2.imread(pred_path)
    res, im_png = cv2.imencode(".png", im_png)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
    # return JSONResponse(content={"filename": file.filename}, status_code=200)


# @app.post("/upload")
# def create_file(file: UploadFile = File(...)):
#     with open(os.path.join('./images', file.filename), 'wb+') as upload_folder:
#         shutil.copyfileobj(file.file, upload_folder)
#         upload_folder.close()
#     return {"filename": file.filename}
# shutil.move(file.filename, './images/'+file.filename)


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
