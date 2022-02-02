# Web API FastAPI

<div align="center" id="top">
  
  
  https://user-images.githubusercontent.com/24257959/152055296-aa1f2674-4275-450f-b441-d7f574cec3fb.mp4


</div>

## About ##
~~~
This project is a Web API for Urban Scene Segmentation for Autonomous Car, using two models `Baseline` and `MTKT`, that can handle unsupervised domain adaptation (UDA) problem for multi-target datasets.
~~~

## [Deeplab V2](https://arxiv.org/abs/1606.00915v2) ##
~~~
 It is built upon the ResNet-101 backbone initialized with ImageNet pre-trained weights, conducted with PyTorch. 
~~~
![image](https://user-images.githubusercontent.com/24257959/152067777-4917fac5-f5a3-48a2-97d1-820fda721f86.png)

## Baseline ##
~~~
Our first Model `Baseline` strategy is to merge all target datasets into a single one and then deal as a single target.
~~~
<div align="center" id="top" > 
  <img src="https://user-images.githubusercontent.com/24257959/152067945-b462b4fa-6599-4fae-94a5-841378ef7e8b.png" style="max-width: 50%;">
</div>

## MTKT ##
~~~
our second model `MTKT` is a multi-target knowledge transfer, which has a target-specific teacher for each specific target that learns a target-agnostic model thanks to a multi-teacher/single-student distillation mechanism.
~~~
<div align="center" id="top" style="max-width: 40%;"> 
  
  ![image](https://user-images.githubusercontent.com/24257959/152068123-d6e60e89-c9ce-483a-95a8-0387371b6ba8.png)
</div>

The UI for this api is found in [Web-Application-Client-Flask](https://github.com/mohamedelmesawy/Web-Application-Client-Flask) repo.

## Features ##
:heavy_check_mark: Fine Segmentation for unlabeled or coarse segmentated image. \
:heavy_check_mark: Segmentation with calculation of mIOU for labeled image.

## Technologies ##
The following tools were used in the REST-API project:
- [Pthon 3.7](https://www.python.org/)
- [fastapi](https://fastapi.tiangolo.com/)
- [Pytorch >= 0.4.1](https://pytorch.org/)
- [CUDA 9.0 or higher](https://developer.nvidia.com/)

## Requirements ##
Before starting, you need to have [Git](https://git-scm.com), [Python 3.7](https://www.python.org/), [torch](https://pytorch.org/) and [cuda](https://developer.nvidia.com/) installed.

Also pre-trained models can be downloaded here(https://github.com/valeoai/MTAF/releases) and put in `<root_dir>/model`

```bash

# Clone this project
$ git clone https://github.com/mohamedelmesawy/Web-API-FastAPI

# install FastAPI :
$ pip install fastapi

# Also install uvicorn to work as the server:
$ pip install "uvicorn[standard]"

# Run the Application main.py
$ uvicorn main:app --reload
```



By RAM-Team: <a href="https://github.com/mohamedelmesawy" target="_blank">MOhamed ElMesawy</a>, <a href="https://github.com/Afnaaan" target="_blank">Afnan Hamdy</a>, <a href="https://github.com/Rawan-97" target="_blank">Rowan ElMahalawy</a>, <a href="https://github.com/alihasanine" target="_blank">Ali Hassanin</a>, and <a href="https://github.com/MSamaha91" target="_blank">MOhamed Samaha</a>
&#xa0;


## Acknowledgements ##
The pretrained models are borrowed from [MTAF](https://github.com/mohamedelmesawy/MTAF).
