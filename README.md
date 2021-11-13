# FGI-Matting
The official repository for Deep Image Matting with Flexible Guidance Input.

Paper:  https://arxiv.org/abs/2110.10898

![image](https://github.com/Charch-630/FGI-Matting/blob/main/gif/demo_gif.gif?raw=true) 

![all](https://user-images.githubusercontent.com/52871875/141040506-02393e89-1f46-4578-a0ad-545d23158c4f.png)

## Requirements
  - easydict
  - numpy
  - opencv-python
  - Pillow
  - PyQt5
  - scikit-image
  - scipy
  - toml
  - torch>=1.5.0
  - torchvision
 
## Models and supplementary data for DIM test set(Composition-1k) and Distinctions-646 test set
Google drive: https://drive.google.com/drive/folders/13qnlXUSKS5HfkfvzdMKAv7FvJ6YV_wPK?usp=sharing  
百度网盘： https://pan.baidu.com/s/1ZYcbwyCIrL6G9t7pkCIBYw 提取码: zjtj 


- `Weight_DIM.pth`  The model trained with Adobe matting dataset.

- `Weight_D646.pth`  The model trained with Distincions-646 dataset.

- `DIM_test_supp_data.zip`  Scribblemaps and Clickmaps for DIM test set.

- `D-646_test_supp_data.zip`  Scribblemaps and Clickmaps for Distinctions-646 test set.

Place `Weight_DIM.pth` and `Weight_D646.pth` in `./checkpoints`.  
Edit `./config/FGI_config` to modify the path of the testset and choose the checkpoint name.


## Test on DIM test set(Composition-1k)

| Methods  | SAD | MSE | Grad | Conn |
| :----------: | :-----------:| :-----------:| :-----------:| :-----------:|
| Trimap test   | 30.19   | 0.0061 | 13.07 | 26.66 |
| Scribblemap test   | 32.86   | 0.0090 | 14.18 | 29.09 |
| Clickmap test   | 34.67 | 0.0112 | 15.45 | 30.96 |
| No guidance test   | 36.36   | 0.0141 | 15.23 | 32.76 |

`"checkpoint"` in `./config/FGI_config` should be "Weight_DIM".  
`bash test.sh`  
Modify `"guidancemap_phase"` in `./config/FGI_config` to test on trimap, scribblemap, clickmap and No_guidance.  
For further test, please use the code in `./DIM_evaluation_code` and the predicted alpha mattes in `./alpha_pred`.

## Test on Distinctions-646 test set(Not appear in the paper)

| Methods  | SAD | MSE | Grad | Conn |
| :----------: | :-----------:| :-----------:| :-----------:| :-----------:|
| Trimap test   | 28.90 | 0.0105 | 24.67 | 27.40 |
| Scribblemap test   | 33.22 | 0.0131 | 26.93 | 31.38 |
| Clickmap test   | 34.97 | 0.0146 | 27.60 | 33.11 |
| No guidance test   | 36.83 | 0.0156 | 28.28 | 34.90 |

`"checkpoint"` in `./config/FGI_config` should be "Weight_D646".  
`bash test.sh`  
Modify `"guidancemap_phase"` in `./config/FGI_config` to test on trimap, scribblemap, clickmap and No_guidance.  
For further test, please use the code in `./DIM_evaluation_code` and the predicted alpha mattes in `./alpha_pred`.

## The QT Demo

Copy one of the pth file and rename it `"Weight_qt_in_use.pth"`, also place it in `./checkpoints`.  
Run `test_one_img_qt.py`. 
Try images in `./testimg`. It will use GPU if avaliable, otherwise it will use CPU.

![demo](https://user-images.githubusercontent.com/52871875/141238176-2020b881-0177-4d2d-b8f3-d823442aed7e.png)

I recommend to use the one trained on DIM dataset.  
Have fun :D

## Acknowledgment
GCA-Matting: https://github.com/Yaoyi-Li/GCA-Matting
