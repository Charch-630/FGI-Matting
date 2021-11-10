import torch
from   torch.utils.data import DataLoader
import cv2
import numpy as np

from   .image_file import ImageFileTrain, ImageFileTest
from   .data_generator import DataGenerator
from   .prefetcher import Prefetcher

from utils import CONFIG





def get_Test_dataloader():

    test_merged = CONFIG.test.test_merged
    test_alpha = CONFIG.test.test_alpha
    test_trimap =CONFIG.test.test_trimap

    test_clickmap = CONFIG.test.test_clickmap
    test_scribblemap = CONFIG.test.test_scribblemap

    test_guidancemap = None
    if CONFIG.test.guidancemap_phase == "trimap":
        test_guidancemap = test_trimap
    elif CONFIG.test.guidancemap_phase == "clickmap":
        test_guidancemap = test_clickmap
    elif CONFIG.test.guidancemap_phase == "scribblemap":
        test_guidancemap = test_scribblemap
    elif CONFIG.test.guidancemap_phase == "No_guidance":
        test_guidancemap = None
    else:
        NotImplementedError("Unknown guidancemap type")

    print(test_merged)
    print(test_alpha)
    print(test_trimap)

    test_image_file = ImageFileTest(alpha_dir=test_alpha,
                                        merged_dir=test_merged,
                                        trimap_dir=test_trimap,
                                        guidancemap_dir = test_guidancemap)
    test_dataset = DataGenerator(test_image_file, phase='test', test_scale= "origin")
    test_dataloader = DataLoader(test_dataset,
                                    batch_size= 1,
                                    shuffle=False,
                                    num_workers=4,
                                    drop_last=False)

    return test_dataloader
 


if __name__=="__main__":
    train_loader, testloader = get_DIM_click_gradual_change_dataloader(2,4)
    
    for i, data in enumerate(train_loader):
        image = data['image'][0]
        clickmap = data['clickmap'][0]
        alpha = data['alpha'][0]

        # cv2.imshow('image', image.numpy().transpose(1,2,0))
        # cv2.imshow('trimap', trimap.numpy())
        # cv2.imshow('alpha', alpha.numpy())

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        print('image', image.size())
        print('clickmap', clickmap.size())
        print('alpha', alpha.size())

