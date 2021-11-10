import cv2
import numpy as np
import torch
from   torch.nn import functional as F

import networks
import utils
import os
from time import time

class Tester_one_image(object):

    def __init__(self, test_config):

        self.model_config = {'encoder': "res_shortcut_encoder_29_spatial_attn", 'decoder': "res_shortcut_decoder_22_spatial_attn", 'trimap_channel':1}
        self.test_config = test_config

        self.build_model()
        self.resume_step = None

        if self.test_config['checkpoint']:
            self.restore_model(self.test_config['checkpoint'])

    def build_model(self):
        self.G = networks.get_generator(encoder=self.model_config['encoder'], decoder=self.model_config['decoder'])
        if torch.cuda.is_available():
            self.G.cuda()

    def restore_model(self, resume_checkpoint):
        """
        Restore the trained generator and discriminator.
        :param resume_checkpoint: File name of checkpoint
        :return:
        """
        pth_path = os.path.join(self.test_config['checkpoint_path'], '{}.pth'.format(resume_checkpoint))
        checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
        self.G.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

    def test(self, img, trimap):
        self.G = self.G.eval()
        with torch.no_grad():
            alpha_shape = img.shape[1:3]
            img = img.unsqueeze(0)
            trimap = trimap.unsqueeze(0)

            if torch.cuda.is_available():
                img = img.cuda()
                trimap = trimap.cuda()
                print("Using GPU")
            else:
                print("Using CPU")

            start = time()
            alpha_pred, _ = self.G(img, trimap)
            end = time()
            inference_time = end - start
            print('inference_time:', inference_time)
            if self.model_config['trimap_channel'] == 3:
                trimap = trimap.argmax(dim=1, keepdim=True)

            alpha_pred[trimap == 2] = 1
            alpha_pred[trimap == 0] = 0

            alpha_pred = alpha_pred[0][0]
            alpha_pred = alpha_pred.cpu()
            return alpha_pred

            
def inference(img_ori, trimap):

    test_config = {'checkpoint_path':"./checkpoints", 'checkpoint':"Weight_qt_in_use"}

    """长宽处理到32的倍数,保存输入长宽，之后剪裁回来"""
    h_ori, w_ori = trimap.shape

    target_h = 32 * ((h_ori - 1) // 32 + 1)
    target_w = 32 * ((w_ori - 1) // 32 + 1)

    # img = cv2.resize(img_ori, (target_w, target_h))
    # trimap = cv2.resize(trimap, (target_w, target_h))
    pad_h = target_h - h_ori
    pad_w = target_w - w_ori
    img = np.pad(img_ori, ((0,pad_h), (0, pad_w), (0,0)), mode="reflect")
    trimap = np.pad(trimap, ((0,pad_h), (0, pad_w)), mode="reflect")


    """转为tensor"""
    
    
    trimap[trimap < 20] = 0
    trimap[trimap > 230] = 2
    trimap[(trimap>=20) & (trimap<=230)] = 1


    
    # print(trimap.max())
    
    # cv2.imshow('trimap', trimap)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    trimap_inp = trimap.copy()
    trimap_inp = torch.from_numpy(trimap_inp).float().unsqueeze(0)
    # trimap_inp = F.one_hot(trimap_inp, num_classes=3).permute(2,0,1).float()



    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.
    img = torch.from_numpy(img)
    img = img.sub_(mean).div_(std)





    tester = Tester_one_image(test_config)
    alpha_pred = tester.test(img, trimap_inp)

    



    """将预测alpha图、新的背景图转换为输入原图的大小，输出的alpha图剪裁一下就行了"""
    test_pred = alpha_pred.data.cpu().numpy() * 255
    test_pred = test_pred.astype(np.uint8)
    test_pred = test_pred[:h_ori, :w_ori]

    test_pred = test_pred.astype(np.float)/255

    fg = img_ori * test_pred[:, :, None]
    fg = fg.astype(np.uint8)

    test_pred*=255
    test_pred = test_pred.astype(np.uint8)
    
    

    return test_pred, fg




if __name__=='__main__':

    pass