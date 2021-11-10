import cv2
import os
import math
import numbers
import random
import logging
import copy
import numpy as np

import torch
from   torch.utils.data import Dataset
from   torch.nn import functional as F
from   torchvision import transforms



trimap_channel = 1
random_interp = False
crop_size = 512
augmentation = True
radius = 15
radius_extend = 50
num_points =10




interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

def maybe_random_interp(cv2_interp):
    if random_interp:
        return np.random.choice(interp_list)
    else:
        return cv2_interp

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, phase="test"):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        self.phase = phase

    def __call__(self, sample):
        if(self.phase == "train"):
            # convert BGR images to RGB
            image, alpha, trimap = sample['image'][:,:,::-1], sample['alpha'], sample['trimap']
            # convert HWC to CHW
            image = image.transpose((2, 0, 1)).astype(np.float32)
            # normalize image
            image /= 255.

            alpha = np.expand_dims(alpha.astype(np.float32), axis=0)

            trimap = trimap.astype(np.float32)
            trimap[trimap < 85] = 0
            trimap[trimap >= 170] = 2
            trimap[trimap >= 85] = 1

             # convert BGR fg and bg to RGB
            fg = sample['fg'][:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['fg'] = torch.from_numpy(fg).sub_(self.mean).div_(self.std)
            bg = sample['bg'][:,:,::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['bg'] = torch.from_numpy(bg).sub_(self.mean).div_(self.std)

            sample['image'], sample['alpha'], sample['trimap'] = \
            torch.from_numpy(image), torch.from_numpy(alpha), torch.from_numpy(trimap)
            sample['image'] = sample['image'].sub_(self.mean).div_(self.std)

            if trimap_channel == 3:
                sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2,0,1).float()
            elif trimap_channel == 1:
                sample['trimap'] = sample['trimap'][None,...].float()
            else:
                raise NotImplementedError("trimap_channel can only be 3 or 1")
        
        elif(self.phase == "val"):
            # convert BGR images to RGB
            image, alpha, trimap = sample['image'][:,:,::-1], sample['alpha'], sample['trimap']
            # convert HWC to CHW
            image = image.transpose((2, 0, 1)).astype(np.float32)
            # normalize image
            image /= 255.
            alpha = np.expand_dims(alpha.astype(np.float32), axis=0)

            trimap = trimap.astype(np.float32)
            trimap[trimap < 85] = 0
            trimap[trimap >= 170] = 2
            trimap[trimap >= 85] = 1

            sample['image'], sample['alpha'], sample['trimap'] = \
            torch.from_numpy(image), torch.from_numpy(alpha), torch.from_numpy(trimap)
            sample['image'] = sample['image'].sub_(self.mean).div_(self.std)

            if trimap_channel == 3:
                sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2,0,1).float()
            elif trimap_channel == 1:
                sample['trimap'] = sample['trimap'][None,...].float()
            else:
                raise NotImplementedError("trimap_channel can only be 3 or 1")

        else:#test
            # convert BGR images to RGB
            image, alpha, trimap = sample['image'][:,:,::-1], sample['alpha'], sample['trimap']
            guidancemap_ = sample["guidancemap"]
            # convert HWC to CHW
            image = image.transpose((2, 0, 1)).astype(np.float32)
            # normalize image
            image /= 255.
            alpha = np.expand_dims(alpha.astype(np.float32), axis=0)

            guidancemap_ = guidancemap_.astype(np.float32)
            guidancemap_[guidancemap_ < 85] = 0       #0     bg
            guidancemap_[guidancemap_ >= 170] = 2  #255    fg
            guidancemap_[guidancemap_ >= 85] = 1    #128    tr

            trimap = trimap.astype(np.float32)
            trimap[trimap < 85] = 0
            trimap[trimap >= 170] = 2
            trimap[trimap >= 85] = 1

            sample['image'], sample['alpha'], sample['trimap'] = \
            torch.from_numpy(image), torch.from_numpy(alpha), torch.from_numpy(trimap).to(torch.long)
            sample['image'] = sample['image'].sub_(self.mean).div_(self.std)
            sample["guidancemap"] = torch.from_numpy(guidancemap_ ).to(torch.long)#为了能使用通道为3的情况

            if trimap_channel == 3:
                sample['guidancemap'] = F.one_hot(sample['guidancemap'], num_classes=3).permute(2,0,1).float()
                sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2,0,1).float()
            elif trimap_channel == 1:
                sample['guidancemap'] = sample['guidancemap'][None,...].float()
                sample['trimap'] = sample['trimap'][None,...].float()
            else:
                raise NotImplementedError("trimap_channel can only be 3 or 1")

        return sample


class RandomAffine(object):
    """
    Random affine translation
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        rows, cols, ch = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, fg.size)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, fg.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        fg = cv2.warpAffine(fg, M, (cols, rows),
                            flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        sample['fg'], sample['alpha'] = fg, alpha

        return sample


    @ staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        # C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        # RSS is rotation with scale and shear matrix
        # It is different from the original function in torchvision
        # The order are changed to flip -> scale -> rotation -> shear
        # x and y have different scale factors
        # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
        # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
        # [     0                       0                      1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix

class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        # if alpha is all 0 skip
        if np.all(alpha==0):
            return sample
        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        fg = cv2.cvtColor(fg.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = fg[:, :, 1][alpha > 0].mean()
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat>1] = 2 - sat[sat>1]
        fg[:, :, 1] = sat
        # Value noise
        val_bar = fg[:, :, 2][alpha > 0].mean()
        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val>1] = 2 - val[val>1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
        sample['fg'] = fg*255

        return sample


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        if np.random.uniform(0, 1) < self.prob:
            fg = cv2.flip(fg, 1)
            alpha = cv2.flip(alpha, 1)
        sample['fg'], sample['alpha'] = fg, alpha

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=( crop_size, crop_size)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        fg, alpha, trimap, name = sample['fg'],  sample['alpha'], sample['trimap'], sample['image_name']
        bg = sample['bg']
        h, w = trimap.shape
        bg = cv2.resize(bg, (w, h), interpolation=maybe_random_interp(cv2.INTER_CUBIC))#背景长宽变成alpha/trimap长宽
        if w < self.output_size[0]+1 or h < self.output_size[1]+1:
            ratio = 1.1*self.output_size[0]/h if h < w else 1.1*self.output_size[1]/w#计算短边resize到1.1*self.output_size的ratio
            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0]+1 or w < self.output_size[1]+1:#当长或宽小于self.output_size时，使用ratio对fg、alpha、trimap、bg进行resize
                fg = cv2.resize(fg, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha = cv2.resize(alpha, (int(w*ratio), int(h*ratio)),
                                   interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                trimap = cv2.resize(trimap, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_NEAREST)
                bg = cv2.resize(bg, (int(w*ratio), int(h*ratio)), interpolation=maybe_random_interp(cv2.INTER_CUBIC))
                h, w = trimap.shape
        small_trimap = cv2.resize(trimap, (w//4, h//4), interpolation=cv2.INTER_NEAREST)
        unknown_list = list(zip(*np.where(small_trimap[self.margin//4:(h-self.margin)//4,#self.margin为输出大小的一半，w/h是输出大小的1.1，这里的操作是抠出中间0.1的那一小块
                                                       self.margin//4:(w-self.margin)//4] == 128)))
        unknown_num = len(unknown_list)
        if len(unknown_list) < 10:
            # self.logger.warning("{} does not have enough unknown area for crop.".format(name))
            left_top = (np.random.randint(0, h-self.output_size[0]+1), np.random.randint(0, w-self.output_size[1]+1))#数量少，整张图随机
        else:
            idx = np.random.randint(unknown_num)#数量多，在那一小块里面挑一个点出来当左上角
            left_top = (unknown_list[idx][0]*4, unknown_list[idx][1]*4)

        fg_crop = fg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]
        alpha_crop = alpha[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]
        bg_crop = bg[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1],:]
        trimap_crop = trimap[left_top[0]:left_top[0]+self.output_size[0], left_top[1]:left_top[1]+self.output_size[1]]

        if len(np.where(trimap==128)[0]) == 0:#当整张trimap中过渡区域没有的时候，fg/alpha/trimap/bg都直接resize成指定输出大小
            self.logger.error("{} does not have enough unknown area for crop. Resized to target size."
                                "left_top: {}".format(name, left_top))
            fg_crop = cv2.resize(fg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(alpha, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            trimap_crop = cv2.resize(trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
            bg_crop = cv2.resize(bg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_CUBIC))
            # cv2.imwrite('../tmp/tmp.jpg', fg.astype(np.uint8))
            # cv2.imwrite('../tmp/tmp.png', (alpha*255).astype(np.uint8))
            # cv2.imwrite('../tmp/tmp2.png', trimap.astype(np.uint8))
            # raise ValueError("{} does    not have enough unknown area for crop.".format(name))

        sample['fg'], sample['alpha'], sample['trimap'] = fg_crop, alpha_crop, trimap_crop
        sample['bg'] = bg_crop

        return sample

class ResizeAug(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image =  sample['image']
        rand = np.random.randint(3)
        h,w = image.shape[:2]
        if(rand == 0):
            image = cv2.resize(image, (int(1.5*w), int(1.5*h)), interpolation = cv2.INTER_NEAREST)
            image = cv2.resize(image, (w, h), interpolation = cv2.INTER_NEAREST)
        elif(rand == 1):
            image = cv2.resize(image, (int(1.5*w), int(1.5*h)), interpolation = cv2.INTER_LINEAR)
            image = cv2.resize(image, (w, h), interpolation = cv2.INTER_LINEAR)
        else:
            image = cv2.resize(image, (int(1.5*w), int(1.5*h)), interpolation = cv2.INTER_CUBIC)
            image = cv2.resize(image, (w, h), interpolation = cv2.INTER_CUBIC)
        
        sample['image'] = image
        return sample

class Rescale(object):
    """
    Rescale the image in a sample to a given size.
    :param output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, alpha, trimap = sample['image'], sample['alpha'], sample['trimap']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        trimap = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        sample['image'], sample['alpha'], sample['trimap'] = image, alpha, trimap

        return sample


class OriginScale(object):
    def __init__(self, phase = "test"):
        self.phase = phase

    def __call__(self, sample):
        if(self.phase ==  "val"):
            h, w = sample["alpha_shape"]
            # sample['origin_trimap'] = sample['trimap']
            # # if h % 32 == 0 and w % 32 == 0:
            # #     return sample
            # # target_h = h - h % 32
            # # target_w = w - w % 32
            # target_h = 32 * ((h - 1) // 32 + 1)
            # target_w = 32 * ((w - 1) // 32 + 1)
            # sample['image'] = cv2.resize(sample['image'], (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            # sample['trimap'] = cv2.resize(sample['trimap'], (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            if h % 32 == 0 and w % 32 == 0:
                return sample
            target_h = 32 * ((h - 1) // 32 + 1)
            target_w = 32 * ((w - 1) // 32 + 1)
            pad_h = target_h - h
            pad_w = target_w - w

            padded_image = np.pad(sample['image'], ((0,pad_h), (0, pad_w), (0,0)), mode="reflect")
            padded_trimap = np.pad(sample['trimap'], ((0,pad_h), (0, pad_w)), mode="reflect")

            sample['image'] = padded_image
            sample['trimap'] = padded_trimap

        else:#test
            h, w = sample["alpha_shape"]
            # sample['origin_trimap'] = sample['trimap']
            # # if h % 32 == 0 and w % 32 == 0:
            # #     return sample
            # # target_h = h - h % 32
            # # target_w = w - w % 32
            # target_h = 32 * ((h - 1) // 32 + 1)
            # target_w = 32 * ((w - 1) // 32 + 1)
            # sample['image'] = cv2.resize(sample['image'], (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            # sample['trimap'] = cv2.resize(sample['trimap'], (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            if h % 32 == 0 and w % 32 == 0:
                return sample
            target_h = 32 * ((h - 1) // 32 + 1)
            target_w = 32 * ((w - 1) // 32 + 1)
            pad_h = target_h - h
            pad_w = target_w - w

            padded_image = np.pad(sample['image'], ((0,pad_h), (0, pad_w), (0,0)), mode="reflect")
            padded_trimap = np.pad(sample['trimap'], ((0,pad_h), (0, pad_w)), mode="reflect")
            padded_guidancemap = np.pad(sample['guidancemap'], ((0,pad_h), (0, pad_w)), mode="reflect")

            sample['image'] = padded_image
            sample['trimap'] = padded_trimap
            sample['guidancemap'] = padded_guidancemap

        return sample


class GenTrimap(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]

    def __call__(self, sample):
        alpha = sample['alpha']
        # Adobe 1K
        fg_width = np.random.randint(1, 30)
        bg_width = np.random.randint(1, 30)
        fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

        trimap = np.ones_like(alpha) * 128#tr
        trimap[fg_mask == 1] = 255#fg
        trimap[bg_mask == 1] = 0#bg

        sample['trimap'] = trimap
        return sample

class Genclickmap(object):
    def __init__(self, r=15, r_extend = 50, point_num=10):
        self.r = r
        self.r_extend = r_extend
        self.m = point_num

    def __call__(self, sample):
        trimap = sample['trimap']
        #前景采样
        fg_click_num = np.random.randint(1, self.m)#最少1个，最多m-1个
        fg_click_num_ = copy.deepcopy(fg_click_num)
        trimap_fg_sample = copy.deepcopy(trimap)
        fg_points = []
        while(fg_click_num_>0):
            fg_area = list(zip(*np.where(trimap_fg_sample==255)))#找出前景区域的坐标
            fg_num = len(fg_area)

            if(fg_num>0):
                point_chosen = fg_area[np.random.randint(fg_num)]
                fg_points.append(point_chosen)
                cv2.circle(trimap_fg_sample, (point_chosen[1], point_chosen[0]), self.r_extend, 128, -1)
                fg_click_num_-=1
            else:
                break#没点了，剩下的数量用来采样背景

        #背景采样
        bg_click_num = self.m - fg_click_num
        bg_click_num_ = copy.deepcopy(bg_click_num)
        trimap_bg_sample = copy.deepcopy(trimap)
        bg_points = []
        while(bg_click_num_>0):
            bg_area = list(zip(*np.where(trimap_bg_sample==0)))#找出前景区域的坐标
            bg_num = len(bg_area)

            if(bg_num>0):
                point_chosen = bg_area[np.random.randint(bg_num)]
                bg_points.append(point_chosen)
                cv2.circle(trimap_bg_sample, (point_chosen[1], point_chosen[0]), self.r_extend, 128, -1)
                bg_click_num_-=1
            else:
                break#如果连背景都没了那就不采了

        
        #画采样图  fg 255  bg 128 tr 0
        # clickmap = np.zeros_like(trimap)
        # # clickmap = trimap
        # for point in fg_points:
        #     cv2.circle(clickmap, (point[1], point[0]), self.r, 255, -1)

        # for point in bg_points:
        #     cv2.circle(clickmap, (point[1], point[0]), self.r, 128, -1)

        #把随机出来的坐标放在tensor里
        fg_points_tensor = torch.ones([10,2])*9999
        bg_points_tensor = torch.ones([10,2])*9999

        for i, point in enumerate(fg_points):
            fg_points_tensor[i] = torch.tensor(point)

        for i, point in enumerate(bg_points):
            bg_points_tensor[i] = torch.tensor(point)


        # sample['clickmap'] = clickmap
        sample["fg_points"] = fg_points_tensor.long()
        sample["bg_points"] = bg_points_tensor.long()
        return sample

class Composite(object):
    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        alpha[alpha < 0 ] = 0
        alpha[alpha > 1] = 1
        fg[fg < 0 ] = 0
        fg[fg > 255] = 255
        bg[bg < 0 ] = 0
        bg[bg > 255] = 255

        image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
        sample['image'] = image
        return sample


class DataGenerator(Dataset):
    def __init__(self, data, phase="train", test_scale="resize"):
        self.phase = phase
        self.crop_size = crop_size
        self.alpha = data.alpha
        if self.phase == "train":
            self.fg = data.fg
            self.bg = data.bg
            self.merged = []
            self.trimap = []
        else:
            self.fg = []
            self.bg = []
            self.merged = data.merged
            self.trimap = data.trimap
            self.guidancemap = data.guidancemap

        if augmentation:
            train_trans = [
                            RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),#对alpha和fg做仿射变换
                            GenTrimap(),#用alpha图产生trimap, 长宽和alpha图一样
                            RandomCrop((self.crop_size, self.crop_size)),
                            RandomJitter(),
                            Genclickmap(radius, radius_extend, num_points),
                            Composite(),#将fg/bg/alpha组合成为image
                            ResizeAug(),#RI 只对image
                            ToTensor(phase="train")]
        else:
            train_trans = [ GenTrimap(),
                            RandomCrop((self.crop_size, self.crop_size)),
                            Composite(),
                            ResizeAug(),
                            ToTensor(phase="train")]

        if test_scale.lower() == "origin":
            test_trans = [ OriginScale(), ToTensor(phase="test") ]
        elif test_scale.lower() == "resize":
            test_trans = [ Rescale((self.crop_size, self.crop_size)), ToTensor(phase="test") ]
        elif test_scale.lower() == "crop":
            test_trans = [ RandomCrop((self.crop_size, self.crop_size)), ToTensor(phase="test") ]
        else:
            raise NotImplementedError("test_scale {} not implemented".format(test_scale))

        self.transform = {
            'train':
                transforms.Compose(train_trans),
            'val':
                transforms.Compose([
                    OriginScale(phase="val"),
                    ToTensor(phase="val")
                ]),
            'test':
                transforms.Compose(test_trans)
        }[phase]

        self.fg_num = len(self.fg)
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,20)]

    def __getitem__(self, idx):
        if self.phase == "train":
            fg = cv2.imread(self.fg[idx % self.fg_num])
            alpha = cv2.imread(self.alpha[idx % self.fg_num], 0).astype(np.float32)/255
            bg = cv2.imread(self.bg[idx], 1)

            if augmentation:
                fg, alpha = self._composite_fg(fg, alpha, idx)
                pass

            image_name = os.path.split(self.fg[idx % self.fg_num])[-1]
            sample = {'fg': fg, 'alpha': alpha, 'bg': bg, 'image_name': image_name}

        elif self.phase == "val":
            image = cv2.imread(self.merged[idx])
            # image = cv2.imread(self.merged[idx].replace('png','jpg'))
            # print('======',self.merged[idx].replace('png','jpg'))
            # alpha = cv2.imread(self.alpha[idx].replace('png','jpg'), 0)/255.
            alpha = cv2.imread(self.alpha[idx], 0)/255.
            trimap = cv2.imread(self.trimap[idx], 0)
            image_name = os.path.split(self.merged[idx])[-1]

            sample = {'image': image, 'alpha': alpha, 'trimap': trimap,  'image_name': image_name, 'alpha_shape': alpha.shape}
        else:#test
            image = cv2.imread(self.merged[idx])
            # image = cv2.imread(self.merged[idx].replace('png','jpg'))
            # print('======',self.merged[idx].replace('png','jpg'))
            # alpha = cv2.imread(self.alpha[idx].replace('png','jpg'), 0)/255.
            alpha = cv2.imread(self.alpha[idx], 0)/255.
            if  self.guidancemap == None:
                guidancemap_ = np.ones_like(alpha).astype(np.uint8)*128
            else:
                guidancemap_ = cv2.imread(self.guidancemap[idx], 0)

            trimap = cv2.imread(self.trimap[idx], 0)
            image_name = os.path.split(self.merged[idx])[-1]

            sample = {'image': image, 'alpha': alpha, 'trimap': trimap, 'guidancemap': guidancemap_,  'image_name': image_name, 'alpha_shape': alpha.shape}

        sample = self.transform(sample)

        return sample

    def _composite_fg(self, fg, alpha, idx):

        if np.random.rand() < 0.5:
            idx2 = np.random.randint(self.fg_num) + idx
            fg2 = cv2.imread(self.fg[idx2 % self.fg_num])
            alpha2 = cv2.imread(self.alpha[idx2 % self.fg_num], 0).astype(np.float32)/255.
            h, w = alpha.shape
            fg2 = cv2.resize(fg2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha2 = cv2.resize(alpha2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

            alpha_tmp = 1 - (1 - alpha) * (1 - alpha2)
            if  np.any(alpha_tmp < 1):
                fg = fg.astype(np.float32) * alpha[:,:,None] + fg2.astype(np.float32) * (1 - alpha[:,:,None])
                # The overlap of two 50% transparency should be 25%
                alpha = alpha_tmp
                fg = fg.astype(np.uint8)

        if np.random.rand() < 0.25:
            fg = cv2.resize(fg, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha = cv2.resize(alpha, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        return fg, alpha

    def __len__(self):
        if self.phase == "train":
            return len(self.bg)
        else:
            return len(self.alpha)


if __name__ == '__main__':
    pass
    
