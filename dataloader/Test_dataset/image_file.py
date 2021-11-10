import os
import glob
import logging
import functools
import numpy as np

class ImageFile(object):
    def __init__(self, phase='train'):
        # self.logger = logging.getLogger("Logger")
        self.phase = phase
        self.rng = np.random.RandomState(0)#伪随机数生成器

    def _get_valid_names(self, *dirs, shuffle=True):
        # Extract valid names
        name_sets = [self._get_name_set(d) for d in dirs]
        # print("====================",dirs)
        # Reduce
        def _join_and(a, b):
            # print("**********",len(a),len(b))
            return a & b
        # print("---------------", len(name_sets[1]))
        # print(_join_and)
        valid_names = list(functools.reduce(_join_and, name_sets))#文件名，没有后缀，找出FG和GT_alpha里都有的名字
        # print("+++++++++++",valid_names)
        if shuffle:
            self.rng.shuffle(valid_names)

        if len(valid_names) == 0:
            # print('========',dirs)
            print('No image valid')
        else:
            # print('--------',dirs)
            print('{}: {} foreground/images are valid'.format(self.phase.upper(), len(valid_names)))

        return valid_names

    @staticmethod
    def _get_name_set(dir_name):
        path_list = glob.glob(os.path.join(dir_name, '*'))
        name_set = set()
        for path in path_list:
            name = os.path.basename(path)#返回path最后的文件名
            name = os.path.splitext(name)[0]#分离文件名与扩展名
            name_set.add(name)
        return name_set

    @staticmethod
    def _list_abspath(data_dir, ext, data_list):#得到图像的绝对路径
        return [os.path.join(data_dir, name + ext)
                for name in data_list]


class ImageFileTrain(ImageFile):
    def __init__(self,
                 alpha_dir="train_alpha",
                 fg_dir="train_fg",
                 bg_dir="train_bg",
                 alpha_ext=".jpg",
                 fg_ext=".jpg",
                 bg_ext=".jpg"):
        super(ImageFileTrain, self).__init__(phase="train")

        self.alpha_dir  = alpha_dir
        self.fg_dir     = fg_dir
        self.bg_dir     = bg_dir
        self.alpha_ext  = alpha_ext
        self.fg_ext     = fg_ext
        self.bg_ext     = bg_ext

        print('Load Training Images From Folders')

        self.valid_fg_list = self._get_valid_names(self.fg_dir, self.alpha_dir)#获取文件名交集
        self.valid_bg_list = [os.path.splitext(name)[0] for name in os.listdir(self.bg_dir)]

        self.alpha = self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_fg_list)#将路径连接之前获取的文件名交集
        self.fg = self._list_abspath(self.fg_dir, self.fg_ext, self.valid_fg_list)
        self.bg = self._list_abspath(self.bg_dir, self.bg_ext, self.valid_bg_list)

    def __len__(self):
        return len(self.alpha)


class ImageFileTest(ImageFile):
    def __init__(self,
                 alpha_dir="test_alpha",
                 merged_dir="test_merged",
                 trimap_dir="test_trimap",
                 guidancemap_dir = "test_guidancemap",
                 alpha_ext=".png",
                 merged_ext=".png",
                 trimap_ext=".png",
                 guidancemap_ext=".png"):
        super(ImageFileTest, self).__init__(phase="test")

        self.alpha_dir  = alpha_dir
        self.merged_dir = merged_dir
        self.trimap_dir = trimap_dir
        self.guidancemap_dir = guidancemap_dir

        self.alpha_ext  = alpha_ext
        self.merged_ext = merged_ext
        self.trimap_ext = trimap_ext
        self.guidancemap_ext = guidancemap_ext

        print('Load Testing Images From Folders')

        # self.valid_image_list = self._get_valid_names(self.alpha_dir, self.merged_dir, shuffle=False)
        self.valid_image_list = self._get_valid_names(self.alpha_dir, self.merged_dir, self.trimap_dir, shuffle=False)

        self.alpha = self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_image_list)
        self.merged = self._list_abspath(self.merged_dir, self.merged_ext, self.valid_image_list)
        self.trimap = self._list_abspath(self.trimap_dir, self.trimap_ext, self.valid_image_list)
        self.guidancemap = None
        if(self.guidancemap_dir == None):
            self.guidancemap = None
        else:
            self.guidancemap = self._list_abspath(self.guidancemap_dir, self.guidancemap_ext, self.valid_image_list)

    def __len__(self):
        return len(self.alpha)


if __name__ == '__main__':


    # train_data = ImageFileTrain(alpha_dir="/home/dell/Lun4/wrr/Deep-Image-Matting-v2-master/data/train/mask",
    #                             fg_dir="/home/dell/Lun4/wrr/Deep-Image-Matting-v2-master/data/train/fg",
    #                             bg_dir="/home/dell/Lun4/wrr/Deep-Image-Matting-v2-master/data/train/bg",
    #                             alpha_ext=".jpg",
    #                             fg_ext=".jpg",
    #                             bg_ext=".jpg")
    test_data = ImageFileTest(alpha_dir="/home/dell/Lun4/wrr/Deep-Image-Matting-v2-master/data/test/alpha_copy",
                              merged_dir="/home/dell/Lun4/wrr/Deep-Image-Matting-v2-master/data/test/merged",
                              trimap_dir="/home/dell/Lun4/wrr/Deep-Image-Matting-v2-master/data/test/trimaps",
                              alpha_ext=".png",
                              merged_ext=".png",
                              trimap_ext=".png")

    # print(train_data.alpha[0], train_data.fg[0], train_data.bg[0])
    # print(len(train_data.alpha), len(train_data.fg), len(train_data.bg))
    print(test_data.alpha[0], test_data.merged[0], test_data.trimap[0])
    print(len(test_data.alpha), len(test_data.merged), len(test_data.trimap))
