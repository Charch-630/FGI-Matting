import sys
import os
import copy

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *

import numpy as np
from PIL import Image, ImageQt

# from trimap_painter import PaintWindow
from paint_board import PaintBoard


#########################
class mattingWindow(QMainWindow):
    def __init__(self, parent=None):
        super(mattingWindow, self).__init__(parent)
        # self.setWindowModality(Qt.ApplicationModal)
        # print("init")
        self.resize(1800, 1000)
        self.setWindowTitle('FGI-matting')

        #获取画板区域和颜色
        self.__paintBoard = PaintBoard(self, None)
        self.__colorList = ["black", "white"]#QColor.colorNames()
        #UI初始化
        self.setUpUI()

        self.img_ori = None
        self.img_PIL = None
        self.trimap = None
        self.test_pred = None
        import cv2
        self.new_bg = np.ones((400, 400, 3), np.uint8)*255

        


    def save_alpha_Btn_callback(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', 'alpha.png')
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return

        import cv2
        cv2.imwrite(savePath[0], self.test_pred)



    def save_fg_Btn_callback(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', 'FG.png')
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return

        import cv2
        cv2.imwrite(savePath[0], self.fg)


    def save_finalimg_Btn_callback(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', 'composition.png')
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return

        import cv2
        cv2.imwrite(savePath[0], self.new_image)



    def loadimg_Btn_callback(self):
        fileName,fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取img", os.getcwd(), 
        "All Files(*)")

        print(fileName)
        if fileName == "":
            print("Select cancel")
            return

        import cv2
        self.img_ori = cv2.imread(fileName, 1)

        label_width = self.img_imglabel.width()
        label_height = self.img_imglabel.height()

        # print(label_height, label_width)

        self.img_PIL = Image.fromarray(self.img_ori.astype('uint8')[:,:,::-1]).convert('RGB')#转换为PIL格式

        """对图片做resize和pad"""
        img_ori_disp = self.img_disp_preprocess(self.img_ori[:,:,::-1], label_width, label_height)

        """显示图片"""
        temp_imgSrc = QImage(img_ori_disp, img_ori_disp.shape[1], img_ori_disp.shape[0], img_ori_disp.shape[1]*3, QImage.Format_RGB888)
        pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)
        self.img_imglabel.setPixmap(pixmap_imgSrc)

        """在画图板中显示图片"""
        self.__paintBoard.Reset_paintboard(self.img_PIL)
        print(self.img_PIL.size)




    def loadbg_Btn_callback(self):
        fileName,fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取背景", os.getcwd(), 
        "All Files(*)")

        print(fileName)
        if fileName == "":
            print("Select cancel")
            return

        import cv2
        self.new_bg = cv2.imread(fileName, 1)

        label_width = self.img_imglabel.width()
        label_height = self.img_imglabel.height()

        if(type(self.test_pred) == type(None)):#如果没有预测结果则直接显示背景图
            """对图片做resize和pad"""
            new_img_disp = self.img_disp_preprocess(self.new_bg[:,:,::-1], label_width, label_height)

            """显示图片"""
            temp_imgSrc = QImage(new_img_disp, new_img_disp.shape[1], new_img_disp.shape[0], new_img_disp.shape[1]*3, QImage.Format_RGB888)
            pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)
            self.result_imglabel.setPixmap(pixmap_imgSrc)

        elif(type(self.img_ori) != type(None)):#如果 已经有了预测的alpha 且原图不为None 则用alpha图合成新图再进行显示
            h_ori = self.test_pred.shape[0]
            w_ori = self.test_pred.shape[1]
            new_bg = cv2.resize(self.new_bg, (w_ori, h_ori))

            alpha = self.test_pred.astype(np.float)/255
            self.new_image = self.img_ori* alpha[:, :]  + new_bg * (1.0 - alpha[:, :])
            self.new_image = self.new_image.astype(np.uint8)

            cv2.imwrite("finalimg_auto.png", self.new_image)
            new_img_disp =  cv2.cvtColor(self.new_image, cv2.COLOR_BGR2RGB)
            label_width = self.result_imglabel.width()
            label_height = self.result_imglabel.height()

            """对图片做resize和pad"""
            new_img_disp = self.img_disp_preprocess(new_img_disp, label_width, label_height)

            """显示图片"""
            temp_imgSrc = QImage(new_img_disp, new_img_disp.shape[1], new_img_disp.shape[0], new_img_disp.shape[1]*3, QImage.Format_RGB888)
            pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)
            self.result_imglabel.setPixmap(pixmap_imgSrc)




    # def drawtrimap_Btn_callback(self):

    #     if(self.img_ori.all()!=None):
    #         # self.paintdialogue = PaintWindow(img_ori = self.img_ori)
    #         # self.paintdialogue.show()

    #         # self.paintdialogue.setWindowModality(Qt.NonModal)
    #         # self.paintdialogue.exec_()

    #         self.__paintBoard.Reset_paintboard(self.img_PIL)

    #         import cv2
    #         self.trimap = cv2.imread("trimap_temp.png", 0) 

    #         label_width = self.img_imglabel.width()
    #         label_height = self.img_imglabel.height()

    #         """对图片做resize和pad"""
    #         # trimap_disp = cv2.imread("trimap_temp.png", 1)
    #         # trimap_disp = self.img_disp_preprocess(trimap_disp[:,:
    # ,::-1], label_width, label_height)

    #         # """显示图片"""
    #         # temp_imgSrc = QImage(trimap_disp, trimap_disp.shape[1], trimap_disp.shape[0], trimap_disp.shape[1]*3, QImage.Format_RGB888)
    #         # pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)
    #         # self.trimap_imglabel.setPixmap(pixmap_imgSrc)

    #     else:
    #         print("未读入原图")
        

    

    def pred_Btn_callback(self):
        from test_one_img import inference
        import cv2

        """获得画板上的trimap"""
        self.trimap  = self.__paintBoard.GetContentAsPILImage()
        self.trimap = np.array(self.trimap).astype(np.uint8)[:, :, ::-1]
        self.trimap = cv2.cvtColor(self.trimap, cv2.COLOR_RGB2GRAY)

        # print(self.img_ori.shape)
        # print(self.trimap.shape)
        """inference"""
        self.test_pred, self.fg = inference(self.img_ori, self.trimap)

        #print(self.test_pred.shape, self.trimap.shape)



        #alpha
        
        label_width = self.alpha_imglabel.width()
        label_height = self.alpha_imglabel.height()
        cv2.imwrite("alpha_auto.png", self.test_pred)

        self.test_pred = cv2.cvtColor(self.test_pred, cv2.COLOR_GRAY2RGB)

        """对图片做resize和pad"""
        test_pred_disp = self.img_disp_preprocess(self.test_pred, label_width, label_height)

        """显示图片"""
        temp_imgSrc = QImage(test_pred_disp, test_pred_disp.shape[1], test_pred_disp.shape[0], test_pred_disp.shape[1]*3, QImage.Format_RGB888)
        pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)

        self.alpha_imglabel.setPixmap(pixmap_imgSrc)



        #fg
        cv2.imwrite("fg_auto.png", self.fg)
        fg_ = copy.deepcopy(self.fg)
        fg_ =  cv2.cvtColor(self.fg, cv2.COLOR_BGR2RGB)
        label_width = self.fg_imglabel.width()
        label_height = self.fg_imglabel.height()

        """对图片做resize和pad"""
        fg_disp = self.img_disp_preprocess(fg_, label_width, label_height)

        """显示图片"""
        temp_imgSrc = QImage(fg_disp, fg_disp.shape[1], fg_disp.shape[0], fg_disp.shape[1]*3, QImage.Format_RGB888)
        pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)
        self.fg_imglabel.setPixmap(pixmap_imgSrc)


        #new_img
        h_ori  = self.test_pred.shape[0]
        w_ori  = self.test_pred.shape[1]
        new_bg = cv2.resize(self.new_bg, (w_ori, h_ori))

        alpha = self.test_pred.astype(np.float)/255
        self.new_image = self.img_ori* alpha[:, :]  + new_bg * (1.0 - alpha[:, :])
        self.new_image = self.new_image.astype(np.uint8)

        cv2.imwrite("finalimg_auto.png", self.new_image)
        new_img_disp =  cv2.cvtColor(self.new_image, cv2.COLOR_BGR2RGB)
        label_width = self.result_imglabel.width()
        label_height = self.result_imglabel.height()

        """对图片做resize和pad"""
        new_img_disp = self.img_disp_preprocess(new_img_disp, label_width, label_height)

        """显示图片"""
        temp_imgSrc = QImage(new_img_disp, new_img_disp.shape[1], new_img_disp.shape[0], new_img_disp.shape[1]*3, QImage.Format_RGB888)
        pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(label_width, label_height)
        self.result_imglabel.setPixmap(pixmap_imgSrc)



    def img_disp_preprocess(self, img, label_width, label_height):#只能处理3通道图
        import cv2

        h, w, _ = img.shape

        h_resize = label_height#先将h resize到控件h
        resize = h_resize/h
        w_resize = w*resize
        if(w_resize > label_width):#如果图片宽度还是大于控件宽，则以w resize到控件宽为标准
            w_resize = label_width#先将w resize到控件w
            resize = w_resize/w
            h_resize = h*resize

        w_resize = int(w_resize)
        h_resize = int(h_resize)

        img_disp = cv2.resize(img, (w_resize, h_resize))

        pad_h = 0
        pad_w = 0
        if(h_resize < label_height):
            pad_h = int((label_height - h_resize)/2)
            img_disp = np.pad(img_disp, ((pad_h, pad_h), (pad_w, pad_w), (0,0)), mode='constant')
        elif(w_resize < label_width):
            pad_w = int((label_width - w_resize)/2)
            img_disp = np.pad(img_disp, ((pad_h, pad_h), (pad_w, pad_w), (0,0)), mode='constant')
        
        return img_disp

        #################################画板函数###########################
    def __fillColorList(self, comboBox):

        index_black = 0
        index = 0
        for color in self.__colorList: 
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(120,20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix),None)
            comboBox.setIconSize(QSize(120,20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(index_black)

    def on_PenColorChange(self):
        color_index = self.__comboBox_penColor.currentIndex()
        color_str = self.__colorList[color_index]
        self.__paintBoard.ChangePenColor(color_str)

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__paintBoard.ChangePenThickness(penThickness)


    def save_drawn_img_and_quit(self):
        board_qimage = self.__paintBoard.GetContentAsQImage()
        board_PIL = ImageQt.fromqimage(board_qimage)
        board_PIL = board_PIL.resize((self.__paintBoard.inp_img_w, self.__paintBoard.inp_img_h))
        board_PIL.save("trimap_temp.png", "PNG")


    def on_btn_Save_Clicked(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', 'trimap.png')
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return
        image = self.__paintBoard.GetContentAsPILImage()
        image.save(savePath[0], "PNG")

    def on_cbtn_Eraser_clicked(self):
        if self.__cbtn_Eraser.isChecked():
            self.__paintBoard.EraserMode = True #进入橡皮擦模式
        else:
            self.__paintBoard.EraserMode = False #退出橡皮擦模式


    def loadtrimap_Btn_callback(self):
        fileName,fileType = QtWidgets.QFileDialog.getOpenFileName(self, "从文件选取trimap", os.getcwd(), 
        "All Files(*)")

        print(fileName)
        if fileName == "":
            print("Select cancel")
            return

        import cv2
        self.trimap = cv2.imread(fileName, 1)

        """转PIL并加载到画板上"""
        trimap_PIL = Image.fromarray(self.trimap.astype('uint8')[:, :, -1]).convert('RGB')
        self.__paintBoard.Set_paint_image(trimap_PIL)
    
    def setUpUI(self):
        #布局
        self.total_layout = QHBoxLayout()

        self.imgs_btn_layout = QVBoxLayout()

        self.img_layout = QVBoxLayout()
        self.alpha_layout = QVBoxLayout()
        self.fg_layout = QVBoxLayout()
        self.result_layout = QVBoxLayout()

        self.imgs_row1_layout = QHBoxLayout()
        self.imgs_row2_layout = QHBoxLayout()
        self.imgs_layout = QVBoxLayout()

        self.btns_layout = QHBoxLayout()


        self.paint_board_layout = QHBoxLayout()
        self.paint_board_layout.setSpacing(10) #设置主布局内边距以及控件间距为10px
        self.paint_board_btns_layout = QVBoxLayout()
        self.paint_board_btns_layout.setContentsMargins(10, 10, 10, 10) #设置此子布局和内部控件的间距为10px


        # --图片标签
        self.img_label = QLabel("输入图片\ninput image")
        self.img_label.setAlignment(Qt.AlignCenter)

        self.alpha_label = QLabel("预测alpha\npredicted alpha")
        self.alpha_label.setAlignment(Qt.AlignCenter)

        self.fg_label = QLabel("预测前景\npredicted FG")
        self.fg_label.setAlignment(Qt.AlignCenter)

        self.result_label = QLabel("新图\ncomposition")
        self.result_label.setAlignment(Qt.AlignCenter)


        # --图片
        self.img_imglabel = QLabel()
        self.img_imglabel.setFixedSize(300,300)
        # self.img_imglabel.setGeometry(0,0,400,400)
        # self.img_imglabel.setScaledContents(True)
        # self.picture = QPixmap('waveform.jpg')
        # self.piclabel.setPixmap(self.picture)
        # self.piclabel.setScaledContents(True)#是否填充

        # self.trimap_imglabel = QLabel()
        # self.trimap_imglabel.setFixedSize(300,300)

        self.alpha_imglabel = QLabel()
        self.alpha_imglabel.setFixedSize(300,300)

        self.fg_imglabel = QLabel()
        self.fg_imglabel.setFixedSize(300,300)

        self.result_imglabel = QLabel()
        self.result_imglabel.setFixedSize(300,300)

        # --按钮
        #左下角
        self.loadimg_Btn= QPushButton("加载图片\nload image")
        self.loadimg_Btn.clicked.connect(self.loadimg_Btn_callback)

        self.loadbg_Btn= QPushButton("加载背景\nload BG")
        self.loadbg_Btn.clicked.connect(self.loadbg_Btn_callback)

        self.pred_Btn= QPushButton("预测\npredict")
        self.pred_Btn.clicked.connect(self.pred_Btn_callback)

        self.save_alpha_Btn= QPushButton("保存alpha\nsave alpha")
        self.save_alpha_Btn.clicked.connect(self.save_alpha_Btn_callback)

        self.save_fg_Btn= QPushButton("保存前景\nsave FG")
        self.save_fg_Btn.clicked.connect(self.save_fg_Btn_callback)

        self.save_finalimg_Btn= QPushButton("保存新图\nsave composition")
        self.save_finalimg_Btn.clicked.connect(self.save_finalimg_Btn_callback)


        #右上角
        self.__btn_Clear = QPushButton("清空画板\nclear paint board")
        # self.__btn_Clear.setParent(self) #设置父对象为本界面
        self.__btn_Clear.setMaximumWidth(150)
        self.__btn_Clear.clicked.connect(self.__paintBoard.Clear)

        self.__btn_Load = QPushButton("加载trimap\nload trimap")
        self.__btn_Load.setMaximumWidth(150)
        self.__btn_Load.clicked.connect(self.loadtrimap_Btn_callback)

        self.__btn_Save = QPushButton("保存trimap\nsave trimap")
        self.__btn_Save.setMaximumWidth(150)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)

        self.__cbtn_Eraser = QCheckBox("使用橡皮擦\neraser mode")
        self.__cbtn_Eraser.setMaximumWidth(150)
        self.__cbtn_Eraser.clicked.connect(self.on_cbtn_Eraser_clicked)

        #占位符
        self.splitter = QSplitter(self)

        #粗细设置框标签
        self.__label_penThickness = QLabel(self)
        self.__label_penThickness.setText("画笔粗细 thickness")
        self.__label_penThickness.setMaximumWidth(150)
        self.__label_penThickness.setFixedHeight(20)

        #粗细设置框
        self.__spinBox_penThickness = QSpinBox(self)#设置粗细
        self.__spinBox_penThickness.setMaximum(60)
        self.__spinBox_penThickness.setMinimum(2)
        self.__spinBox_penThickness.setValue(40) #默认粗细为10
        self.__spinBox_penThickness.setSingleStep(2) #最小变化值为2
        self.__spinBox_penThickness.setMaximumWidth(150)
        self.__spinBox_penThickness.valueChanged.connect(self.on_PenThicknessChange)#关联spinBox值变化信号和函数on_PenThicknessChange

        thickness = self.__spinBox_penThickness.value()#立即将粗细设置框中的数字设置为画笔的粗细
        self.__paintBoard.ChangePenThickness(thickness)

        #颜色下拉列表标签
        self.__label_penColor = QLabel(self)
        self.__label_penColor.setText("画笔颜色 color")
        self.__label_penColor.setMaximumWidth(150)
        self.__label_penColor.setFixedHeight(20)

        #颜色下拉列表
        self.__comboBox_penColor = QComboBox(self)
        self.__fillColorList(self.__comboBox_penColor) #用各种颜色填充下拉列表
        self.__comboBox_penColor.setMaximumWidth(150)
        self.__comboBox_penColor.currentIndexChanged.connect(self.on_PenColorChange) #关联下拉列表的当前索引变更信号与函数on_PenColorChange

        #局部布局添加控件
        #4个文本和img
        self.img_layout.addWidget(self.img_label)
        self.img_layout.addWidget(self.img_imglabel)
        # self.img1_Widiget = QWidget()
        # self.img1_Widiget.setLayout(self.img1_layout)

        self.alpha_layout.addWidget(self.alpha_label)
        self.alpha_layout.addWidget(self.alpha_imglabel)

        self.fg_layout.addWidget(self.fg_label)
        self.fg_layout.addWidget(self.fg_imglabel)

        self.result_layout.addWidget(self.result_label)
        self.result_layout.addWidget(self.result_imglabel)


        #按钮
        self.btns_layout.addWidget(self.loadimg_Btn)
        self.btns_layout.addWidget(self.loadbg_Btn)
        self.btns_layout.addWidget(self.pred_Btn)
        self.btns_layout.addWidget(self.save_alpha_Btn)
        self.btns_layout.addWidget(self.save_fg_Btn)
        self.btns_layout.addWidget(self.save_finalimg_Btn)


        #画板区域
        self.paint_board_btns_layout.addWidget(self.__btn_Clear)
        self.paint_board_btns_layout.addWidget(self.__btn_Save)
        self.paint_board_btns_layout.addWidget(self.__btn_Load)
        self.paint_board_btns_layout.addWidget(self.__cbtn_Eraser)
        self.paint_board_btns_layout.addWidget(self.splitter)
        self.paint_board_btns_layout.addWidget(self.__label_penThickness)
        self.paint_board_btns_layout.addWidget(self.__spinBox_penThickness)
        self.paint_board_btns_layout.addWidget(self.__label_penColor)
        self.paint_board_btns_layout.addWidget(self.__comboBox_penColor)

        self.paint_board_layout.addWidget(self.__paintBoard)
        self.paint_board_layout.addLayout(self.paint_board_btns_layout)


        #4个图结合在一起成为layout
        self.imgs_row1_layout.addLayout(self.img_layout)
        self.imgs_row1_layout.addLayout(self.alpha_layout)
        self.imgs_row2_layout.addLayout(self.fg_layout)
        self.imgs_row2_layout.addLayout(self.result_layout)
        self.imgs_layout.addLayout(self.imgs_row1_layout)
        self.imgs_layout.addLayout(self.imgs_row2_layout)

        self.imgs_btn_layout.addLayout(self.imgs_layout)
        self.imgs_btn_layout.addLayout(self.btns_layout)

        self.total_layout.addLayout(self.imgs_btn_layout)
        self.total_layout.addLayout(self.paint_board_layout)

        # self.setLayout(self.total_layout)

        self.total_widget = QWidget()
        self.total_widget.setLayout(self.total_layout)

        self.setCentralWidget(self.total_widget)


        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = mattingWindow()
    mainWindow.show()


    sys.exit(app.exec_())

