import sys
import socket

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *

import threading
from time import sleep
import time
import os

import numpy as np
from PIL import Image, ImageQt
# import torch
# from  torch.nn import functional as F

# import networks
# import utils
# import os

from paint_board import PaintBoard


#########################
# class KWSplot(QWidget):
#     def __init__(self, parent=None):
#         super(KWSplot, self).__init__(parent)
#         """窗口"""
#         self.setWindowModality(Qt.WindowModal)
#         self.resize(600, 600)
#         self.setWindowTitle('trimap 画板')

#         #setMouseTracking设置为False，否则不按下鼠标时也会跟踪鼠标事件
#         self.setMouseTracking(False)

#         """变量"""
#         self.pos_x = 20
#         self.pos_y = 20

#         """UI"""
#         self.setUpUI()

#         self.show()


#     def paintEvent(self, event):
#         painter = QPainter()
#         painter.begin(self)
#         pen = QPen(Qt.black, 2, Qt.SolidLine)
#         painter.setPen(pen)
        
#         #定点(20, 20) 到 (self.pos_x, self.pos_y)之间画线
#         painter.drawLine(20, 20, self.pos_x, self.pos_y)
        
#         painter.end()
        
#     def mouseMoveEvent(self, event):
#         '''
#         按住鼠标移动事件：更新pos_x和pos_y的值
#         调用update()函数在这里相当于调用paintEvent()函数
#         每次update()时，之前调用的paintEvent()留下的痕迹都会清空
#         '''
#         self.pos_x = event.pos().x()
#         self.pos_y = event.pos().y()
        
#         self.update()



    
#     def setUpUI(self):
#         #布局
#         self.total_layout = QVBoxLayout()

#         # --图片标签
#         self.img_label = QLabel("输入图片")
#         self.img_label.setAlignment(Qt.AlignCenter)


#         # --图片
        
#         self.img_imglabel = QLabel()
#         self.img_imglabel.setFixedSize(300,300)
#         # self.img_imglabel.setGeometry(0,0,400,400)
#         # self.img_imglabel.setScaledContents(True)
#         # self.picture = QPixmap('waveform.jpg')
#         # self.piclabel.setPixmap(self.picture)
#         # self.piclabel.setScaledContents(True)#是否填充


class PaintWindow(QDialog):
    
    def __init__(self, Parent=None, img_ori = None):
        super().__init__(Parent)
        self.img = img_ori 
        self.img_PIL = Image.fromarray(self.img.astype('uint8')[:,:,::-1]).convert('RGB')#转换为PIL格式

        # self.img_qimage = ImageQt.ImageQt(image)#转换为qimage

        self.__InitData()
        self.__InitView()

        
    def __InitData(self):
        '''
                  初始化成员变量
        '''
        self.__paintBoard = PaintBoard(self, self.img_PIL)
        #获取颜色列表(字符串类型)
        self.__colorList = QColor.colorNames() 
        
    def __InitView(self):
        '''
                  初始化界面
        '''
        # self.setFixedSize(640,480)
        self.setWindowTitle("Trimap painter")
        
        #新建一个水平布局作为本窗体的主布局
        main_layout = QHBoxLayout(self) 
        #设置主布局内边距以及控件间距为10px
        main_layout.setSpacing(10) 
    
        #在主界面左侧放置画板
        main_layout.addWidget(self.__paintBoard) 
        
        #新建垂直子布局用于放置按键
        sub_layout = QVBoxLayout() 
        
        #设置此子布局和内部控件的间距为10px
        sub_layout.setContentsMargins(10, 10, 10, 10) 

        self.__btn_Clear = QPushButton("清空画板")
        self.__btn_Clear.setParent(self) #设置父对象为本界面
       
        #将按键按下信号与画板清空函数相关联
        self.__btn_Clear.clicked.connect(self.__paintBoard.Clear) 
        sub_layout.addWidget(self.__btn_Clear)
        
        self.__btn_Quit = QPushButton("退出")
        self.__btn_Quit.setParent(self) #设置父对象为本界面
        self.__btn_Quit.clicked.connect(self.Quit)
        sub_layout.addWidget(self.__btn_Quit)

        self.__btn_save_and_Quit = QPushButton("完成")
        self.__btn_save_and_Quit.setParent(self) #设置父对象为本界面
        self.__btn_save_and_Quit.clicked.connect(self.save_drawn_img_and_quit)
        sub_layout.addWidget(self.__btn_save_and_Quit)
        
        self.__btn_Save = QPushButton("保存作品")
        self.__btn_Save.setParent(self)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)
        sub_layout.addWidget(self.__btn_Save)
        
        self.__cbtn_Eraser = QCheckBox("  使用橡皮擦")
        self.__cbtn_Eraser.setParent(self)
        self.__cbtn_Eraser.clicked.connect(self.on_cbtn_Eraser_clicked)
        sub_layout.addWidget(self.__cbtn_Eraser)
        
        splitter = QSplitter(self) #占位符
        sub_layout.addWidget(splitter)
        
        self.__label_penThickness = QLabel(self)
        self.__label_penThickness.setText("画笔粗细")
        self.__label_penThickness.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penThickness)
        
        self.__spinBox_penThickness = QSpinBox(self)
        self.__spinBox_penThickness.setMaximum(40)
        self.__spinBox_penThickness.setMinimum(2)
        self.__spinBox_penThickness.setValue(20) #默认粗细为10
        self.__spinBox_penThickness.setSingleStep(2) #最小变化值为2
        self.__spinBox_penThickness.valueChanged.connect(self.on_PenThicknessChange)#关联spinBox值变化信号和函数on_PenThicknessChange
        sub_layout.addWidget(self.__spinBox_penThickness)
        
        self.__label_penColor = QLabel(self)
        self.__label_penColor.setText("画笔颜色")
        self.__label_penColor.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penColor)
        
        self.__comboBox_penColor = QComboBox(self)
        self.__fillColorList(self.__comboBox_penColor) #用各种颜色填充下拉列表
        self.__comboBox_penColor.currentIndexChanged.connect(self.on_PenColorChange) #关联下拉列表的当前索引变更信号与函数on_PenColorChange
        sub_layout.addWidget(self.__comboBox_penColor)

        main_layout.addLayout(sub_layout) #将子布局加入主布局


    def __fillColorList(self, comboBox):

        index_black = 0
        index = 0
        for color in self.__colorList: 
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70,20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix),None)
            comboBox.setIconSize(QSize(70,20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(index_black)
        
    def on_PenColorChange(self):
        color_index = self.__comboBox_penColor.currentIndex()
        color_str = self.__colorList[color_index]
        self.__paintBoard.ChangePenColor(color_str)

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__paintBoard.ChangePenThickness(penThickness)
    
    def on_btn_Save_Clicked(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.png')
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return
        image = self.__paintBoard.GetContentAsQImage()
        image.save(savePath[0])
        
    def on_cbtn_Eraser_clicked(self):
        if self.__cbtn_Eraser.isChecked():
            self.__paintBoard.EraserMode = True #进入橡皮擦模式
        else:
            self.__paintBoard.EraserMode = False #退出橡皮擦模式
        
        
    def Quit(self):
        self.close()

    def save_drawn_img_and_quit(self):
        board_qimage = self.__paintBoard.GetContentAsQImage()
        board_PIL = ImageQt.fromqimage(board_qimage)
        board_PIL = board_PIL.resize((self.__paintBoard.inp_img_w, self.__paintBoard.inp_img_h))
        board_PIL.save("trimap_temp.png", "PNG")

        # self.close()






        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = PaintWindow()
    mainWindow.show()
    sys.exit(app.exec_())

