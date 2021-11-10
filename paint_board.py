from PyQt5.QtWidgets import QWidget
from PyQt5.Qt import QPixmap, QPainter, QPoint, QPaintEvent, QMouseEvent, QPen,\
    QColor, QSize
from PyQt5.QtCore import Qt
from PIL import Image, ImageQt
# from cv2 import findTransformECC
import numpy as np
# from torch._C import _cuda_resetAccumulatedMemoryStats
import copy

class PaintBoard(QWidget):


    def __init__(self, Parent=None, bg_img = None):#bg_img为PIL格式
        '''
        Constructor
        '''
        super().__init__(Parent)

        self.bg_img = bg_img

        self.disp_img_longside = 800

        if(self.bg_img!=None):
            self.inp_img_w, self.inp_img_h = self.bg_img.size

            
            if(self.inp_img_w >= self.inp_img_h):
                self.disp_img_w = 800
                self.disp_img_h = int(self.inp_img_h*(800.0/self.inp_img_w))
            else:
                self.disp_img_h = 800
                self.disp_img_w = int(self.inp_img_w*(800.0/self.inp_img_h))


            self.bg_img = self.bg_img.resize((self.disp_img_w, self.disp_img_h))
        else:
            self.inp_img_w, self.inp_img_h = 800, 800
            self.disp_img_w = 800
            self.disp_img_h = 800

            self.bg_img = np.ones([self.disp_img_h, self.disp_img_w, 3]).astype(np.uint8)*255
            self.bg_img = Image.fromarray(self.bg_img).convert('RGB')#转换为PIL格式
        

        self.__InitData() #先初始化数据，再初始化界面
        self.__InitView()

        #绘图工具相关
        self.__painter = QPainter()#新建绘图工具
        self.__thickness = 10       #默认画笔粗细为10px
        self.__penColor = QColor("black")#设置默认画笔颜色为黑色
        self.__colorList = QColor.colorNames() #获取颜色列表
        
    def __InitData(self):
        self.__size = QSize(self.disp_img_w, self.disp_img_h)
        
        #新建QPixmap作为画板，尺寸为__size
        self.__board = QPixmap(self.__size)#这是主画板，缩放之后进行的绘制都要同步到这一张上来
        self.__board.fill(Qt.gray) #用灰色填充画板
        
        self.__IsEmpty = True #默认为空画板 
        self.EraserMode = False #默认为禁用橡皮擦模式
        
        self.__lastPos = QPoint(0,0)#上一次鼠标位置
        self.__currentPos = QPoint(0,0)#当前的鼠标位置
        
     
    def __InitView(self):
        #设置界面的尺寸为__size
        self.setFixedSize(self.__size)

    def Reset_paintboard(self, bg_img):#把新图片加载到画板上
        self.bg_img = bg_img

        self.inp_img_w, self.inp_img_h = self.bg_img.size

        if(self.inp_img_w >= self.inp_img_h):
            self.disp_img_w = 800
            self.disp_img_h = int(self.inp_img_h*(800.0/self.inp_img_w))
        else:
            self.disp_img_h = 800
            self.disp_img_w = int(self.inp_img_w*(800.0/self.inp_img_h))

        self.bg_img = self.bg_img.resize((self.disp_img_w, self.disp_img_h))


        self.__InitData() #先初始化数据，再初始化界面
        self.__InitView()

        self.update()



    def Clear(self):
        #清空画板
        self.__board.fill(Qt.gray)


        self.update()
        self.__IsEmpty = True
        
    def ChangePenColor(self, color="black"):
        #改变画笔颜色
        self.__penColor = QColor(color)
        
    def ChangePenThickness(self, thickness=10):
        #改变画笔粗细
        self.__thickness = thickness
        
    def IsEmpty(self):
        #返回画板是否为空
        return self.__IsEmpty
    
    def GetContentAsQImage(self):
        #获取画板内容（返回QImage）
        image = self.__board.toImage()
        return image

    def GetContentAsPILImage(self):
        #获取画板内容（返回PIL）
        image = self.__board.toImage()
        image_PIL = ImageQt.fromqimage(image)
        image_array = np.array(image_PIL).astype(np.uint8)[:,:,::-1]#RGB转BGR

        import cv2
        image_array = cv2.resize(image_array, (self.inp_img_w, self.inp_img_h), interpolation=cv2.INTER_NEAREST)

        image_PIL_resized = Image.fromarray(image_array[:,:,::-1]).convert('RGB')
        # print("in", (self.inp_img_w, self.inp_img_h))
        # image_PIL.resize((self.inp_img_w, self.inp_img_h), Image.NEAREST)
        # print("out", image_PIL.size)
        return image_PIL_resized
        
    def paintEvent(self, paintEvent):
        #绘图事件
        #只要窗口部件需要被重绘就被调用 update可以调用
        #绘图时必须使用QPainter的实例，此处为__painter
        #绘图在begin()函数与end()函数间进行
        #begin(param)的参数要指定绘图设备，即把图画在哪里
        #drawPixmap用于绘制QPixmap类型的对象

        self.__painter.begin(self)#表示在整个控件上画
        # 0,0为绘图的左上角起点的坐标，__board即要绘制的图
        

        #self.__draw_board转换为array
        bg_qimage = self.__board.toImage()
        bg_PIL = ImageQt.fromqimage(bg_qimage)
        bg_array = np.array(bg_PIL)

        #和self.local_bg_img融合
        fuse = 0.5*bg_array+0.5*np.array(self.bg_img)
        fuse_PIL = Image.fromarray(fuse.astype('uint8')).convert('RGB')
        fuse_qimage = ImageQt.ImageQt(fuse_PIL)


        #转换成Qpixmap显示
        fuse_qpixmap = QPixmap.fromImage(fuse_qimage).scaled(self.disp_img_w, self.disp_img_h)
        self.__painter.drawPixmap(0,0,fuse_qpixmap)

        self.__painter.end()
        
    def mousePressEvent(self, mouseEvent):
        #鼠标按下时，获取鼠标的当前位置保存为上一次位置
        self.__currentPos =  mouseEvent.pos()

        self.__painter.begin(self.__board)#表示在Qpixmap上画
        
        if self.EraserMode == False:
            #非橡皮擦模式
            self.__painter.setPen(QPen(self.__penColor,self.__thickness, cap = 	Qt.RoundCap)) #设置画笔颜色，粗细
        else:
            #橡皮擦模式下画笔为灰色，粗细为20
            self.__painter.setPen(QPen(Qt.gray, 20, cap = 	Qt.RoundCap))
            
        #画
        self.__painter.drawPoints(self.__currentPos)
        self.__painter.end()
        self.__lastPos = self.__currentPos

                
        self.update() #更新显示

        
    def mouseMoveEvent(self, mouseEvent):
        #鼠标移动时，更新当前位置，并在上一个位置和当前位置间画线
        self.__currentPos =  mouseEvent.pos()
        self.__painter.begin(self.__board)#表示在Qpixmap上画
        
        if self.EraserMode == False:
            #非橡皮擦模式
            self.__painter.setPen(QPen(self.__penColor,self.__thickness, cap = 	Qt.RoundCap)) #设置画笔颜色，粗细
        else:
            #橡皮擦模式下画笔为灰色，粗细为20
            self.__painter.setPen(QPen(Qt.gray, 20, cap = 	Qt.RoundCap))
            
        #画线    
        self.__painter.drawLine(self.__lastPos, self.__currentPos)
        self.__painter.end()
        self.__lastPos = self.__currentPos


        self.update() #更新显示




    def Set_paint_image(self, paintImg_PIL):#加载已经画好的trimap
        #先加载trimap到self.__board中
        paintImg_PIL.resize((self.disp_img_w, self.disp_img_h), Image.NEAREST)
        paintImg_qimage = ImageQt.ImageQt(paintImg_PIL)
        paintImg_qpixmap = QPixmap.fromImage(paintImg_qimage).scaled(self.disp_img_w, self.disp_img_h)

        self.__painter.begin(self.__board)

        self.__painter.drawPixmap(0,0,paintImg_qpixmap)

        self.__painter.end()

        self.update()


        
    def mouseReleaseEvent(self, mouseEvent):
        self.__IsEmpty = False #画板不再为空





        




        
        
            




