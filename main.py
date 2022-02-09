from det import *
from screenprint import *
from gamer import *
from utils.torch_utils import time_synchronized
import globalvar as gl

from PyQt5.QtWidgets import QApplication,QWidget, QTableWidgetItem
from PyQt5.uic import loadUi
from PyQt5 import QtCore

from threading import Thread

class GameWindow(QWidget):
    def __init__(self, parent=None):
        super(GameWindow, self).__init__(parent)
        loadUi("./game.ui", self)
        self.setFixedSize(self.sizeHint())
        self.resetButton.clicked.connect(self.reset)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        # self.cardDict = {"JOKER":0, "2":1, "A":2, "K":3, "Q":4, "J":5, "10":6, "9":7, "8":8, "7":9, "6":10, "5":11, "4":12, "3":13}

    def reset(self):
        thread = Thread(target=self.det_thread)
        thread.start()

    def changeCnt(self, s):
        for i in s:
            if i != 'PASS':
                # row = self.cardDict[i]
                # print(i,"/",row)
                # # cnt = self.cardTable.item(row,0).text()
                # # cnt = str(int(cnt) - 1)
                # # t = QTableWidgetItem(cnt)
                # # self.cardTable.setItem(row, 0, t)
                if i == "JOKER":
                    cnt = self.CJOKER.text()
                    cnt = str(int(cnt)-1)
                    self.CJOKER.setText(cnt)
                if i == "2":
                    cnt = self.C2.text()
                    cnt = str(int(cnt)-1)
                    self.C2.setText(cnt)
                if i == "A":
                    cnt = self.CA.text()
                    cnt = str(int(cnt)-1)
                    self.CA.setText(cnt)
                if i == "K":
                    cnt = self.CK.text()
                    cnt = str(int(cnt)-1)
                    self.CK.setText(cnt)
                if i == "Q":
                    cnt = self.CQ.text()
                    cnt = str(int(cnt)-1)
                    self.CQ.setText(cnt)
                if i == "J":
                    cnt = self.CJ.text()
                    cnt = str(int(cnt)-1)
                    self.CJ.setText(cnt)
                if i == "10":
                    cnt = self.C10.text()
                    cnt = str(int(cnt)-1)
                    self.C10.setText(cnt)
                if i == "9":
                    cnt = self.C9.text()
                    cnt = str(int(cnt)-1)
                    self.C9.setText(cnt)
                if i == "8":
                    cnt = self.C8.text()
                    cnt = str(int(cnt)-1)
                    self.C8.setText(cnt)
                if i == "7":
                    cnt = self.C7.text()
                    cnt = str(int(cnt)-1)
                    self.C7.setText(cnt)
                if i == "6":
                    cnt = self.C6.text()
                    cnt = str(int(cnt)-1)
                    self.C6.setText(cnt)
                if i == "5":
                    cnt = self.C5.text()
                    cnt = str(int(cnt)-1)
                    self.C5.setText(cnt)
                if i == "4":
                    cnt = self.C4.text()
                    cnt = str(int(cnt)-1)
                    self.C4.setText(cnt)
                if i == "3":
                    cnt = self.C3.text()
                    cnt = str(int(cnt)-1)
                    self.C3.setText(cnt)

    def printRecord(self, a, b, c):
        self.changeCnt(a+b+c)
        if a:
            str_a = ""
            for s in a:
                str_a = str_a + s + "/"
            self.listWidget.addItem("上:"+str_a)
        if b:
            str_b = ""
            for s in b:
                str_b = str_b + s + "/"
            self.listWidget.addItem("我:"+str_b)
        if c:
            str_c = ""
            for s in c:
                str_c = str_c + s + "/"
            self.listWidget.addItem("下:"+str_c)
        self.listWidget.scrollToBottom()
        

    def det_thread(self):
        self.listWidget.clear()
        self.CJOKER.setText("2")
        self.C2.setText("4")
        self.CA.setText("4")
        self.CK.setText("4")
        self.CQ.setText("4")
        self.CJ.setText("4")
        self.C10.setText("4")
        self.C9.setText("4")
        self.C8.setText("4")
        self.C7.setText("4")
        self.C6.setText("4")
        self.C5.setText("4")
        self.C4.setText("4")
        self.C3.setText("4")
        print("已重置")
        
        while True:
            # 截图
            image = print_screen(hwnd)
            # 检测
            pos, mask, cards = detect(opt, image, model)
            # 输出结果
            a,b,c = card.update(pos, mask, cards)
            # 修改
            self.printRecord(a,b,c)


if __name__ == "__main__":
    # 初始化
    hwnd_title = dict()
    card = Card()
    print_hwnd()
    hwnd=input()
    opt = Opt()
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)
    
    app = QApplication(sys.argv)
    w = GameWindow()
    w.show()
    sys.exit(app.exec())

                
