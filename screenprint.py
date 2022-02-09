from PyQt5.QtWidgets import QApplication
import sys
import win32gui
import time
import numpy as np

hwnd_title = dict()

def get_all_hwnd(hwnd, _):
  if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
    hwnd_title.update({hwnd:win32gui.GetWindowText(hwnd)})

def print_hwnd():
    win32gui.EnumWindows(get_all_hwnd, 0)
    for h,t in hwnd_title.items():
        if t is not "":
            print(h, t)

def qtpixmap_to_cvimg(qtpixmap):

    qimg = qtpixmap.toImage()
    temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
    temp_shape += (4,)
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
    result = result[..., :3]

    return result

def print_screen(hwnd):
    # app=QApplication(sys.argv)
    screen=QApplication.primaryScreen()
    pix=screen.grabWindow(int(hwnd))
    image = qtpixmap_to_cvimg(pix)
    return image