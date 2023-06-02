import mss
import torch
from torchvision import transforms
from torchvision import utils as Tutils
from utils import check_make_folder
import os
import threading
from PIL import Image

def range_dim(big, small):
    ratio = small/big #ratio*n
    x0 = (1-ratio)/2
    return ratio, x0

def get_monitor(submonitor):
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        width_fll = monitor['width']
        height_fll = monitor['height']
    w_dif = width_fll - submonitor['width']
    h_dif = height_fll - submonitor['height']
    return {
        'left': w_dif//2,
        'top': h_dif//2,
        'width': submonitor['width'],
        'height': submonitor['height'],
    }, monitor

def save_tensor_image(tensor, name):
    #tensor in range [-1.0,1.0]
    sample_path = os.path.join('./samples', name + '.png')
    itensor = tensor.detach()/2 + 0.5
    Tutils.save_image(itensor, sample_path)

class Screenshot(object):
    def __init__(self, submonitor, ch):
        self.ch = ch
        self.monitor,_ = get_monitor(submonitor)
        self.mss = mss.mss()
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Resize((self.h, self.w), Image.BICUBIC)
        ])
        
    def pre_sample(self):
        pass

    def take(self, save=False, output = 'Screenshot.png'):
        with self.mss as sct:
            im = sct.grab(self.monitor)
            if save:
                mss.tools.to_png(im.rgb, im.size, output = output)
            self.im = im
            
    def get_pytorch_img(self, device):
        img = Image.frombytes("RGB", self.im.size, self.im.bgra, "raw", "BGRX")
        img = self.transforms(img)
        img = img.to(device=device) * 2 - 1
        img.unsqueeze_(dim=0)
        return img