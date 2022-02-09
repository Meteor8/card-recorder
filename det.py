from utils.torch_utils import time_synchronized
import torch

from models.experimental import attempt_load
from utils.dataload import  myloadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device

class Opt:
   def __init__(self):
        # self.source = source
        self.weights = r".\weights\best.pt"
        # self.image = image

        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = '0'
        self.view_img = False
        self.save_txt = False
        self.save_conf = False
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.update = False
        
        self.project = 'runs/detect'
        self.name = 'exp'

        self.exist_ok = True


def detect(opt, image, mdl):
    imgsz = opt.img_size

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    model = mdl
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    img, im0s = myloadImages(image)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # Result
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        im0= im0s

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # 删除重复
            center_list=[]
            card_list=[]
            wrong_mask=[0]*len(det)
            for i, (*xyxy, conf, cls) in enumerate(det):
                xc, yc = (int(xyxy[0])+(int(xyxy[2]) - int(xyxy[0]))//2, int(xyxy[1])+(int(xyxy[3]) - int(xyxy[1]))//2)
                for j, c in enumerate(center_list):
                    dis = ((xc-c[0])**2+(yc-c[1])**2)**0.5
                    if dis < 10:
                        # 比较置信度
                        cf1 = round(float(conf),2)
                        cf2 = round(float(det[j][-2]),2)
                        if cf1 < cf2:
                            wrong_mask[i] = 1
                        else:
                            wrong_mask[j] = 1
                        break
                center_list.append((xc,yc))
                card_list.append(names[int(cls)])

            # 中间多检测
            for i, c1 in enumerate(center_list):
                if(wrong_mask[i]==0):
                    cnt=0
                    for j, c2 in enumerate(center_list):
                        if(wrong_mask[j]==0) and i!=j:
                            # 阈值
                            if abs(c1[0]-c2[0])<20 and abs(c1[1]-c2[1])<60:
                                cnt+=1
                            if cnt>=2:
                                wrong_mask[i] = 2
                                break

            return center_list,wrong_mask,card_list
