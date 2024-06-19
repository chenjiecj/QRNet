import random

import numpy as np
import torch
from PIL import Image
from torch import functional as F
import torchvision.transforms.functional as Fun
#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
    
def get_new_img_size(height, width, min_length=640):
    if width <= height:
        f = float(min_length) / width
        resized_height = int(f * height)
        resized_width = int(min_length)
    else:
        f = float(min_length) / height
        resized_width = int(f * width)
        resized_height = int(min_length)

    return resized_height, resized_width

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
# def resize_image(image, min_length):
#     iw, ih  = image.size
#     h, w    = get_new_img_size(ih, iw, min_length=min_length)
#     new_image = image.resize((w, h), Image.BICUBIC)
#     return new_image

def resize_image(image, size, letterbox_image=False):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

# def resize_image(image, size, max_size=None, target=None,):
#     # size can be min_size (scalar) or (w, h) tuple
#
#     def get_size_with_aspect_ratio(image_size, size, max_size=None):
#         w, h = image_size
#         if max_size is not None:
#             min_original_size = float(min((w, h)))
#             max_original_size = float(max((w, h)))
#             if max_original_size / min_original_size * size > max_size:
#                 size = int(round(max_size * min_original_size / max_original_size))
#
#         if (w <= h and w == size) or (h <= w and h == size):
#             return (h, w)
#
#         if w < h:
#             ow = size
#             oh = int(size * h / w)
#         else:
#             oh = size
#             ow = int(size * w / h)
#
#         # r = min(size / min(h, w), max_size / max(h, w))
#         # ow = int(w * r)
#         # oh = int(h * r)
#
#         return (oh, ow)
#
#     def get_size(image_size, size, max_size=None):
#         if isinstance(size, (list, tuple)):
#             return size[::-1]
#         else:
#             return get_size_with_aspect_ratio(image_size, size, max_size)
#
#     size = get_size(image.size, size, max_size)
#     rescaled_image = Fun.resize(image, size)
#     if target is None:
#         return rescaled_image
#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
