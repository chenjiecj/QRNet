import colorsys
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
import copy
from nets.QRNet import QRNet
from utils.utils import (cvtColor, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox


def reparameterize_model(model: torch.nn.Module):
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model

class Detection_Transformers(object):
    _defaults = {
        "model_path"        : 'model_data/last_epoch_weights.pth',
        "classes_path"      : 'model_data/voc_classes.txt',
        "input_shape"        : [640, 640],
        "min_length": 640,
        "confidence"        : 0.9,
        "backbone"          : 'qrnet',
        "cuda"              : True,
        "inference_mode"   : True,
    }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.bbox_util                      = DecodeBox()
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        
        show_config(**self._defaults)


    def generate(self, onnx=False):
        self.net    = QRNet(self.backbone,  self.num_classes, self.inference_mode)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        if self.inference_mode :
            self.net = reparameterize_model(self.net.eval())
        self.net    = self.net.eval()


        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image, crop = False, count = False):
        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)

        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images          = torch.from_numpy(image_data)
            images_shape    = torch.unsqueeze(torch.from_numpy(image_shape), 0)
            if self.cuda:
                images          = images.cuda()
                images_shape    = images_shape.cuda()

            outputs = self.net(images)

            results = self.bbox_util(outputs, images_shape, self.confidence)
                                                    
            if results[0] is None: 
                return image
            
            _results    = results[0].cpu().numpy()
            top_label   = np.array(_results[:, 5], dtype = 'int32')
            top_conf    = _results[:, 4]
            top_boxes   = _results[:, :4]

        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // self.min_length, 1))

        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
            #if top - label_size >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image


    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)
        image_data  = resize_image(image, self.min_length)
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images          = torch.from_numpy(image_data)
            images_shape    = torch.unsqueeze(torch.from_numpy(image_shape), 0)
            if self.cuda:
                images          = images.cuda()
                images_shape    = images_shape.cuda()

            outputs = self.net(images)
            results = self.bbox_util(outputs, images_shape, self.confidence)
            
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                images          = torch.from_numpy(image_data)
                images_shape    = torch.unsqueeze(torch.from_numpy(image_shape), 0)
                if self.cuda:
                    images          = images.cuda()
                    images_shape    = images_shape.cuda()
                
                outputs = self.net(images)
                results = self.bbox_util(outputs, images_shape, self.confidence)
                            
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)

        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images          = torch.from_numpy(image_data)
            images_shape    = torch.unsqueeze(torch.from_numpy(image_shape), 0)
            if self.cuda:
                images          = images.cuda()
                images_shape    = images_shape.cuda()
            outputs = self.net(images)
            results = self.bbox_util(outputs, images_shape, self.confidence)
                                                    
            if results[0] is None: 
                return 
            
            _results    = results[0].cpu().numpy()
            top_label   = np.array(_results[:, 5], dtype = 'int32')
            top_conf    = _results[:, 4]
            top_boxes   = _results[:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
