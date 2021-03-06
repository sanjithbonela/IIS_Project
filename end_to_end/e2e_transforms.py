import random
import numpy as np

from PIL import Image
import imutils
from math import *

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

class E2E_Transforms():
    def __init__(self):
        pass

    def rotate(self, image, angle):
        angle = random.uniform(-angle, +angle)

        transformation_matrix = torch.tensor([
            [+cos(radians(angle)), -sin(radians(angle))],
            [+sin(radians(angle)), +cos(radians(angle))]
        ])

        image = imutils.rotate(np.array(image), angle)

        return Image.fromarray(image)

    '''
    def resize(self, image, landmarks, img_size):
        image = TF.resize(image, img_size)
        return image, landmarks
    '''

    def color_jitter(self, image):
        color_jitter = transforms.ColorJitter(brightness=0.3,
                                              contrast=0.3,
                                              saturation=0.3,
                                              hue=0.1)
        image = color_jitter(image)
        return image

    '''
    def crop_face(self, image, landmarks, crops):
        left = int(crops['left'])
        top = int(crops['top'])
        width = int(crops['width'])
        height = int(crops['height'])

        image = TF.crop(image, top, left, height, width)

        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks
    '''

    def __call__(self, image):
        image = Image.fromarray(image)
        #image, landmarks = self.crop_face(image, landmarks, crops)
        #image, landmarks = self.resize(image, landmarks, (224, 224))
        image = self.color_jitter(image)
        image= self.rotate(image, angle=10)
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        return image