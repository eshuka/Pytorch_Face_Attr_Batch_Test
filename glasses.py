# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import torch.nn as nn
from network import Resnet_low_param, Resnet_low_param_
from torch.nn import functional as F
from torchvision import transforms as trn
from torch.autograd import Variable as V
from PIL import Image
import time
import os
import torchvision
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

centre_crop = trn.Compose([
    trn.Resize((160, 160)),
    trn.CenterCrop(160),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def test_model(glasses_model, image_path):

    glasses_model.eval()
    g_classes = ['none', 'Eyeglasses']

    image_list = os.listdir(image_path)
    image_list.sort()

    for image in image_list:
        img_path = os.path.join(image_path, image)
        img = Image.open(img_path)
        start_time = time.time()
        g_input_image = V(centre_crop(img).unsqueeze(0)).to(device)
        print(np.shape(g_input_image))
        g_outputs = glasses_model.forward(g_input_image)
        end_time = time.time()
        g_h_x = F.softmax(g_outputs, 1).data.squeeze()
        g_probs, g_idx = g_h_x.sort(0, True)

        print('label: ', g_classes[g_idx[0]])
        print('prob: ', g_probs[0])
        print('time: ', end_time - start_time)
        print()


def batch_test_model(glasses_model, image_path, batch_size):

    glasses_model.eval()
    g_classes = ['none', 'Eyeglasses']

    image_list = os.listdir(image_path)
    image_list.sort()

    img_list = []
    for im in image_list:
        img_path = os.path.join(image_path, im)
        img = Image.open(img_path)
        g_input_image = V(centre_crop(img).unsqueeze(0))
        img_list.extend(g_input_image)

    testloader = torch.utils.data.DataLoader(img_list, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    for images in testloader:
        images = images.to(device)

        start_time = time.time()

        g_outputs = glasses_model.forward(images)
        g_h_x = F.softmax(g_outputs, 1).data.squeeze()

        end_time = time.time()
        print('time: ', end_time - start_time)

        for result in g_h_x:
            g_probs, g_idx = result.sort(0, True)
            print('label: ', g_classes[g_idx[0]])
            print('prob: ', g_probs[0])

            print()

if __name__ == "__main__":

    glasses_model_file = 'glasses_7.pth'
    image_path = './images'
    batch_size = 10

    glasses_model_ft = Resnet_low_param.resnet34()
    num_ftrs = glasses_model_ft.fc.in_features
    glasses_model_ft.fc = nn.Linear(num_ftrs, 2)
    glasses_weight = torch.load(glasses_model_file)
    glasses_model_ft.load_state_dict(glasses_weight)
    glasses_model_ft = glasses_model_ft.to(device)
    print('glasses model done')

    # test_model(glasses_model_ft, image_path)
    batch_test_model(glasses_model_ft, image_path, batch_size)
