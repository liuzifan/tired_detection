import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os

# load model
model_path = './epochs/epoch_20.pth' 
net = models.resnet18(pretrained=False) 
net.avgpool =nn.AdaptiveAvgPool2d((1, 1))
net.fc = nn.Linear(in_features=512, out_features=2, bias=True)
net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
net = net.cuda()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor() ])

class_list = ['close', 'open']

with torch.no_grad():
    net.eval()

    # load testing data    
    picfold = './test_data/'
    piclist = os.listdir(picfold)
    for pic in piclist:
        picdir = picfold + pic
        image = Image.open(picdir).convert('RGB')
        image = transform(image)
        image = image.reshape(1, *image.shape)

        image = image.cuda()
        out = net(image)

        pre_label_onehot = torch.softmax(out, 1)
        topk_values, topk_indices = pre_label_onehot.topk(1, dim=1)
        topk_indices = topk_indices[0]
        topk_values = topk_values[0]

        # print results
        for i, item in enumerate(topk_indices):
            p = topk_values[i]
            class_name = class_list[item%2]
            print(picdir)
            print(class_name + ' : ' + str(p.item()))
            print('\n')

