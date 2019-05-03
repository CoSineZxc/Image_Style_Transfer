from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

# imsize=512
# dtype = torch.cuda.FloatTensor
#
# loader = transforms.Compose([
#     transforms.Resize((imsize,imsize)), # scale imported image
#     transforms.ToTensor()               # transform it into a torch tensor
# ])
#
# def image_loader(image_name):
#     image = Image.open(image_name)
#     imgsize=image.size
#     image = Variable(loader(image))
#     # fake batch dimension required to fit network's input dimensions
#     image = image.unsqueeze(0)
#     return image,imgsize
#
# style_img,(imgwidth,imgheight) = image_loader("Img/style/style5.jpg")[0].type(dtype),image_loader("Img/style/style5.jpg")[1]
#
# unloader = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((imgheight,imgwidth))
# ])
#
# def image_unloader(image_name,image_tensor):
#     image = image_tensor.cpu().clone()
#     image = image.squeeze(0)
#     image = unloader(image)
#     image.save(image_name)

# image_unloader("Img/result/re1.jpg",style_img)
# imsize=256
# dtype = torch.cuda.FloatTensor
# loader = transforms.Compose([
#     transforms.Resize((imsize,imsize)),  # scale imported image
#     transforms.ToTensor()])  # transform it into a torch tensor
# def image_loader(image_name):
#     image = Image.open(image_name)
#     imgsize=image.size
#     image = Variable(loader(image))
#     # fake batch dimension required to fit network's input dimensions
#     image = image.unsqueeze(0)
#     return image,imgsize
# content_img = image_loader("Img/content/cont2.jpg")[0].type(dtype)
# class ContentLoss(nn.Module):
#
#     def __init__(self, target, weight):
#         super(ContentLoss, self).__init__()
#         # we 'detach' the target content from the tree used
#         self.target = target.detach() * weight
#         # to dynamically compute the gradient: this is a stated value,
#         # not a variable. Otherwise the forward method of the criterion
#         # will throw an error.
#         self.weight = weight
#         self.criterion = nn.MSELoss()
#
#     def forward(self, input):
#         self.loss = self.criterion(input * self.weight, self.target)
#         self.output = input
#         return self.output
#
#     def backward(self, retain_graph=True):
#         self.loss.backward(retain_graph=retain_graph)
#         return self.loss
#
# content_layers=['conv_1','conv_2','conv_3','conv_4']
# cnn = models.vgg19(pretrained=True).features.cuda()
# content_losses=[]
# model=nn.Sequential()
# i=1
# for layer in list(cnn):
#     if isinstance(layer,nn.Conv2d):
#         name="conv_"+str(i)
#         model.add_module(name,layer)
#         if name in content_layers:
#             target=model(content_img).clone()
#             content_loss=ContentLoss(target,1)
#             model.add_module("content_loss_"+str(i),content_loss)
#             content_losses.append(content_loss)
#     if isinstance(layer,nn.ReLU):
#         name="relu_"+str(i)
#         model.add_module(name,layer)
#         if name in content_layers:
#             target=model(content_img)
#             content_loss=ContentLoss(target,1)
#             model.add_module("content_loss_"+str(i),content_loss)
#             content_losses.append(content_loss)
#         i+=1
#     if isinstance(layer,nn.MaxPool2d):
#         name="pool_"+str(i)
#         model.add_module(name,layer)

# print(cnn)
# print(model)
# print(content_losses)

Style_imgdir="Img/result/rslt.jpg"
stl=Style_imgdir.split('/')[-1]
stl=stl.split('.')[0]
print(stl)