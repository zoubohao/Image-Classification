import cv2
import numpy as np
import torch.nn.functional as F
import torch
# import torch.nn as nn
# import torchvision.transforms as T
# import copy
# import PIL.Image as Image
#
# upS = nn.UpsamplingBilinear2d(scale_factor=2)
#
# image = np.array(Image.open("test.jpg").convert("RGB"))
# h,w = image.shape[0],image.shape[1]
# testImgTensor = torch.from_numpy(image).reshape([1,3,h,w]).float()
# ###
# upSampling = F.interpolate(testImgTensor[:,0,:,:].unsqueeze(1),size=[h * 2, w *2],mode="bicubic")
# cv2.imwrite("testTransUp.jpg",cv2.cvtColor(upSampling.view([h * 2,w * 2,1]).detach().cpu().numpy(),cv2.COLOR_BGR2RGB))
# ###
# conv = nn.Conv2d(3,3,3,1,1)(testImgTensor).reshape([h,w,3]).detach().cpu().numpy()
# cv2.imwrite("testTransConv.jpg",conv)
# ###
# resize = cv2.resize(image[:,:,0],(w * 2, h *2),interpolation=cv2.INTER_BITS)
# cv2.imwrite("testTranscv2Resize.jpg",resize)
# ###
# print(resize)
# print(upSampling.view([h * 2,w * 2,1]))

#p4_Up = torch.nn.ConvTranspose2d(32,32,3,2,padding=[1,1],output_padding=[1,1],groups=32)
#print(p4_Up(torch.randn(size=[5,32,17,17]).float()).shape)


#print(np.ones(shape=[5,7]) @ np.ones(shape=[7,3]))
# print(torch.cuda.get_device_name(device=torch.device("cuda:0")))
# print(torch.cuda.get_device_name(device=torch.device("cuda:1")))
# a = [0,1,2,3,4]
# aIter = iter(a)
# a_next = next(aIter)
# print(a_next)
#
# for b in aIter:
#     print(b)

from Tools import drop_connect_A
testInput = torch.ones(size=[5,8,32,32]).float()
print(drop_connect_A(testInput,0.5,True).sum())

