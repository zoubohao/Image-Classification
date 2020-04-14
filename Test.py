import cv2
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.transforms as T
import copy
import PIL.Image as Image

upS = nn.UpsamplingBilinear2d(scale_factor=2)

image = np.array(Image.open("test.jpg").convert("RGB"))
h,w = image.shape[0],image.shape[1]
testImgTensor = torch.from_numpy(image).reshape([1,3,h,w]).float()
###
upSampling = F.interpolate(testImgTensor[:,0,:,:].unsqueeze(1),size=[h * 2, w *2],mode="bicubic")
cv2.imwrite("testTransUp.jpg",cv2.cvtColor(upSampling.view([h * 2,w * 2,1]).detach().cpu().numpy(),cv2.COLOR_BGR2RGB))
###
conv = nn.Conv2d(3,3,3,1,1)(testImgTensor).reshape([h,w,3]).detach().cpu().numpy()
cv2.imwrite("testTransConv.jpg",conv)
###
resize = cv2.resize(image[:,:,0],(w * 2, h *2),interpolation=cv2.INTER_BITS)
cv2.imwrite("testTranscv2Resize.jpg",resize)
###
print(resize)
print(upSampling.view([h * 2,w * 2,1]))





