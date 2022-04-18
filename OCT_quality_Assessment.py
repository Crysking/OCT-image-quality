# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 09:12:10 2022

@author: crysking Wang
"""
from PIL import Image
import torch 
import torch.nn as nn
from torchvision import models,transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np

device = torch.device("cpu") # if CUDA is not available
#-------------load model------------------------------------------------
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs,4)
model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft.load_state_dict(torch.load('./Trained_model/quality_fold_3_resnet.pkl'))
model_ft.eval()
transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


#--------------load test image-------------------------------------------
image_path = 'where is your image'
img = Image.open(image_path).convert('RGB')
img_pre = transform(img)
batch_t = torch.unsqueeze(img_pre,0).to(device)

#-------------output-----------------------------------------------------
output = model_ft(batch_t)
probs = output.data.cpu().numpy()
class_idx = np.argmax(probs,axis=1)
if class_idx==0:
    quality='Good'

elif class_idx==1:
    quality='Off-center'
    
elif class_idx==2:
    quality='Other'

elif class_idx==3:
    quality='Signal-Shield'

print(quality)
