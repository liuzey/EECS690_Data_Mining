import numpy as np
import os
import copy
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from PIL import Image

import training
from data import Dataset
# from pytorch_grad_cam import CAM
from gradcam_01 import CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization


torch.cuda.empty_cache()
torch.random.manual_seed(100)
INDEX = 10
method = 'gradcam++'  # Can be gradcam/gradcam++/scorecam
dataset_name = 'LFW'
if dataset_name == 'LFW':
    ACC = '9991'
elif dataset_name == 'kaggle':
    ACC = 'saved_78'
elif dataset_name == 'realworld':
    ACC = '98'
else:
    raise NotImplementedError
# dataset_name = 'kaggle'
# dataset_name = 'realworld'
BATCH_SIZE = 4
valid_data_dir = './'+dataset_name+'_cropped'
val_dataset = Dataset(dataset_name, valid_data_dir, shuffle_pairs=(dataset_name=='realworld'), augment=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
# data_dir = './kaggle/img'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
transform = transforms.Compose([
                transforms.Resize((512, 512)),
                np.float32,
                transforms.ToTensor(),
                fixed_image_standardization
            ])
if dataset_name=='kaggle' or dataset_name=='realworld':
    standard = Image.open('./GRADCAM/'+dataset_name+'/standard.png').convert("RGB")
else:
    standard = Image.open('./GRADCAM/'+dataset_name+'/standard.jpg').convert("RGB")
STAN = transform(standard).float().to(device).unsqueeze(0)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=8631)
        self.standard = STAN

        self.cls_head = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(8631, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, img1):
        feat2 = self.model(self.standard)
        feat1 = self.model(img1)
        combined_features = feat1 * feat2
        output = self.cls_head(combined_features)
        return output

    def get_heatmap(self, _input, _label, cmap=get_cmap('jet')):
        from trojanvision.utils import apply_cmap

        _label = 0
        squeeze_flag = False
        if _input.dim() == 3:
            _input = _input.unsqueeze(0)    # (N, C, H, W)
            squeeze_flag = True
        if isinstance(_label, int):
            _label = [_label] * len(_input)
        _label = torch.as_tensor(_label, device=_input.device)
        heatmap = _input
        _input.requires_grad_()

        _output = self(_input).gather(dim=1, index=_label.unsqueeze(1)).sum()

        grad = torch.autograd.grad(_output, _input)[0]  # (N,C,H,W)
        zero = torch.zeros_like(grad)
        grad = torch.where(grad < 0, zero, grad)
        _input.requires_grad_(False)

        heatmap = grad.abs().max(dim=1)[0]  # (N,H,W)
        heatmap.sub_(heatmap.min(dim=-2, keepdim=True)[0].min(dim=-1, keepdim=True)[0])
        heatmap.div_(heatmap.max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0])
        heatmap = apply_cmap(heatmap.detach().cpu(), cmap)

        return heatmap[0] if squeeze_flag else heatmap


def gradcam(t_label, t_img, i, target_layer, index):
    index = (index - 1) * BATCH_SIZE + i
    label = 0
    input_tensor = t_img[i, :, :, :].view(1, 3, 512, 512).to(device)
    input_image = (input_tensor.clone().cpu() * 128 + 127.5)/255
    input_image = input_image.numpy()
    rgb_img = np.ones((512, 512, 3))
    rgb_img[:, :, 0] = input_image[0, 2, :, :]
    rgb_img[:, :, 1] = input_image[0, 1, :, :]
    rgb_img[:, :, 2] = input_image[0, 0, :, :]
    cam = CAM(model=model, target_layer=target_layer, use_cuda=True)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=label, method=method)
    visualization = show_cam_on_image(rgb_img, grayscale_cam)
    # print(visualization.shape)
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = np.uint8(255 * heatmap)
    cv2.imwrite('./GRADCAM/'+dataset_name+'/original' + str(index) + '.png',
                np.uint8(255 * rgb_img), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.imwrite('./GRADCAM/'+dataset_name+'/mix' + str(index) + '.png',
                visualization, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.imwrite('./GRADCAM/'+dataset_name+'/heatmap' + str(index) + '.png',
                heatmap, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


model = SiameseNetwork()
resnet = model.to(device)
resnet.eval()
resnet.load_state_dict(torch.load('./'+dataset_name+'_best_model_'+ACC+'.pth'), strict=True)

# target_layer = resnet.feature3[-5]
target_layer = resnet.model.conv2d_1a

index = 0
for (img1, img2), y, (class1, class2) in val_loader:
    index += 1
    if index < INDEX:
        for i in range(BATCH_SIZE):
            gradcam(y, img1, i, target_layer, index)
            # gradcam(y, img2, i, target_layer)
    else:
        break