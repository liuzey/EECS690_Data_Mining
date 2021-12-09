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
from pytorch_grad_cam.utils.image import show_cam_on_image
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from visualizer import DeepFeatures


torch.cuda.empty_cache()
torch.random.manual_seed(100)
INDEX = 1
MODE = 'models'  # 'embeddings'
method = 'gradcam++'  # Can be gradcam/gradcam++/scorecam
dataset_name = 'kaggle'
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
valid_data_dir = './small_'+dataset_name+'_cropped'
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
        # print(combined_features.shape)
        output = self.cls_head(combined_features)
        if MODE == 'embeddings':
            return torch.hstack((1-output, output))
        else:
            return output


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


model = SiameseNetwork()
resnet = model.to(device)
resnet.load_state_dict(torch.load('./'+dataset_name+'_best_model_'+ACC+'.pth'), strict=True)

if MODE == 'embeddings':
    resnet.model.logits = Identity()

resnet.eval()

index = 0
for (img1, img2), y, (class1, class2) in val_loader:
    index += 1
    if index == INDEX:
        if MODE == 'embeddings':
            IMGS_FOLDER = './projector/imgs'
            EMBS_FOLDER = './projector/embeddings'
            TB_FOLDER = './projector/tensorboard_runs'

            DF = DeepFeatures(model=resnet, imgs_folder=IMGS_FOLDER,
                              embs_folder=EMBS_FOLDER, tensorboard_folder=TB_FOLDER,
                              experiment_name='Visual')

            DF.write_embeddings(x=img1.to(device))
            DF.create_tensorboard_log()
        else:
            img1 = img1.cuda()
            with SummaryWriter(comment='Net8')as w:
                w.add_graph(model, img1)
            w.close()
        break




