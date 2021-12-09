from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import training
from data import Dataset
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=8631)

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

    def forward(self, img1, img2):
        feat1 = self.model(img1)
        feat2 = self.model(img2)
        combined_features = feat1 * feat2
        output = self.cls_head(combined_features)
        return output


data_dir = './observations-master/experiements/data'
dataset_name = 'realworld'
VALID_ONLY = True
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
train_data_dir = './small_'+dataset_name+'_cropped'
valid_data_dir = './'+dataset_name+'_cropped'
# data_dir = './kaggle/img'

CROP = False
INTERVAL = 5
LIMIT = 2000
LEARNING_RATE = 1e-4
DECAY_LIST = [1000]
batch_size = 8
epochs = 100
workers = 0 if os.name == 'nt' else 8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

if CROP:
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
    dataset.samples = [
        (p, p.replace(data_dir, data_dir + '_cropped'))
        for p, _ in dataset.samples
    ]

    loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil
    )

    for i, (x, y) in enumerate(loader):
        mtcnn(x, save_path=y)
        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')

    # Remove mtcnn to reduce GPU memory usage
    del mtcnn
    exit()

model = SiameseNetwork()
resnet = model.to(device)

optimizer = optim.Adam(resnet.parameters(), lr=LEARNING_RATE)
scheduler = MultiStepLR(optimizer, DECAY_LIST)

train_dataset = Dataset(dataset_name, train_data_dir, shuffle_pairs=True, augment=True)
val_dataset = Dataset(dataset_name, valid_data_dir, shuffle_pairs=(dataset_name=='realworld'), augment=False)
# train_dataset = datasets.ImageFolder(train_data_dir, transform=train_trans)
# valid_dataset = datasets.ImageFolder(valid_data_dir, transform=train_trans)
# img_inds = np.arange(len(val_dataset))
# np.random.shuffle(img_inds)
# inds = img_inds[:min((4*LIMIT, len(val_dataset)))]
# val_inds = img_inds[int(0.8 * len(img_inds)):]

train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=4)
print(len(train_loader))
print(len(val_loader))


loss_fn = torch.nn.BCELoss()
metrics = {
    'fps': training.BatchTimer(),
    'TP': training.TP,
    'TP_and_FP': training.TP_and_FP,
    'TP_and_FN': training.TP_and_FN,
    'acc': training.accuracy
}

writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10
'''
print('\n\nInitial')
print('-' * 10)
resnet.eval()
training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)
'''

best_acc = 0.00

if not VALID_ONLY:
    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        resnet.train()
        training.pass_epoch(
            resnet, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

        if epoch % INTERVAL == 0:
            if epoch == 0:
                pass
            resnet.eval()
            _, metrics_ = training.pass_epoch(
                resnet, loss_fn, val_loader,
                batch_metrics=metrics, show_running=True, device=device,
                writer=writer
            )
            acc = metrics_['acc'].numpy()
            if acc > best_acc:
                best_acc = acc
                torch.save(resnet.state_dict(), './'+dataset_name+'_best_model.pth')
else:
    resnet.load_state_dict(torch.load('./' + dataset_name + '_best_model_' + ACC + '.pth'),
                           strict=True)  # load pre-trained model

resnet.eval()
training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
)

writer.close()