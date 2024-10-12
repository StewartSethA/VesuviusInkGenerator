import os.path as osp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import random
import yaml

import numpy as np
import pandas as pd

import wandb

from torch.utils.data import DataLoader

import pandas as pd
import os
import random
from contextlib import contextmanager
import cv2

import scipy as sp
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from tap import Tap
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW

import datetime
import segmentation_models_pytorch as smp
import numpy as np
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from i3dallnl import InceptionI3d
import torch.nn as nn
import torch
from warmup_scheduler import GradualWarmupScheduler
from scipy import ndimage
import time
import json
import numba
from numba import jit

import scipy.stats as st

class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    # comp_dir_path = './'
    comp_dir_path = './' #'/content/gdrive/MyDrive/vesuvius_model/training'
    comp_folder_name = './' #'/content/gdrive/MyDrive/vesuvius_model/training'
    # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    comp_dataset_path = './' #f'/content/gdrive/MyDrive/vesuvius_model/training'
    
    exp_name = 'pretraining_all'

    # ============== pred target =============
    target_size = 1

    # ============== model cfg =============
    model_name = 'Unet'
    # backbone = 'efficientnet-b0'
    # backbone = 'se_resnext50_32x4d'
    backbone='resnet3d'
    in_chans = 30 # 65
    encoder_depth=5
    # ============== training cfg =============
    size = 64
    tile_size = 256
    stride = 64 #tile_size // 8

    train_batch_size = 16 # 32
    valid_batch_size = train_batch_size
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    # scheduler = 'CosineAnnealingLR'
    epochs = 30 # 30

    # adamW warmupあり
    warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    # lr = 1e-4 / warmup_factor
    lr = 2e-5
    # ============== fold =============
    valid_id = '20230820203112'

    # objective_cv = 'binary'  # 'binary', 'multiclass', 'regression'
    metric_direction = 'maximize'  # maximize, 'minimize'
    # metrics = 'dice_coef'

    # ============== fixed =============
    pretrained = True
    inf_weight = 'best'  # 'best'

    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 100

    print_freq = 50
    num_workers = 10

    seed = 0

    # ============== set dataset path =============
    print('set dataset path')

    outputs_path = f'./outputs' #/content/gdrive/MyDrive/vesuvius_model/training/outputs'

    submission_dir = outputs_path + 'submissions/'
    submission_path = submission_dir + f'submission_{exp_name}.csv'

    model_dir = outputs_path + \
        f'{comp_name}-models/'

    figures_dir = outputs_path + 'figures/'

    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'

    # ============== augmentation =============
    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.6),

        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0.1,p=0.75),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=2, max_width=int(size * 0.2), max_height=int(size * 0.2), 
                        mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]
    rotate = A.Compose([A.Rotate(8,p=1)])
def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False
def make_dirs(cfg):
    for dir in [cfg.model_dir, cfg.figures_dir, cfg.submission_dir, cfg.log_dir]:
        os.makedirs(dir, exist_ok=True)
def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
    # set_env_name()
    # set_dataset_path(cfg)

    if mode == 'train':
        make_dirs(cfg)
cfg_init(CFG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_image_mask(fragment_id,start_idx=15,end_idx=45,rotation=0):

    images = []

    # idxs = range(65)
    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start_idx, end_idx)


    t = time.time()
    if os.path.isfile(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{fragment_id}.npy"):
      images = np.load(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{fragment_id}.npy")
      pad0 = (CFG.tile_size - images.shape[0] % CFG.tile_size)
      pad1 = (CFG.tile_size - images.shape[1] % CFG.tile_size)
      print(time.time()-t, "seconds taken to load images from", CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{fragment_id}.npy")
    else:
      for i in idxs:
        
        image = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.tif", 0)

        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        # image = ndimage.median_filter(image, size=5) # TODO: Why median filtering?
        
        # image = cv2.resize(image, (image.shape[1]//2,image.shape[0]//2), interpolation = cv2.INTER_AREA)
        if 'frag' in fragment_id:
            image = cv2.resize(image, (image.shape[1]//2,image.shape[0]//2), interpolation = cv2.INTER_AREA)
        image=np.clip(image,0,200)
        if fragment_id=='20230827161846':
            image=cv2.flip(image,0)
        images.append(image)
      print(time.time()-t, "seconds taken to load images.")
      images = np.stack(images, axis=2)
      t = time.time()
      print(time.time()-t, "seconds taken to stack images.")
      t = time.time()
      np.save(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{fragment_id}.npy", images)
      print(time.time()-t, "seconds taken to save images as npy.")
    if fragment_id in ['20230701020044','verso','20230901184804','20230901234823','20230531193658','20231007101615','20231005123333','20231011144857','20230522215721', '20230919113918', '20230625171244','20231022170900','20231012173610','20231016151000']:

        images=images[:,:,::-1]

    if fragment_id=='20231022170900':
        mask = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.tiff", 0)
    else:
        mask = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.png", 0)

    # mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)
    fragment_mask=cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id}_mask.png", 0)
    if fragment_id=='20230827161846':
        fragment_mask=cv2.flip(fragment_mask,0)

    fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)

    kernel = np.ones((16,16),np.uint8)
    if 'frag' in fragment_id:
        fragment_mask = cv2.resize(fragment_mask, (fragment_mask.shape[1]//2,fragment_mask.shape[0]//2), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(mask , (mask.shape[1]//2,mask.shape[0]//2), interpolation = cv2.INTER_AREA)

    print("images.shape,dtype", images.shape, images.dtype, "mask", ((mask.shape, mask.dtype) if mask is not None else None), "fragment_mask", fragment_mask.shape, fragment_mask.dtype)
    #print("images.shape,dtype", images.shape, images.dtype, "mask", mask.shape, mask.dtype, "fragment_mask", fragment_mask.shape, fragment_mask.dtype)
    return images, mask,fragment_mask

def generate_xyxys_ids(fragment_id, image, mask, fragment_mask, tile_size, size, stride, is_valid=True):
        is_valid = True
        xyxys = []
        ids = []
        x1_list = list(range(0, image.shape[1]-tile_size+1, stride))
        y1_list = list(range(0, image.shape[0]-tile_size+1, stride))
        #x1_list = list(range(0, image.size()[1]-tile_size+1, stride))
        #y1_list = list(range(0, image.size()[0]-tile_size+1, stride))
        #windows_dict={}
        for a in y1_list:
            for b in x1_list:
                for yi in range(0,tile_size,size):
                    for xi in range(0,tile_size,size):
                        y1=a+yi
                        x1=b+xi
                        y2=y1+size
                        x2=x1+size
                        if False and not is_valid:
                            if not np.all(np.less(mask[a:a + tile_size, b:b + tile_size],0.01)):
                                if not np.any(np.equal(fragment_mask[a:a+ tile_size, b:b + tile_size],0)):
                                    # if (y1,y2,x1,x2) not in windows_dict:
                                    #train_images.append(image[y1:y2, x1:x2])
                                    xyxys.append([x1,y1,x2,y2])
                                    ids.append(fragment_id)
                                    #train_masks.append(mask[y1:y2, x1:x2, None])
                                    #assert image[y1:y2, x1:x2].shape==(size,size,in_chans)
                                        # windows_dict[(y1,y2,x1,x2)]='1'
                        else:
                            if not np.any(np.equal(fragment_mask[a:a + tile_size, b:b + tile_size], 0)): # SethS workaround
                                    #valid_images.append(image[y1:y2, x1:x2])
                                    #valid_masks.append(mask[y1:y2, x1:x2, None])
                                    #print("Appending xyxy", [x1,y1,x2,y2])
                                    ids.append(fragment_id)
                                    xyxys.append([x1, y1, x2, y2])
                                    #assert image[y1:y2, x1:x2].shape==(size,size,in_chans)
        return xyxys, ids


def get_xyxys(fragment_ids, is_valid=True):
    xyxys = []
    ids = []
    images = {}
    masks = {}
    for fragment_id in fragment_ids:
        #start_idx = len(fragment_ids)
        print('reading ',fragment_id)
        image, mask,fragment_mask = read_image_mask(fragment_id)

        images[fragment_id] = image
        if mask is not None:
          masks[fragment_id] = mask[:,:,None]
        t = time.time()
        train_ids = set(['20230702185753','20230929220926','20231005123336','20231007101619','20231012184423','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']) - set([CFG.valid_id])
        if fragment_id not in train_ids: #20230702185753
          print("Loading crops for", fragment_id)
          if os.path.isfile(fragment_id + "valid.ids.json"):
            with open(fragment_id + ".valid.ids.json", 'r') as f:
              id = json.load(f)
          if os.path.isfile(fragment_id + ".valid.xyxys.json"):
            with open(fragment_id + ".valid.xyxys.json", 'r') as f:
              xyxy = json.load(f)
        else:
          print("REGENERATING", is_valid)
          xyxy, id = generate_xyxys_ids(fragment_id, image, mask, fragment_mask, CFG.tile_size, CFG.size, CFG.stride, is_valid)
          with open(fragment_id + ".valid.ids.json", 'w') as f:
            #if fragment_id != CFG.valid_id:
              json.dump(id, f) #[start_idx:], f)
            #else:
            #  json.dump(valid_ids, f)
          with open(fragment_id + ".valid.xyxys.json", 'w') as f:
            #if fragment_id != CFG.valid_id:
            json.dump(xyxy, f) #[start_idx:],f)
            #else:
            #  json.dump(valid_xyxys, f)
        xyxys = xyxys + xyxy
        ids = ids + [fragment_id,] * len(xyxy) #id

        print(time.time()-t, "seconds taken to generate crops for fragment", fragment_id)
    return images, masks, xyxys, ids

#@jit(nopython=True)
def get_train_valid_dataset():
    train_images = {}
    train_masks = {}
    train_xyxys= []
    train_ids = []
    valid_images = {}
    valid_masks = {}
    valid_xyxys = []
    valid_ids = []
    train_ids = set(['20230702185753','20230929220926','20231005123336','20231007101619','20231012184423','20231016151002','20231022170901','20231031143852','20231106155351','20231210121321','20231221180251','20230820203112']) - set([CFG.valid_id])
    valid_ids = set([CFG.valid_id])
    train_images, train_masks, train_xyxys, train_ids = get_xyxys(train_ids, False)
    valid_images, valid_masks, valid_xyxys, valid_ids = get_xyxys(valid_ids, True)
    return train_images, train_masks, train_xyxys, train_ids, valid_images, valid_masks, valid_xyxys, valid_ids

def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    return aug

class CustomDataset(Dataset):
    def __init__(self, images, cfg, xyxys=None, labels=None, ids=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.xyxys=xyxys
        self.ids = ids
        self.rotate=CFG.rotate
    def __len__(self):
        return len(self.xyxys)
    def cubeTranslate(self,y):
        x=np.random.uniform(0,1,4).reshape(2,2)
        x[x<.4]=0
        x[x>.633]=2
        x[(x>.4)&(x<.633)]=1
        mask=cv2.resize(x, (x.shape[1]*64,x.shape[0]*64), interpolation = cv2.INTER_AREA)
        x=np.zeros((self.cfg.size,self.cfg.size,self.cfg.in_chans)).astype(np.uint8)
        for i in range(3):
            x=np.where(np.repeat((mask==0).reshape(self.cfg.size,self.cfg.size,1), self.cfg.in_chans, axis=2),y[:,:,i:self.cfg.in_chans+i],x)
        return x
    def fourth_augment(self,image):
        image_tmp = np.zeros_like(image)
        cropping_num = random.randint(24, 30)

        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

        tmp = np.arange(start_paste_idx, cropping_num)
        np.random.shuffle(tmp)

        cutout_idx = random.randint(0, 2)
        temporal_random_cutout_idx = tmp[:cutout_idx]

        image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[..., crop_indices]

        if random.random() > 0.4:
            image_tmp[..., temporal_random_cutout_idx] = 0
        image = image_tmp
        return image

    def __getitem__(self, idx):
        if self.xyxys is not None:
            id = self.ids[idx]
            x1,y1,x2,y2=xy=self.xyxys[idx]
            image = self.images[id][y1:y2,x1:x2] #,self.start:self.end] #[idx]
            label = self.labels[id][y1:y2,x1:x2]
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate((label/255).unsqueeze(0).float(),(self.cfg.size//4,self.cfg.size//4)).squeeze(0)
            return image, label,xy
        else:
            image = self.images[idx]
            label = self.labels[idx]
            #3d rotate
            image=image.transpose(2,1,0)#(c,w,h)
            image=self.rotate(image=image)['image']
            image=image.transpose(0,2,1)#(c,h,w)
            image=self.rotate(image=image)['image']
            image=image.transpose(0,2,1)#(c,w,h)
            image=image.transpose(2,1,0)#(h,w,c)

            image=self.fourth_augment(image)
            
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate((label/255).unsqueeze(0).float(),(self.cfg.size//4,self.cfg.size//4)).squeeze(0)
            return image, label
class CustomDatasetTest(Dataset):
    def __init__(self, images, xyxys, ids, cfg, transform=None):
        self.images = images
        self.xyxys=xyxys
        self.ids = ids
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        # return len(self.df)
        return len(self.xyxys)

    def __getitem__(self, idx):
        x1,y1,x2,y2=xy=self.xyxys[idx]
        id = self.ids[idx]
        image = self.images[id][y1:y2,x1:x2]
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)

        return image,xy

# from resnetall import generate_model
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m, mode='fan_out', nonlinearity='relu')
class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask

class RegressionPLModel(pl.LightningModule):
    def __init__(self,pred_shape,size=256,enc='',with_norm=False):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)

        self.backbone=InceptionI3d(in_channels=1,num_classes=512,non_local=True)        
        self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,20,256,256))], upscale=1)

        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)
            
    def forward(self, x):
        if x.ndim==4:
            x=x[:,None]
        if self.hparams.with_norm:
            x=self.normalization(x)
        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        
        return pred_mask
    
    def training_step(self, batch, batch_idx):
        x, y, xys = batch
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        if torch.isnan(loss1):
            print("Loss nan encountered")
        self.log("train/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}

    def validation_step(self, batch, batch_idx):
        x,y,xyxys= batch
        batch_size = x.size(0)
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy()
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}
    
    def on_validation_epoch_end(self):
        self.mask_pred = np.divide(self.mask_pred, self.mask_count, out=np.zeros_like(self.mask_pred), where=self.mask_count!=0)
        wandb_logger.log_image(key="masks", images=[np.clip(self.mask_pred,0,1)], caption=["probs"])

        #reset mask
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=CFG.lr)
    
        scheduler = get_scheduler(CFG, optimizer)
        return [optimizer],[scheduler]



class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 10, eta_min=1e-6)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)
   

def predict_fn(test_loader, model, device, test_xyxys,pred_shape):
    mask_pred = np.zeros(pred_shape) #, dtype=np.uint8)
    rand_pred = np.zeros(pred_shape) #, dtype=np.uint8)
    mask_count = np.ones(pred_shape) #, dtype=np.int32) #np.zeros(pred_shape)
    model.eval()
    kernel=gkern(CFG.size,1)
    kernel=kernel/kernel.max()
    kernel = kernel.astype(np.float16)
    global activations
    activations = None # []
    
    def hook_fn(module, inp, output):
      global activations
      activations = output #activations.append(output) # OR just store the one activation.
    layer = model.decoder.up #[-1]
    handle = layer.register_forward_hook(hook_fn)
    
    def norm(img):
      img = (img - img.min()) / (img.max() - img.min())
      return img
    
    def imshow(title, img):
      #img = img.detach().cpu().numpy()
      img = 255 * norm(img)
      print("img", img.shape, img.dtype)
      for i in range(img.shape[1]): # Batch is 256
        #cv2.imshow(title, img[0,0,i,:,:].astype(np.uint8))
        cv2.imshow(title, img[0,i,:,:]) #.astype(np.uint8))
    
    inkmaximized = np.zeros((30, pred_shape[0], pred_shape[1]), dtype=np.float16)
    inkmaximizedrand = np.zeros((30, pred_shape[0], pred_shape[1]), dtype=np.float16)
    for step, (images,xys) in tqdm(enumerate(test_loader),total=len(test_loader)):
       
        # Create template images for gradient ascent visualization
        images = images.half().to(device)
        imagesrand = torch.rand(images.shape, requires_grad=True, device=images.device, dtype=images.dtype) # Max normalize value is .7842 for scroll content
        imagesrandnoink = imagesrand.clone().detach().requires_grad_(True)
        imagesrandorig = imagesrand.clone().detach().requires_grad_(True)
        batch_size = images.size(0)

        # Run the model to obtain gradients on the inputs
        print("Running inference on images", step, images.shape, images.device, model.device)
        with torch.no_grad():
            y_preds = model.half()(images)
            z_preds = model(imagesrand)
            print("y_preds", y_preds.shape, y_preds.min(), y_preds.max(), y_preds.mean(), y_preds.std())
            print("z_preds", z_preds.shape, z_preds.min(), z_preds.max(), z_preds.mean(), z_preds.std())
            print("images", images.shape, images.min(), images.max(), images.mean(), images.std())
            print("imagesrand", imagesrand.shape, imagesrand.min(), imagesrand.max(), imagesrand.mean(), imagesrand.std())

        # Create template images for gradient ascent visualization
        imagesgrad = images.clone().detach().requires_grad_(True)
        imagesgradnoink = images.clone().detach().requires_grad_(True)
        imagesgradorig = imagesrand.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([imagesgrad,], lr=100000)
        optimizer2 = torch.optim.SGD([imagesrand,], lr=100000)

        # Define the hook function to save activations
        def gradup(image, model, alpha, theta_decay = 0.2):
          image.retain_grad()
          global activations
          pred = model.float()(image.float())
          loss = - activations[:,0,:,:].mean()
          loss.backward(retain_graph=True)
          image = torch.add(image, torch.mul(image.grad, alpha))
          image = torch.mul(image, (1.0 - theta_decay)).requires_grad_(True)
          return image.half(), loss, pred

        alpha = torch.tensor(10000) # LR
        for i in tqdm(range(50)):
          imagesgrad, loss, pred = gradup(imagesgrad, model, alpha)
          print("Loss:", loss, "gradstats", imagesgrad[0,0].min(), imagesgrad[0,0].max(), imagesgrad[0,0].mean())
          #for d in range(imagesgrad.shape[2]):
          #  cv2.imwrite("ink"+str(xys[0].tolist())+"_"+str(d)+"z_"+str(i)+".png", (255*norm(imagesgrad[0,0,d].detach().cpu().numpy())).astype(np.uint8))
          #  cv2.imwrite("meansubink"+str(xys[0].tolist())+"_"+str(d)+"z_"+str(i)+".png", (255*norm((imagesgrad[0,0,d]-imagesgradorig[0,0,d]).detach().cpu().numpy())).astype(np.uint8))
          #images = images + self.gradients * self.lr
          #imshow('images', images[0].detach().cpu().float().numpy())

          imagesgradnoink, loss, pred = gradup(imagesgradnoink, model, -alpha)
          print("Loss:", loss, "gradstats", imagesgradnoink[0,0].min(), imagesgradnoink[0,0].max(), imagesgradnoink[0,0].mean())
          #for d in range(imagesgradnoink.shape[2]):
          #  cv2.imwrite("noink"+str(xys[0].tolist())+"_"+str(d)+"z_"+str(i)+".png", (255*norm(imagesgradnoink[0,0,d].detach().cpu().numpy())).astype(np.uint8))
          #  cv2.imwrite("meansubnoink"+str(xys[0].tolist())+"_"+str(d)+"z_"+str(i)+".png", (255*norm((imagesgradnoink[0,0,d]-imagesgradorig[0,0,d]).detach().cpu().numpy())).astype(np.uint8))

          imagesrandnoink, loss, pred = gradup(imagesrandnoink, model, -alpha)
          print("Loss:", loss, "randstats", imagesrandnoink[0,0].min(), imagesrandnoink[0,0].max(), imagesrandnoink[0,0].mean())
          #for d in range(imagesrandnoink.shape[2]):
          #  cv2.imwrite("randnoink_"+str(xys[0].tolist())+"_"+str(d)+"z_"+str(i)+".png", (255*norm(imagesrandnoink[0,0,d].detach().cpu().numpy())).astype(np.uint8))
          #  cv2.imwrite("randmeansubnoink_"+str(xys[0].tolist())+"_"+str(d)+"z_"+str(i)+".png", (255*norm((imagesrandnoink[0,0,d]-imagesrandorig[0,0,d]).detach().cpu().numpy())).astype(np.uint8))

          imagesrand, loss, pred = gradup(imagesrand, model, -alpha)
          print("Loss:", loss, "randstats", imagesrand[0,0].min(), imagesrand[0,0].max(), imagesrand[0,0].mean())
          #for d in range(imagesrand.shape[2]):
          #  cv2.imwrite("rand_"+str(xys[0].tolist())+"_"+str(d)+"z_"+str(i)+".png", (255*norm(imagesrand[0,0,d].detach().cpu().numpy())).astype(np.uint8))
          #  cv2.imwrite("randmeansub_"+str(xys[0].tolist())+"_"+str(d)+"z_"+str(i)+".png", (255*norm((imagesrand[0,0,d]-imagesrandorig[0,0,d]).detach().cpu().numpy())).astype(np.uint8))
          '''
          imagesrand.retain_grad()
          #optimizer2.zero_grad()
          predtemp2 = model(imagesrand)
          #print("activations2.shape", activations.shape) 
          #print("Len of activations2", len(activations), "shape", [a.shape for a in activations])
          loss = - activations[:, 0, :, :].mean()
          print("randloss", loss)
          loss.backward(retain_graph=True) #retain_graph = True)
          #optimizer2.step()
          imagegrad = imagesrand.grad
          imagesrand = torch.add(imagesrand, torch.mul(imagegrad, alpha))
          theta_decay = 0.1
          imagesrand = torch.mul(imagesrand, (1.0 - theta_decay)).requires_grad_(True)
          print("randstats", imagesrand[0,0].min(), imagesrand[0,0].max(), imagesrand[0,0].mean())
          for d in range(imagesrand.shape[2]):
            cv2.imwrite("randink"+str(xys[0].tolist())+"_"+str(d)+"z_"+str(i)+".png", (255*norm(imagesrand[0,0,d].detach().cpu().numpy())).astype(np.uint8))
            cv2.imwrite("randmeansubink"+str(xys[0].tolist())+"_"+str(d)+"z_"+str(i)+".png", (255*norm((imagesrand[0,0,d]-imagesrandorig[0,0,d]).detach().cpu().numpy())).astype(np.uint8))
          '''
          #imshow('imagesrand', imagesrand[0].detach().cpu().float().numpy())
          #cv2.waitKey(1)

        y_preds = torch.sigmoid(y_preds).to('cpu')
        z_preds = torch.sigmoid(z_preds).to('cpu')
        print("xys", len(xys), xys[-4:])

        # OPTIONAL step:
        '''
        for i, (x1, y1, x2, y2) in enumerate(xys):
            mask_pred[y1:y2, x1:x2] += np.multiply(F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy(),kernel)
            mask_count[y1:y2, x1:x2] += np.ones((CFG.size, CFG.size))
            rand_pred[y1:y2, x1:x2] += np.multiply(F.interpolate(z_preds[i].unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy(),kernel)
            mask_count[y1:y2, x1:x2] += np.ones((CFG.size, CFG.size))
            inkmaximized[:, y1:y2, x1:x2] += imagesgrad[i,0].detach().cpu().numpy()
            inkmaximizedrand[:, y1:y2, x1:x2] += imagesrand[i,0].detach().cpu().numpy()
        '''

    mask_pred /= mask_count
    mask_pred *= 2  #max(1, mask_count)
    rand_pred /= mask_count
    rand_pred *= 2  #max(1, mask_count)
    inkmaximized /= mask_count
    inkmaximizedrand /= mask_count
    
    # Now also subtract out the original image and imagerand tensor to see what the "residual" of ink is at each layer. Render as a volume somehow. A 3D explorable pointcloud.
    
    # mask_pred/=mask_pred.max()
    return mask_pred, rand_pred, inkmaximized

def get_img_splits(fragment_id,s,e,rotation=0):
    images = []
    xyxys = []
    fragment_mask = None
    images, mask, xyxys, ids = get_xyxys([fragment_id], is_valid=True)
    test_dataset = CustomDatasetTest(images,np.stack(xyxys), ids, CFG,transform=A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(
            mean= [0] * CFG.in_chans,
            std= [1] * CFG.in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]))

    test_loader = DataLoader(test_dataset,
                              batch_size=CFG.valid_batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False,
                              )
    return test_loader, np.stack(xyxys),(images[fragment_id].shape[0],images[fragment_id].shape[1]),fragment_mask

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

class InferenceArgumentParser(Tap):
    segment_id: str ='20230925002745'
    segment_path:str='./eval_scrolls'
    model_path:str= 'wild12_64_20230820203112_0_fr_i3depoch=18.ckpt' #outputs/vesuvius/pretraining_all/vesuvius-models/valid_20230827161847_0_fr_i3depoch=7.ckpt'
    out_path:str='./'
    stride: int = 16 #128 #1024 #16 #2
    start_idx:int=15
    workers: int = 25
    batch_size: int = 1
    size:int=64 #256 #2048 #64
    reverse:int=0
args = InferenceArgumentParser().parse_args()

fragments=os.listdir('./eval_scrolls')
fragment_id=args.segment_id
args.start_idx = 15
print("Running with fragment id", fragment_id)
test_loader,test_xyxz,test_shape,fragment_mask=get_img_splits(fragment_id,args.start_idx,args.start_idx+26,0) # Get all the slices!
model=RegressionPLModel.load_from_checkpoint(args.model_path,strict=False)
model.cuda().half() # SethS half
model.eval()
print("Predicting...")
mask_pred, rand_pred, inkmaximized = predict_fn(test_loader, model, device, test_xyxz,test_shape)
mask_pred=np.clip(np.nan_to_num(mask_pred),a_min=0,a_max=1)
mask_pred/=max(1,mask_pred.max())
mask_pred=(mask_pred*255).astype(np.uint8)
mask_pred=Image.fromarray(mask_pred)
print("saving mask pred", fragment_id)
mask_pred.save(f'{args.out_path}/NEW_{fragment_id}_{args.stride}_{args.size}_{args.start_idx}.png')
print("Done!")
