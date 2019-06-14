#数据导入与处理
import os
import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import PIL
import argparse
import cv2
import numpy as np
from config import cfg
from network.densenet import densenet121, densenet161
import torch
from collections import OrderedDict
import shutil
from torchvision import transforms
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True

class ImageSet(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        #image = self.transformer(Image.open(image_path))#.convert('RGB'))
        image =Image.open(image_path)
        #image = image.filter(ImageFilter.GaussianBlur(radius=2))
        image =self.transformer(image)
        label_idx = self.df.iloc[item]['label_idx']
        sample = {
            'dataset_idx': item,
            'image': image,
            'label_idx': label_idx,
            'filename':os.path.basename(image_path)
        }
        return sample


#使用opencv打开
class ImageSet2(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        tmp = cv2.imread(image_path)
        img = cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB)
        dst = cv2.bilateralFilter(img,50,50,50)
        dst = Image.fromarray(dst)
        image = self.transformer(dst)#.convert('RGB'))
        label_idx = self.df.iloc[item]['label_idx']
        sample = {
            'dataset_idx': item,
            'image': image,
            'label_idx': label_idx,
            'filename':os.path.basename(image_path)
        }
        return sample

def load_data_for_training_cnn2(dataset_dir, img_size, batch_size=16):

    print(os.path.join(dataset_dir, './*/*.jpg'))
    all_imgs = glob.glob(os.path.join(dataset_dir, './*/*.jpg'))
    all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]
    train = pd.DataFrame({'image_path':all_imgs,'label_idx':all_labels})
    train_data, val_data = train_test_split(train,
                            stratify=train['label_idx'].values, train_size=0.95, test_size=0.05,random_state=0)

    #构建攻击数据集
    val_all_imgs = glob.glob(os.path.join(cfg.adv_path, '*.png'))
    print('adv_path:', os.path.join(cfg.adv_path, '*.png'))
    df = pd.read_csv('./dev/dev_data_110/dev.csv')
    devFilename = df['filename'].values
    devLabel = df['trueLabel'].values
    val_all_labels = []
    for img_path in val_all_imgs:
        a = np.argwhere(devFilename == img_path.split('/')[3])
        val_all_labels.append(devLabel[a[0][0]])
    adc_val = pd.DataFrame({'image_path': val_all_imgs, 'label_idx': val_all_labels})

    transformer_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(img_size, (0.7, 1), interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
    ])
    transformer = transforms.Compose([
        transforms.Resize([img_size, img_size], interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
    ])

    # 攻击验证集的transforms
    transformer_adv = transforms.Compose([
        transforms.Resize([img_size, img_size], interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
    ])

    datasets = {
        'train_data': ImageSet(train_data, transformer_train),
        'val_data':   ImageSet(val_data, transformer),
        'adv_val_data':ImageSet(adc_val,transformer_adv)
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=True,pin_memory=True) for ds in datasets.keys()
    }

    return dataloaders


#测试的图片,png格式
def load_data_for_defense(input_dir, img_size,jpgOrpng=True,batch_size=11):
    if jpgOrpng:
        all_img_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    else:
        all_img_paths = glob.glob(os.path.join(input_dir, '*.png'))
    all_labels = [-1 for i in range(len(all_img_paths))]
    dev_data = pd.DataFrame({'image_path':all_img_paths, 'label_idx':all_labels})

    transformer = transforms.Compose([
        transforms.Resize([img_size, img_size], interpolation=PIL.Image.ANTIALIAS),
        transforms.ToTensor(),
    ])
    datasets = {
        'dev_data': ImageSet(dev_data, transformer)
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=0,
                       shuffle=False) for ds in datasets.keys()
    }
    return dataloaders

#测试时使用随机裁剪,不用
def load_data_for_defense2(input_dir, img_size,jpgOrpng=True,batch_size=11):

    all_img_paths = glob.glob(os.path.join(input_dir, '*.png'))
    all_labels = [-1 for i in range(len(all_img_paths))]
    dev_data = pd.DataFrame({'image_path':all_img_paths, 'label_idx':all_labels})

    transformer = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(img_size, (0.9, 1), interpolation=PIL.Image.ANTIALIAS),
        transforms.ToTensor(),
    ])
    datasets = {
        'dev_data': ImageSet(dev_data, transformer)
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=0,
                       shuffle=False) for ds in datasets.keys()
    }
    return dataloaders

#求均值方差的函数
def meanAndStd(dataset_dir,sampleNum,img_size_h,img_size_w):
    print(os.path.join(dataset_dir, './*/*.jpg'))
    all_imgs = glob.glob(os.path.join(dataset_dir, './*/*.jpg'))
    all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]
    train = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    a=train.sample(sampleNum,axis=0)
    img_h, img_w = img_size_w,img_size_h
    imgs = np.zeros([img_w, img_h, 3, 1])
    means, stdevs = [], []
    for index, row in a.iterrows():
        img = cv2.imread(row['image_path'])
        img = cv2.resize(img,(img_size_h,img_size_w))
        img = img[:, :, :, np.newaxis]
        imgs = np.concatenate((imgs, img), axis=3)
        print(index)

    imgs = imgs.astype(np.float32)/255.
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

#命令行参数设计
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model', default='se_densenet121',
                        help='cnn model, e.g. , densenet121, resnet50', type=str)
    parser.add_argument('--gpu_id', default=0, nargs='+',
                        help='gpu ids to use, e.g. 0 1 2 3', type=int)
    parser.add_argument('--batch_size', default=16,
                        help='batch size, e.g. 16, 32, 64...', type=int)
    parser.add_argument('--input_dir',default='./smallSet10',help='allImage to ./all,partImage to ./smallSet10')
    parser.add_argument('--output_file',default='./output_data/result.csv',help='csv save path')
    return parser.parse_args()




