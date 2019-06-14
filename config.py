import os
import numpy
import sys
def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

class Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))#求当前文件夹绝对目录
    allimg_dir = os.path.join(cur_dir,'data','lwx')#总的图片路径
    dataset_mean = [0.6502, 0.6031, 0.5935]
    dataset_std =[0.3182, 0.3239, 0.3257]
    #保存路径
    savePath  = os.path.join(cur_dir,'models')
    #需加载模型路径
    pretrain =''
    #测试时用的攻击样本路径
    adv_path ='./dev/LZP'#dev_adv_110,small_adv

    #用的一个去噪器noisetonosie https://arxiv.org/abs/1803.04189,u-net结构，效果不大
    ckpt_fname ='./models/denoise_lzp/n2n-epoch8-0.00087.pt'

    #训练设置参数
    lr =0.0003



cfg = Config()
add_pypath(cfg.cur_dir)#添加该项目路径放入搜索路径首位
