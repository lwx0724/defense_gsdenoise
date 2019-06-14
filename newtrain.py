#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
from PIL import Image
import torch.nn.functional as F
from trades import trades_loss
from progressbar import *
from network.densenet import densenet121, densenet161
from network.resnetbase import  resnet50
from network.vgg_rse import vgg16_bn
from util import load_data_for_training_cnn2,parse_args
from config import cfg
from network.se_densenet_full_in_loop import se_densenet121
from network.se_resnet import  se_resnet50

Image.LOAD_TRUNCATED_IMAGES = True

model_class_map = {
    'densenet121': densenet121,
    'densenet161': densenet161,
    'resnet50':resnet50,
    'vgg16':vgg16_bn,
    'se_densenet121':se_densenet121,
    'se_resnet50': se_resnet50,
}

import torch
class AdamW(torch.optim.Optimizer):
    """Implements AdamW algorithm.

    It has been proposed in `Fixing Weight Decay Regularization in Adam`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. Fixing Weight Decay Regularization in Adam:
    https://arxiv.org/abs/1711.05101
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # according to the paper, this penalty should come after the bias correction
                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # w = w - wd * lr * w
                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                # w = w - lr * w.grad
                p.data.addcdiv_(-step_size, exp_avg, denom)

                # w = w - wd * lr * w - lr * w.grad
                # See http://www.fast.ai/2018/07/02/adam-weight-decay/

        return loss

def do_train_trades(model_name, model, train_loader, val_loader,adv_val_loader ,device,i_ep, trainStr, lr=0.0001, n_ep=40,
             save_path='/tmp'):
    optimizer = AdamW(model.parameters(),weight_decay=1e-4,lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5, verbose=True, patience=4)
    # do training
    i_ep =i_ep+1
    #绕动规则
    #设置对抗样本的绕动，训练时逐步加大，不要一开始就设置很大，例如可以：10训练收敛后，在设置16,20
    rule_eps=[4,6,8,10,12,14,16,18,20,22]
    perturb_steps= 5
    std =0.1
    # epsilon_low = 3.51
    # epsilon_high = 7.02
    #epsilon =0.086
    print('lr is ',lr)
    while i_ep < n_ep:
        model.train()
        train_losses = []
        widgets = ['train :', Percentage(), ' ', Bar('#'), ' ', Timer(),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets)

        if i_ep <len(rule_eps):
            epsilon = rule_eps[i_ep]/255
            i_ep +=1
        else:
            epsilon =22 /255


        print('current std is ',std)
        print('current epslion is ',epsilon)
        for batch_data in pbar(train_loader):
            image = batch_data['image'].to(device)
            label = batch_data['label_idx'].to(device)
            optimizer.zero_grad()
            #logits = model(image)
            #loss = F.cross_entropy(logits, label)

            #最核心部分,先对图片加入高斯噪声后，再用I-fgsm的方式找其对应的对抗样本,使用trade off的loss函数
            #训练具有高斯噪声的模型，不能以来设很大的std,一开始设置0.1，收敛后再逐步加大继续训练
            buffer = torch.Tensor(image.size()).normal_(0,std).cuda()
            image = image + buffer.detach()
            image = torch.clamp(image,0.0,1.0)
            loss = trades_loss(model, image, label,optimizer,epsilon=epsilon ,perturb_steps =perturb_steps)
            loss.backward()
            optimizer.step()
            train_losses += [loss.detach().cpu().numpy().reshape(-1)]

        train_losses = np.concatenate(train_losses).reshape(-1).mean()

        # 观察模型在验证集上表现
        model.eval()
        val_losses = []
        preds = []
        true_labels = []
        widgets = ['val:', Percentage(), ' ', Bar('#'), ' ', Timer(),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets)
        #测试时同样对图片加入高斯噪声，再前向传播，因为具有随机性，测试时也需要多次测试取logit的均值，30次左右就可稳定输出
        #test_time1 是val干净图片测试次数.数量较多这里设置的10
        #test_time 是攻击过的图片测试次数
        test_time1 =10
        test_time  =30
        for batch_data in pbar(val_loader):
            image = batch_data['image'].to(device)
            label = batch_data['label_idx'].to(device)
            all_logits = torch.zeros((image.size()[0], 110)).to(device)
            for _time in range(test_time1):
                buffer = torch.Tensor(image.size()).normal_(0, std).cuda()
                image_gs = image.detach() + buffer
                image_gs =torch.clamp(image_gs,0.0,1.0)
                with torch.no_grad():
                    logits = model(image_gs)
                all_logits.add_(logits)
            all_logits.div_(test_time1)
            loss = F.cross_entropy(all_logits, label).detach().cpu().numpy().reshape(-1)
            val_losses += [loss]
            true_labels += [label.detach().cpu().numpy()]
            predictedList = all_logits.max(1)[1].detach().cpu().numpy()
            preds += [(predictedList)]

        preds = np.concatenate(preds, 0).reshape(-1)
        true_labels = np.concatenate(true_labels, 0).reshape(-1)
        acc = np.mean(true_labels == preds)
        val_losses = np.concatenate(val_losses).reshape(-1).mean()

        # 验证攻击acc
        model.eval()
        val_losses_adv = []
        preds_adv = []
        true_labels_adv = []
        widgets2 = ['val:', Percentage(), ' ', Bar('#'), ' ', Timer(),
                    ' ', ETA(), ' ', FileTransferSpeed()]
        pbar2 = ProgressBar(widgets=widgets2)
        for batch_data in pbar2(adv_val_loader):
            image_adv = batch_data['image'].to(device)
            label_adv = batch_data['label_idx'].to(device)
            all_adv_logits = torch.zeros((image_adv.size()[0], 110)).to(device)
            for _time in range(test_time):
                buffer = torch.Tensor(image_adv.size()).normal_(0, std).cuda()
                x_adv_gs = image_adv.detach() + buffer
                x_adv_gs =torch.clamp(x_adv_gs,0.0,1.0)
                with torch.no_grad():
                    logits_adv = model(x_adv_gs)
                all_adv_logits.add_(logits_adv)
            all_adv_logits.div_(test_time)
            loss_adv = F.cross_entropy(all_adv_logits, label_adv).detach().cpu().numpy().reshape(-1)
            val_losses_adv += [loss_adv]
            true_labels_adv += [label_adv.detach().cpu().numpy()]
            predictedList_adv = all_adv_logits.max(1)[1].detach().cpu().numpy()
            preds_adv += [(predictedList_adv)]

        preds_adv = np.concatenate(preds_adv, 0).reshape(-1)
        true_labels_adv = np.concatenate(true_labels_adv, 0).reshape(-1)
        acc_adv = np.mean(true_labels_adv == preds_adv)
        val_losses_adv = np.concatenate(val_losses_adv).reshape(-1).mean()
        scheduler.step(val_losses)

        print(
            f'Epoch : {i_ep}  val_acc : {acc:.5%} ||| val_adc_acc :{acc_adv:.5%}   ||| train_loss : {train_losses:.5f}  val_loss : {val_losses:.5f}  ||| val_loss_adv:{val_losses_adv:.5f}|||')
        #每个周期都保存
        torch.save({"model_state_dict": model.state_dict(),
                    "i_ep": i_ep,
                    }, os.path.join(save_path, f'ep_{i_ep}_{model_name}_val_acc_{acc:.4f}.pth'))
        model.to(device)
        i_ep +=1

def train_cnn_model(model_name, gpu_ids, batch_size,dataset_dir,trainstr):
    # Loading data for ...
    print('loading data for train %s ....' %model_name)

    num_classes =110
    #从零训练设置为False,否则设置为true，获取当前周期数
    i_ep_bool = False
    if i_ep_bool:
        checkpoint=torch.load(cfg.pretrain)
        i_ep = i_ep = checkpoint["i_ep"]
        print('i_ep is :',i_ep)
    else:
        i_ep =-1
    # Define CNN model
    Model = model_class_map[model_name]

    model = Model(pretrained=False,num_classes=num_classes)

    img_size = model.input_size[0]
    loaders = load_data_for_training_cnn2(dataset_dir, img_size,  batch_size=batch_size*len(gpu_ids))

    # Prepare training options
    save_path =  os.path.join(cfg.savePath,model_name)#'/home/mh/git_lwx/denseNet/models/%s' %model_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Let's use ", len(gpu_ids) , " GPUs!")
    device = torch.device('cuda:%d' %gpu_ids[0])
    model = model.to(device)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])

    print('start training cnn model.....\nit will take several hours, or even dozens of....')
    do_train_trades(model_name, model, loaders['train_data'], loaders['val_data'],loaders['adv_val_data'],
              device,i_ep,trainStr=trainstr ,lr=cfg.lr, save_path=save_path, n_ep=80)

if __name__=='__main__':
    args = parse_args()
    gpu_ids = args.gpu_id
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    batch_size = args.batch_size
    target_model = args.target_model
    dataset_dir = cfg.allimg_dir #切换数据集 #本次训练主题
    trainstr = 'TEST5-14'
    torch.backends.cudnn.benchmark = True
################## Training #######################
    train_cnn_model(target_model, gpu_ids, batch_size,dataset_dir,trainstr)


