#测试防御模型
#模拟线上测试
from collections import OrderedDict
from network.densenet import densenet121, densenet161
from network.resnetbase import resnet50
from util import  load_data_for_defense2,parse_args
import torch
import glob
import pandas as pd
import numpy as np
from progressbar import *
from network.inceptionv4 import  inceptionv4
from network.se_resnet import  se_resnet50
from network.se_densenet_full_in_loop import se_densenet121
from network.vgg_rse import  vgg16_bn
from network.unet import UNet
from config import  cfg
import  time
model_class_map = {
    'densenet121': densenet121,
    'densenet161': densenet161,
    'resnet50' : resnet50,
    'inceptionv4':inceptionv4,
    'senet50' : se_resnet50,
    'se_densenet121':se_densenet121,
    'vgg16':vgg16_bn,

}


#测试时添加高斯噪声
def defense(input_dir, target_model, weights_path, output_file, batch_size):# defense_type, defense_params
    # Define CNN model
    Model = model_class_map[target_model]
    #根据大小数据集更改类别
    model = Model(num_classes=110)
    # Loading data for ...
    print('loading data for defense using %s ....' %target_model)
    img_size = model.input_size[0]
    loaders = load_data_for_defense2(input_dir, img_size, batch_size) #若是randomnosie的模型选择load_data_for_defense

    # Prepare predict options
    device = torch.device('cuda:0')
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    pth_file = glob.glob(os.path.join(weights_path, 'ep_*.pth'))[0]
    print('loading weights from : ', pth_file)
    #model.load_state_dict(torch.load(pth_file)['model_state_dict'])

    state_dict = torch.load(pth_file)['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.'+k # remove `module.`
        #print(name)
        new_state_dict[name] = v # load params

    model.load_state_dict(new_state_dict)

    # for store result
    result = {'filename':[], 'predict_label':[]}
    alllogits =[]
    # Begin predicting
    model.eval()
    widgets = ['dev_data :',Percentage(), ' ', Bar('#'),' ', Timer(),
       ' ', ETA(), ' ', FileTransferSpeed()]

    #加载去噪模块
    model_denoise = UNet(in_channels=3)
    model_denoise.load_state_dict(torch.load(cfg.ckpt_fname))
    model_denoise = model_denoise.to(device)
    model_denoise.eval()
    std =0.40
    testTime =10
    for  _time in range(testTime):
        pbar = ProgressBar(widgets=widgets)
        for batch_data in pbar(loaders['dev_data']):
            image = batch_data['image'].to(device)
            filename = batch_data['filename']
            # 去噪
            denoised_img = model_denoise(image)
            denoised_img = denoised_img.to(device)

            buffer = torch.Tensor(image.size()).normal_(0, std).cuda()
            x_adv = denoised_img.detach() + buffer
            x_adv =torch.clamp(x_adv,0,1)
            with torch.no_grad():
                logits = model(x_adv)
            temp = logits.detach().cpu().numpy().tolist()
            if _time ==0:
                result['filename'].extend(filename)
                alllogits.extend(temp)
            else:
                for numfile in range(len(filename)):
                    index =  result['filename'].index(filename[numfile])
                    #list对应元素相加
                    for _e in range(len(alllogits[index])):
                        alllogits[index][_e] += temp[numfile][_e]

    for k  in range(len(alllogits)):
        ss =alllogits[k]
        y_pred = alllogits[k].index(max(ss))
        result['predict_label'].extend([y_pred])

    print('write result file to : ', output_file)
    pd.DataFrame(result).to_csv(output_file, header=False, index=False)

#多个模型投票测试
def denfense_joint(input_dir, output_file, output_joint,batch_size):
    model_name ={'se_densenet121':'se_densenet121','densenet121_l2':'densenet121','vgg16':'vgg16','densenet_l2gs':'densenet121','resnet50_inf8':'resnet50',\
                 'se_resnet50':'senet50'}
    #对应模型的位置
    model_dir =['./models/se_densenet121/std_0.5','./models/densenet121/trade_l2_random_eps4to12',\
                './models/vgg16/new','./models/densenet121/gs_0.4','./models/resnet50/gs_0.5',\
                './models/senet/gs_0.5']
################################################################
    #第一个模型se_densenet121,18.5
    Model1 = model_class_map[model_name['se_densenet121']]
    # 根据大小数据集更改类别
    model1 = Model1(num_classes=110)
    img_size = model1.input_size[0]
    loaders = load_data_for_defense2(input_dir, img_size, False , batch_size)

    device = torch.device('cuda:0')
    model1 = model1.to(device)
    #model1 = torch.nn.DataParallel(model1)
    pth_file = glob.glob(os.path.join(model_dir[0], 'ep_*.pth'))[0]
    print('loading weights1 from : ', pth_file)
    model1.load_state_dict(torch.load(pth_file)['model_state_dict'])

    # for store result
    result1 = {'filename': [], 'predict_label': []}
    alllogits1 = []
    # Begin predicting

    #加载去噪模块
    model_denoise = UNet(in_channels=3)
    model_denoise.load_state_dict(torch.load(cfg.ckpt_fname))
    model_denoise = model_denoise.to(device)
    model_denoise.eval()

    model1.eval()
    widgets = ['dev_data :', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    std = 0.40
    testTime = 30
    for _time in range(testTime):
        pbar = ProgressBar(widgets=widgets)
        for batch_data in pbar(loaders['dev_data']):
            image = batch_data['image'].to(device)
            filename = batch_data['filename']

            # 去噪
            denoised_img = model_denoise(image)
            denoised_img = denoised_img.to(device)

            buffer = torch.Tensor(image.size()).normal_(0, std).cuda()
            x_adv = denoised_img.detach() + buffer
            x_adv = torch.clamp(x_adv, 0, 1)
            with torch.no_grad():
                logits = model1(x_adv)
                #logits_softmax = F.softmax(logits,dim=1)
            temp = logits.detach().cpu().numpy().tolist()
            if _time == 0:
                result1['filename'].extend(filename)
                alllogits1.extend(temp)
            else:
                for numfile in range(len(filename)):
                    index = result1['filename'].index(filename[numfile])
                    # list对应元素相加
                    for _e in range(len(alllogits1[index])):
                        alllogits1[index][_e] += temp[numfile][_e]

    #每张图概率除以输入次数
    for  _n in range(len(alllogits1)):
        for _e in range(len(alllogits1[_n])):
            alllogits1[_n][_e] /= testTime


    for k in range(len(alllogits1)):
        ss = alllogits1[k]
        y_pred = alllogits1[k].index(max(ss))
        result1['predict_label'].extend([y_pred])
    print('write result file to : ', output_file[0])
    pd.DataFrame(result1).to_csv(output_file[0], header=False, index=False)
################################################################################
    #第二个模型densenet121_l2_2to12 16.6
    Model2 = model_class_map[model_name['densenet121_l2']]
    # 根据大小数据集更改类别
    model2 = Model2(num_classes=110)

    model2 = model2.to(device)
    pth_file = glob.glob(os.path.join(model_dir[1], 'ep_*.pth'))[0]
    print('loading weights2 from : ', pth_file)

    state_dict = torch.load(pth_file)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        #print(name)
        new_state_dict[name] = v # load params

    model2.load_state_dict(new_state_dict)


    # for store result
    result2 = {'filename': [], 'predict_label': []}
    # Begin predicting
    model2.eval()
    widgets = ['dev_data :', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar2 = ProgressBar(widgets=widgets)
    alllogits2 = []
    for batch_data in pbar2(loaders['dev_data']):
        image = batch_data['image'].to(device)
        filename = batch_data['filename']
        #去噪
        denoised_img = model_denoise(image)
        denoised_img = denoised_img.to(device)
        with torch.no_grad():
            logits = model2(denoised_img)
            #logits_softmax = F.softmax(logits,dim=1)

        temp = logits.detach().cpu().numpy().tolist()
        alllogits2.extend(temp)

        y_pred = logits.max(1)[1].detach().cpu().numpy().tolist()
        result2['filename'].extend(filename)
        result2['predict_label'].extend(y_pred)

    print('write result file to : ', output_file[1])
    pd.DataFrame(result2).to_csv(output_file[1], header=False, index=False)

#######################################################################
    #第三个模型vgg+noise层 ,16.5
    Model3 = model_class_map[model_name['vgg16']]
    # 根据大小数据集更改类别
    model3 = Model3(num_classes=110)
    model3 = model3.to(device)
    pth_file = glob.glob(os.path.join(model_dir[2], 'ep_*.pth'))[0]
    print('loading weights3 from : ', pth_file)
    model3.load_state_dict(torch.load(pth_file)['model_state_dict'])

    result3 = {'filename':[], 'predict_label':[]}
    alllogits3 =[]
    # Begin predicting
    model3.eval()
    widgets = ['dev_data :',Percentage(), ' ', Bar('#'),' ', Timer(),
       ' ', ETA(), ' ', FileTransferSpeed()]
    testTime =30
    for  _time in range(testTime):
        pbar = ProgressBar(widgets=widgets)
        for batch_data in pbar(loaders['dev_data']):
            image = batch_data['image'].to(device)
            filename = batch_data['filename']
            # 去噪
            denoised_img = model_denoise(image)
            denoised_img = denoised_img.to(device)

            with torch.no_grad():
                logits = model3(denoised_img)
                #logits_softmax = F.softmax(logits,dim=1)
            temp = logits.detach().cpu().numpy().tolist()
            if _time ==0:
                result3['filename'].extend(filename)
                alllogits3.extend(temp)
            else:
                for numfile in range(len(filename)):
                    index =  result3['filename'].index(filename[numfile])
                    #list对应元素相加
                    for _e in range(len(alllogits3[index])):
                        alllogits3[index][_e] += temp[numfile][_e]

    #每张图概率除以输入次数
    for  _n in range(len(alllogits3)):
        for _e in range(len(alllogits3[_n])):
            alllogits3[_n][_e] /= testTime

    for k  in range(len(alllogits3)):
        ss =alllogits3[k]
        y_pred = alllogits3[k].index(max(ss))
        result3['predict_label'].extend([y_pred])
    print('write result file to : ', output_file[2])
    pd.DataFrame(result3).to_csv(output_file[2], header=False, index=False)

################################################################
    #第六个模型se-resnet gs_0.5  17.5
    Model6= model_class_map[model_name['se_resnet50']]
    # 根据大小数据集更改类别
    model6 = Model6(num_classes=110)
    model6 = model6.to(device)
    # model1 = torch.nn.DataParallel(model1)
    pth_file = glob.glob(os.path.join(model_dir[5], 'ep_*.pth'))[0]
    print('loading weights6 from : ', pth_file)

    state_dict = torch.load(pth_file)['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        #print(name)
        new_state_dict[name] = v # load params

    model6.load_state_dict(new_state_dict)

    # for store result
    result6 = {'filename': [], 'predict_label': []}
    alllogits6 = []
    # Begin predicting
    model6.eval()
    widgets = ['dev_data :', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    std = 0.40
    testTime = 30
    for _time in range(testTime):
        pbar = ProgressBar(widgets=widgets)
        for batch_data in pbar(loaders['dev_data']):
            image = batch_data['image'].to(device)
            filename = batch_data['filename']

            # 去噪
            denoised_img = model_denoise(image)
            denoised_img = denoised_img.to(device)

            buffer = torch.Tensor(image.size()).normal_(0, std).cuda()
            x_adv = denoised_img.detach() + buffer
            x_adv = torch.clamp(x_adv, 0, 1)
            with torch.no_grad():
                logits = model6(x_adv)
                #logits_softmax = F.softmax(logits,dim =1)
            temp = logits.detach().cpu().numpy().tolist()
            if _time == 0:
                result6['filename'].extend(filename)
                alllogits6.extend(temp)
            else:
                for numfile in range(len(filename)):
                    index = result6['filename'].index(filename[numfile])
                    # list对应元素相加
                    for _e in range(len(alllogits6[index])):
                        alllogits6[index][_e] += temp[numfile][_e]

    #每张图概率除以输入次数
    for  _n in range(len(alllogits6)):
        for _e in range(len(alllogits6[_n])):
            alllogits6[_n][_e] /= testTime

    for k in range(len(alllogits6)):
        ss = alllogits6[k]
        y_pred = alllogits6[k].index(max(ss))
        result6['predict_label'].extend([y_pred])
    print('write result file to : ', output_file[5])
    pd.DataFrame(result6).to_csv(output_file[5], header=False, index=False)


    #logit 加权法

    numpy_alllogits1= np.array(alllogits1)#18.5
    numpy_alllogits2 = np.array(alllogits2)#16.6  densenet121_l2
    numpy_alllogits3 = np.array(alllogits3)#16.5  vgg+noise
    #numpy_alllogits4 = np.array(alllogits3) # 17.4  densenet121_l2_gs0.5
    numpy_alllogits6 = np.array(alllogits6)#17.5  se-resnet50_gs0.5

    all_result = {'filename': [], 'predict_label': []}
    numpy_all = 0.8 *numpy_alllogits1+0.2*numpy_alllogits2+0.15*numpy_alllogits3+0.4*numpy_alllogits6
    all = numpy_all.tolist()
    for k in range(len(alllogits1)):
        ss = all[k]
        y_pred = all[k].index(max(ss))
        all_result['predict_label'].extend([y_pred])
    all_result['filename'] = result1['filename'].copy()
    print('write result file to : ', output_joint)
    pd.DataFrame(all_result).to_csv(output_joint, header=False, index=False)

    # #加权概率法
    # all_result = {'filename': [], 'predict_label': []}
    # resultLogit =[]
    # model1_radio =0.4
    # model2_radio =0.15
    # model3_radio =0.2
    # model6_radio =0.25
    # for _num in range(len(alllogits1)):
    #     temp =[]
    #     for _e in range(len(alllogits1[_num])):
    #         a =model1_radio * alllogits1[_num][_e] + model2_radio * alllogits2[_num][_e]  +model3_radio * alllogits3[_num][_e] +model6_radio *alllogits6[_num][_e]
    #         temp.append(a)
    #     resultLogit.append(temp)
    #
    # for k in range(len(alllogits1)):
    #     ss = resultLogit[k]
    #     y_pred = resultLogit[k].index(max(ss))
    #     all_result['predict_label'].extend([y_pred])
    #
    # all_result['filename'] = result3['filename'].copy()
    # print('write result file to : ', output_joint)
    # pd.DataFrame(all_result).to_csv(output_joint, header=False, index=False)
    #return  resultLogit


    # from collections import  Counter
    # #投票法：票数最多的分类定为正确分类,票数相同时计算每个group的线上分数高者为输出,每个分类不同时以最高线上分数的模型分类为输出
    # temp =[]
    # model_num =len(model_name)
    # all_result = {'filename': [], 'predict_label': []}
    # online_nums =[18.5,18.5,16.6,16,17] #要和使用模型对齐
    # for nn in range(len(result1['predict_label'])):
    #     temp.clear()
    #     temp.append(result1['predict_label'][nn])
    #     temp.append(result1['predict_label'][nn])
    #     temp.append(result2['predict_label'][nn])
    #     temp.append(result3['predict_label'][nn])
    #     #temp.append(result4['predict_label'][nn])
    #     #temp.append(result5['predict_label'][nn])
    #     temp.append(result6['predict_label'][nn])
    #     result =Counter(temp)
    #     #排降序
    #     d = sorted(result.items(),key = lambda x:x[1],reverse=True)
    #     print(d)
    #
    #     if len(d) == model_num or len(d)==1:
    #         all_result['predict_label'].append(d[0][0])
    #     else:
    #         group_num =len(d)
    #         #具有相同票数的标签list
    #         labels =[d[0][0]]
    #         jj = 0
    #         while  jj <group_num-1:
    #             if d[jj][1] == d[jj+1][1]:
    #                 labels.append(d[jj][0])
    #                 jj +=1
    #             else:
    #                 break
    #         if len(labels) ==1 :
    #             all_result['predict_label'].append(labels[0])
    #         else:
    #             #计算相同票数标签group的线上分数和
    #             group_sum =[]
    #             for  _out1 in range(len(labels)):
    #                 temp_sum =0
    #                 for _out2 in range(len(temp)):
    #                     if labels[_out1] == temp[_out2]:
    #                         temp_sum += online_nums[_out2]
    #                 group_sum.append(temp_sum)
    #             group_index =group_sum.index(max(group_sum))
    #             all_result['predict_label'].append(labels[group_index])
    #
    # all_result['filename'] = result3['filename'].copy()
    # print('write result file to : ', output_joint)
    # pd.DataFrame(all_result).to_csv(output_joint, header=False, index=False)

def acc(mystr,resultcsv_file,devcsv_file):
    # 本次测试题目str
    df1 = pd.read_csv(resultcsv_file, header=None)  # csv没有列名一定要加
    df2 = pd.read_csv(devcsv_file)
    sorteddf1 = df1.sort_values(by=1)
    sorteddf2 = df2.sort_values(by='trueLabel')
    resultFilename = sorteddf1[0].values
    resultLabel = sorteddf1[1].values
    devFilename = sorteddf2['filename'].values
    devLabel = sorteddf2['trueLabel'].values

    #print(resultFilename)
    allImageNum = len(resultFilename)
    rightImage = 0.0
    wrong2target = {}
    for i in range(allImageNum):
        a = np.argwhere(devFilename == resultFilename[i])

        if devLabel[int(a)] == resultLabel[i]:
            rightImage += 1.0
        else:
            wrong2target[ f'{devLabel[int(a)]}']  =  resultLabel[i]
    print(' ')
    print('right:error')
    dict = sorted(wrong2target.items(), key=lambda d: d[0])
    print(dict)

    print(mystr + ' acc:', rightImage / float(allImageNum))
    f = open('./output_data/record.txt', 'a')
    print(mystr + ' acc:', rightImage / float(allImageNum), file=f)
    f.close()
    return  wrong2target
def acc2(mystr,resultcsv_file,devcsv_file):
    # 本次测试题目str
    df1 = pd.read_csv(resultcsv_file, header=None)  # csv没有列名一定要加
    df2 = pd.read_csv(devcsv_file,header=None)
    sorteddf1 = df1.sort_values(by=1)
    sorteddf2 = df2.sort_values(by=1)
    resultFilename = sorteddf1[0].values
    resultLabel = sorteddf1[1].values
    devFilename = sorteddf2[0].values
    devLabel = sorteddf2[1].values
    #清洗filename
    for i in range(len(devFilename)):
        devFilename[i] =devFilename[i].split('/')[-1]
    #print(devFilename)


    #print(resultFilename)
    allImageNum = len(resultFilename)
    rightImage = 0.0
    for i in range(allImageNum):
        a = np.argwhere(devFilename == resultFilename[i])

        if devLabel[int(a)] == resultLabel[i]:
            rightImage += 1.0

    print(mystr + ' acc:', rightImage / float(allImageNum))
    f = open('./output_data/record.txt', 'a')
    print(mystr + ' acc:', rightImage / float(allImageNum), file=f)
    f.close()

if __name__ == '__main__':
    args = parse_args()
    gpu_ids = args.gpu_id
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    batch_size = args.batch_size
    testTitle ='all_joint'

    input_dir ='./dev/dev_data_110'
    input_dir2 ='./dev/LZP'
    #测试集干净图片位置
    devcsv_file='./dev/dev_data_110/dev.csv'

    #需要更改的部分
    #单位置
    weights_path='./models/se_densenet121/std_0.5'
    #预测csv保存位置
    outputfile =os.path.join(weights_path,'result110.csv')
    outputfile2 =os.path.join(weights_path,'result110_adv.csv')
    # args.target_model,单模型使用
    target_model = 'se_densenet121'

    defense(input_dir, target_model, weights_path, outputfile, 11)
    #自然图片下acc
    acc(testTitle, outputfile, devcsv_file)

    defense(input_dir2, target_model, weights_path, outputfile2, 11)
    #攻击图片下acc
    acc(testTitle, outputfile2, devcsv_file)




    #多个模型，保存位置
    # joint_csvdir ='./output_data/joint'
    # outputfile_list =[os.path.join(joint_csvdir,'model1.csv'),os.path.join(joint_csvdir,'model2.csv'),os.path.join(joint_csvdir,'model3.csv'),\
    #                   os.path.join(joint_csvdir,'model4.csv'),os.path.join(joint_csvdir,'model5.csv'),os.path.join(joint_csvdir,'model6.csv')]
    # outputfile_joint =os.path.join(joint_csvdir,'joint.csv')
    # bool_small =False
    #
    #
    # #联合模型
    # start_time = time.time()
    # denfense_joint(input_dir2,outputfile_list,outputfile_joint,11)
    # a1=acc('se_densenet121', outputfile_list[0], devcsv_file)
    # a2=acc('densenet121_l2', outputfile_list[1], devcsv_file)
    # a3=acc('vgg16', outputfile_list[2], devcsv_file)
    # a6 = acc('seresnet50_gs0.5', outputfile_list[5], devcsv_file)
    #
    # a7=acc('all', outputfile_joint, devcsv_file)
    #
    # print('')
    # #错误的类进行统计
    # real ={}
    # for k in [a1,a2,a3,a6]:
    #     index = k.keys()
    #     for _index in index:
    #         if _index in real.keys() :
    #            real[_index].append(k[_index])
    #         else:
    #             real[_index] =[k[_index]]
    # print(real)
    #
    #
    # end_time = time.time()
    # print('time is :',end_time-start_time)
