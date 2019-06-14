data: 数据集放置位置

dev：攻击图片放置位置，训练时测试在攻击样本上的acc
    lzp：攻击图片 
    dev_data_110：官方110张验证集


config： 配置类，存放路径和训练参数

trades： trade off的loss函数，代替交叉熵

util：数据导入等

network：分类网络，除了vgg_rse外都是标准的分类网络（未修改），vgg_rse在卷积层前加入高斯噪声层


models:保存模型的位置，denosie_LZP下模型使用我们攻击同学产生的攻击样本生成的去噪器
