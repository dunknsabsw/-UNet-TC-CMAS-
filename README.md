## Training+Evaluation
训练+测试脚本为train.py\
输入通道数n_channels=1\
输出类别数n_classes=args.classes=256(由于给定标签为8位灰度图像,0-255,故设置共256个输出类别才能对齐)\
训练batch-size为args.batch_size_train,注意不能超过133(训练集大小)\
测试batch-size为args.batch_size_test,注意不能超过20(训练集大小)\
被调用的测试脚本为evaluate.py\
被调用的打分脚本为utils/dice_score.py\
被调用的数据加载脚本为utils/data_loading.py\
模型脚本为unet/unet_model.py(backbone)以及unet/unet_parts.py(封装模型各模块，如上下采样结构)\
该项目使用了wandb==0.13.5包可视化实验日志.\
本作业的完成基于Autodl服务器，一张Tesla T4.\
镜像:PyTorch1.11.0,Python3.8(ubuntu20.04),Cuda11.3,需要安装的包注明于requirements.txt文件.

## Data
数据在目录/data下.\
子目录train_raw存放133张原始256×512的训练集图像.\
子目录test_raw存放20张原始256×512的测试集图像.\
子目录imgs_train存放133张原始256×256的训练集图像输入.\
子目录imgs_test存放20张原始256×256的测试集图像输入.\
子目录masks_train存放133张原始256×256的训练集图像标签.\
子目录masks_test存放20张原始256×256的测试集图像标签.\
脚本split.py用于将子目录train_raw和子目录test_raw中的图片分开为输入和标签，并且分别存放到上述对应目录.脚本中的需要修改的参数只有三个路径以及主方法的循环轮次(应为读取目录下总图片张数).