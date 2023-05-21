from PIL import Image
import numpy as np


def readData(index):
    image_dir = r"/root/autodl-tmp/Pytorch-UNet-master/data/train_raw/{}.png".format(index)
    x = Image.open(image_dir)  # 打开图片
    data = np.asarray(x)  # 转换为矩阵
    data_origin = data[:, :256]
    data_target = data[:, 256:]
    image_origin = Image.fromarray(data_origin)
    image_target = Image.fromarray(data_target)
    image_origin.save("/root/autodl-tmp/Pytorch-UNet-master/data/imgs_train/{}.png".format(index))
    image_target.save("/root/autodl-tmp/Pytorch-UNet-master/data/masks_train/{}.png".format(index))


for index in range(133):
    readData(index+1)
