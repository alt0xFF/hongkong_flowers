import os
import glob
import sys

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np
import cv2


'''
Author: Alan Chao

Using https://github.com/aleju/imgaug
and numpy, opencv>2.4

Installation:
$ sudo pip install numpy
Manually install python opencv
$ sudo pip install imgaug

Usage:
Assume your dataset is in:
/home/username/Desktop/dataset/006/00.jpg
/home/username/Desktop/dataset/006/01.jpg
...
$ python data_aug.py /home/username/Desktop/dataset
'''

def build_seqlist ():
  seq_list = []  # choose aug method here
  seq_list.append(iaa.Sequential([iaa.Fliplr(1)]))
  seq_list.append(iaa.Sequential([iaa.Flipud(1)]))
  seq_list.append(iaa.Sequential([iaa.Crop(percent=(0, 0.1))]))
  seq_list.append(iaa.Sequential([iaa.Affine(
      scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
      translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
      rotate=(-45, 45), # rotate by -45 to +45 degrees
      shear=(-16, 16), # shear by -16 to +16 degrees
      order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
      cval=(0, 255), # if mode is constant, use a cval between 0 and 255
      mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
  )]))
  seq_list.append(iaa.Sequential([iaa.GaussianBlur((0, 3.0))]))
  seq_list.append(iaa.Sequential([iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))]))
  seq_list.append(iaa.Sequential([iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))]))
  seq_list.append(iaa.Sequential([iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)]))
  seq_list.append(iaa.Sequential([iaa.Dropout((0.01, 0.1), per_channel=0.5)]))
  seq_list.append(iaa.Sequential([iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2)]))
  seq_list.append(iaa.Sequential([iaa.Invert(0.05, per_channel=True)]))
  seq_list.append(iaa.Sequential([iaa.Add((-10, 10), per_channel=0.5)]))
  seq_list.append(iaa.Sequential([iaa.Multiply((0.5, 1.5), per_channel=0.5)]))
  seq_list.append(iaa.Sequential([iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)]))
  seq_list.append(iaa.Sequential([iaa.Grayscale(alpha=(0.0, 1.0))]))
  seq_list.append(iaa.Sequential([iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)]))
  seq_list.append(iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.05))]))

  return seq_list


if __name__ == "__main__":
  seq_list = build_seqlist()

  if len(sys.argv) != 2:
    print "Assume your dataset is in:"
    print "/home/username/Desktop/dataset/006/00.jpg"
    print "/home/username/Desktop/dataset/006/01.jpg"
    print "..."
    print "Usage: python data_aug.py /home/username/Desktop/dataset"
    sys.exit(0)
  else:
    dataset_loc = sys.argv[1]
    if dataset_loc[-1] == "/":
      dataset_loc = dataset_loc[:-1]
    dataset_root_folder_name = dataset_loc.split("/")[-1]

    aug_dataset_loc = dataset_loc + "_aug"  # make aug root folder
    if not os.path.exists(aug_dataset_loc):
      os.makedirs(aug_dataset_loc)

    sub_folder_list = glob.glob(dataset_loc + "/*/")  # make aug sub folder
    for i in sub_folder_list:
      aug_sub_folder = i.replace(dataset_root_folder_name, dataset_root_folder_name + "_aug")
      if not os.path.exists(aug_sub_folder):
        os.makedirs(aug_sub_folder)

  img_path_list = sorted(glob.glob(dataset_loc + "/*/*.jpg"))

  for i in range(len(img_path_list)):
    print img_path_list[i]
    img = cv2.imread(img_path_list[i])
    #img = cv2.resize(img, (224, 224))
    file_name = img_path_list[i].replace(dataset_root_folder_name, dataset_root_folder_name + "_aug")
    cv2.imwrite(file_name, img)

    for j in range(len(seq_list)):
      img_aug = seq_list[j].augment_images([img])[0]
      aug_name = file_name.replace(".jpg", "_aug" + str(j) + ".jpg")
      cv2.imwrite(aug_name, img_aug)
