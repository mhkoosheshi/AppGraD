import numpy as np
from PIL import Image
import os
import math
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
import random
from tensorflow.keras.utils import Sequence
from utils.preprocess import path_lists, test_path_lists, unison_shuffle, ImageToFloatArray
import albumentations as A

class DataGenerator(Sequence):
  '''
  Provide RGB or RGB-D data for your model
  add one-rgb
  '''
  def __init__(self,
              RGB_paths,
              D_paths,
              grasp_paths,
              batch_size=8,
              shape=(224,224),
              shuffle=True,
              multi_inputs=False
              ):

    self.RGB_paths = RGB_paths
    self.D_paths = D_paths
    self.grasp_paths = grasp_paths
    self.batch_size = batch_size
    self.shape = shape
    self.shuffle = shuffle
    self.multi_inputs = multi_inputs
    self.on_epoch_end()

  def on_epoch_end(self):
    if self.shuffle:
      ind = np.random.permutation(len(self.RGB_paths)).astype(np.int)
      self.RGB_paths, self.D_paths, self.grasp_paths = np.array(self.RGB_paths), np.array(self.D_paths), np.array(self.grasp_paths)
      self.RGB_paths, self.D_paths, self.grasp_paths = self.RGB_paths[ind], self.D_paths[ind], self.grasp_paths[ind]
      self.RGB_paths, self.D_paths, self.grasp_paths = list(self.RGB_paths), list(self.D_paths), list(self.grasp_paths)


  def __len__(self):
    return math.ceil(len(self.RGB_paths) / self.batch_size)


  def __getitem__(self, idx):

    batch_RGB = self.RGB_paths[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_D = self.D_paths[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_grasp = self.grasp_paths[idx * self.batch_size : (idx+1) * self.batch_size]

    rgb = []
    d = []
    grsp = []

    for i, (RGB_path, D_path, grasp_path) in enumerate(zip(batch_RGB, batch_D, batch_grasp)):

      # RGB data
      img = cv2.cvtColor(cv2.imread(RGB_path), cv2.COLOR_BGR2RGB)
      pimg = (Image.fromarray(img)).resize((self.shape[0], self.shape[1]))
      img = np.asarray(pimg)
      img = np.float32(img)
      img = img/255
      rgb.append(img)


      # Depth data
      depth = ImageToFloatArray(D_path)
      pimg = (Image.fromarray(depth)).resize((self.shape[0], self.shape[1]))
      depth = np.asarray(pimg)
      depth = np.float32(depth)
      depth = (depth-2.56286328125)/(19.99090234375-2.56286328125)
      if self.multi_inputs is True:
        depth = np.stack((depth, depth, depth), axis=2)
      d.append(depth)

      # grasp data
      with open(grasp_path,"r") as f:
        s = f.read()
      grasp = [float(s.split(",")[i]) for i in range(0,len(s.split(",")))]
      grasp[0] = (grasp[0]-175)/(355-175)
      grasp[1] = (grasp[1]-200)/(400-200)
      grasp[2] = (grasp[2]-0.01)/(0.08-0.01)
      grasp[3] = (grasp[3]-(-180))/(180-(-180))
      grasp[4] = (grasp[4]-25)/(110-25)
      grsp.append(grasp)

      # if grasp[3]<0:
      #   grasp[3] = 360 + grasp[3]
      # sin cos

    rgb = (np.array(rgb))
    d = np.array(d)
    grsp = np.array(grsp)

    if self.multi_inputs is False:
      return rgb, grsp

    elif self.multi_inputs is True:
      return [rgb,d], [grsp]

class DataGenerator2(Sequence):
  '''
  provide your model with batches of inputs and outputs with keras.utils.sequence

  two branches of RGB inputs for sided cameras
  '''
  def __init__(self,
              RGB1_paths,
              RGB2_paths,
              grasp_paths,
              batch_size=8,
              shape=(224,224),
              shuffle=True
              ):

    self.RGB1_paths = RGB1_paths
    self.RGB2_paths = RGB2_paths
    self.grasp_paths = grasp_paths
    self.batch_size = batch_size
    self.shape = shape
    self.shuffle = shuffle
    self.on_epoch_end()

  def on_epoch_end(self):
    if self.shuffle:
      ind = np.random.permutation(len(self.RGB1_paths)).astype(np.int64)
      self.RGB1_paths, self.RGB2_paths, self.grasp_paths = np.array(self.RGB1_paths), np.array(self.RGB2_paths), np.array(self.grasp_paths)
      self.RGB1_paths, self.RGB2_paths, self.grasp_paths = self.RGB1_paths[ind], self.RGB2_paths[ind], self.grasp_paths[ind]
      self.RGB1_paths, self.RGB2_paths, self.grasp_paths = list(self.RGB1_paths), list(self.RGB2_paths), list(self.grasp_paths)


  def __len__(self):
    return math.ceil(len(self.RGB1_paths) / self.batch_size)


  def __getitem__(self, idx):

    batch_RGB1 = self.RGB1_paths[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_RGB2 = self.RGB2_paths[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_grasp = self.grasp_paths[idx * self.batch_size : (idx+1) * self.batch_size]

    rgb1 = []
    rgb2 = []
    grsp = []

    for i, (RGB1_path, RGB2_path, grasp_path) in enumerate(zip(batch_RGB1, batch_RGB2, batch_grasp)):

      # RGB1 data
      img = cv2.cvtColor(cv2.imread(RGB1_path), cv2.COLOR_BGR2RGB)
      pimg = (Image.fromarray(img)).resize((self.shape[0], self.shape[1]))
      img = np.asarray(pimg)
      img = np.float32(img)
      img = img/255
      rgb1.append(img)


      # RGB2 data
      img = cv2.cvtColor(cv2.imread(RGB2_path), cv2.COLOR_BGR2RGB)
      pimg = (Image.fromarray(img)).resize((self.shape[0], self.shape[1]))
      img = np.asarray(pimg)
      img = np.float32(img)
      img = img/255
      rgb2.append(img)

      # grasp data
      with open(grasp_path,"r") as f:
        s = f.read()
      grasp = [float(s.split(",")[i]) for i in range(0,len(s.split(",")))]
      grasp[0] = (grasp[0]-175)/(355-175)
      grasp[1] = (grasp[1]-200)/(400-200)
      grasp[2] = (grasp[2]-0.01)/(0.08-0.01)
      grasp[3] = (grasp[3]-(-180))/(180-(-180))
      grasp[4] = (grasp[4]-25)/(110-25)
      grsp.append(grasp)

      # if grasp[3]<0:
      #   grasp[3] = 360 + grasp[3]
      # sin cos

    rgb1 = (np.array(rgb1))
    rgb2 = (np.array(rgb2))
    grsp = np.array(grsp)

    return [rgb1,rgb2], [grsp]

class DataGenerator3(Sequence):
  '''
  provide your model with batches of inputs and outputs with keras.utils.sequence

  two branches of RGB inputs for sided cameras
  '''
  def __init__(self,
              RGB1_paths,
              RGB2_paths,
              RGB3_paths,
              grasp_paths,
              batch_size=8,
              shape=(224,224),
              shuffle=True,
              aug_p=0.7,
              iso_p=0.8,
              noise_p=0.5,
              others_p=0.5):

    self.RGB1_paths = RGB1_paths
    self.RGB2_paths = RGB2_paths
    self.RGB3_paths = RGB3_paths
    self.grasp_paths = grasp_paths
    self.batch_size = batch_size
    self.shape = shape
    self.shuffle = shuffle
    self.aug_p = aug_p
    self.iso_p = iso_p
    self.noise_p = noise_p
    self.others_p = others_p
    self.on_epoch_end()

    self.noise = A.Compose([
      A.GaussNoise(p=0.5),
      A.MultiplicativeNoise(p=0.5),
    ], p=noise_p)

    self.others = A.Compose([
      A.RandomBrightness(p=0.5),
      # A.FancyPCA(p=0.3),
      A.RandomShadow(p=0.2, shadow_roi=(0, 0.7, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=4),
      A.RandomToneCurve(p=0.3),
      A.Solarize(threshold=50, p=0.5),
      A.PixelDropout(drop_value=0, dropout_prob=0.02, p=0.5),
      # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50, p=0.7)
    ], p=others_p) 

    self.color_transform = A.Compose([
    A.ISONoise(p=iso_p, color_shift=(0.01, 0.05), intensity=(0.2, 0.5)),
    self.others,
    self.noise
    ], p=aug_p)

  def on_epoch_end(self):
    if self.shuffle:
      ind = np.random.permutation(len(self.RGB1_paths)).astype(np.int64)
      self.RGB1_paths, self.RGB2_paths, self.RGB3_paths, self.grasp_paths = np.array(self.RGB1_paths), np.array(self.RGB2_paths), np.array(self.RGB3_paths), np.array(self.grasp_paths)
      self.RGB1_paths, self.RGB2_paths, self.RGB3_paths, self.grasp_paths = self.RGB1_paths[ind], self.RGB2_paths[ind], self.RGB3_paths[ind], self.grasp_paths[ind]
      self.RGB1_paths, self.RGB2_paths, self.RGB3_paths, self.grasp_paths = list(self.RGB1_paths), list(self.RGB2_paths), list(self.RGB3_paths), list(self.grasp_paths)


  def __len__(self):
    return math.ceil(len(self.RGB1_paths) / self.batch_size)


  def __getitem__(self, idx):

    batch_RGB1 = self.RGB1_paths[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_RGB2 = self.RGB2_paths[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_RGB3 = self.RGB3_paths[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_grasp = self.grasp_paths[idx * self.batch_size : (idx+1) * self.batch_size]

    rgb1 = []
    rgb2 = []
    rgb3 = []
    grsp = []

    for i, (RGB1_path, RGB2_path, RGB3_path, grasp_path) in enumerate(zip(batch_RGB1, batch_RGB2, batch_RGB3, batch_grasp)):
      
      
      
      # RGB1 data
      img = cv2.cvtColor(cv2.imread(RGB1_path), cv2.COLOR_BGR2RGB)
      pimg = (Image.fromarray(img)).resize((self.shape[0], self.shape[1]))
      img = np.asarray(pimg)
      # img = np.float32(img)
      img = img

      if self.aug_p !=0:
        rnd = random.randint(1,2)
        rnd = rnd - 1
        img = (rnd)*(255 - img) + (1-rnd)*img
        a = int(100*(random.random()))
        random.seed(a)
        transformed = self.color_transform(image=img)['image']
        img = transformed

      if self.aug_p !=0:
        rnd = random.randint(1,2)
        rnd = 2
        rnd = rnd - 1
        img = (rnd)*(255 - img) + (1-rnd)*img

      # img = np.float32(img)
      rgb1.append(img)


      # RGB2 data
      img = cv2.cvtColor(cv2.imread(RGB2_path), cv2.COLOR_BGR2RGB)
      pimg = (Image.fromarray(img)).resize((self.shape[0], self.shape[1]))
      img = np.asarray(pimg)
      # img = np.float32(img)
      img = img
      
      if self.aug_p !=0:
        rnd = random.randint(1,2)
        rnd = rnd - 1
        img = (rnd)*(255 - img) + (1-rnd)*img
        random.seed(a)
        transformed = self.color_transform(image=img)['image']
        img = transformed

      if self.aug_p !=0:
        rnd = random.randint(1,2)
        rnd = 2
        rnd = rnd - 1
        img = (rnd)*(255 - img) + (1-rnd)*img

      # img = np.float32(img)
      rgb2.append(img)

      # RGB3 data
      img = cv2.cvtColor(cv2.imread(RGB3_path), cv2.COLOR_BGR2RGB)
      pimg = (Image.fromarray(img)).resize((self.shape[0], self.shape[1]))
      img = np.asarray(pimg)
      # img = np.float32(img)
      img = img
      
      if self.aug_p !=0:
        rnd = random.randint(1,2)
        rnd = rnd - 1
        img = (rnd)*(255 - img) + (1-rnd)*img
        random.seed(a)
        transformed = self.color_transform(image=img)['image']
        img = transformed

      if self.aug_p !=0:
        rnd = random.randint(1,2)
        rnd = 2
        rnd = rnd - 1
        img = (rnd)*(255 - img) + (1-rnd)*img

      # img = np.float32(img)
      rgb3.append(img)

      # grasp data
      with open(grasp_path,"r") as f:
        s = f.read()
      grasp = [float(s.split(",")[i]) for i in range(0,len(s.split(",")))]
      # grasp[0] = (grasp[0]-155)/(355-155)
      # grasp[1] = (grasp[1]-185)/(410-185)
      grasp[0] = (((0.5*grasp[0])/512 - 0.125)/0.25)
      grasp[1] = (((0.5*grasp[1])/512 -0.0442 - 0.125)/0.25)
      grasp[2] = (grasp[2]-0.01)/(0.08-0.01)
      grasp[4] = (grasp[4]-25)/(105-25)
      if grasp[3]<0:
        grasp[3] = 270 + grasp[3]
      elif grasp[3]>0:
        grasp[3] = -90 + grasp[3]
      grasp[3]=grasp[3]/180
      
      grsp.append(grasp)
      
      # print(a)



    rgb1 = (np.array(rgb1))/255
    rgb2 = (np.array(rgb2))/255
    rgb3 = (np.array(rgb3))/255
    grsp = np.array(grsp)

    return [rgb1, rgb2, rgb3], [grsp]


def get_loader(batch_size=8,
              branches='one',
              shape=(224,224),
              shuffle=True,
              factor=0.15,
              aug=False,
              aug_p=0,
              iso_p=0.8,
              noise_p=0.5,
              others_p=0.5,
              val_aug_p=0):
  

  if branches=='one' or branches=='two_rgbd':
    RGB_paths, D_paths, grasp_paths = path_lists(branches=branches)
    n = len(RGB_paths)
    RGB_paths, D_paths, grasp_paths = np.array(RGB_paths), np.array(D_paths), np.array(grasp_paths)
    RGB_paths, D_paths, grasp_paths = unison_shuffle(a=RGB_paths,b=D_paths,c=grasp_paths)
    RGB_paths, D_paths, grasp_paths = list(RGB_paths), list(D_paths), list(grasp_paths)
    RGB_train, RGB_val = RGB_paths[int(n*factor):], RGB_paths[:int(n*factor)]
    D_train, D_val = D_paths[int(n*factor):], D_paths[:int(n*factor)]
    grasp_train, grasp_val = grasp_paths[int(n*factor):], grasp_paths[:int(n*factor)]

    if branches=='one':
      train_gen = DataGenerator(RGB_train,
                                D_train,
                                grasp_train, 
                                batch_size=batch_size,
                                shape=shape,
                                shuffle=shuffle,
                                multi_inputs=False)
      val_gen = DataGenerator(RGB_val,
                              D_val,
                              grasp_val, 
                              batch_size=batch_size,
                              shape=shape,
                              shuffle=shuffle,
                              multi_inputs=False)      
    elif branches=='two_rgbd':
      train_gen = DataGenerator(RGB_train,
                                D_train,
                                grasp_train, 
                                batch_size=batch_size,
                                shape=shape,
                                shuffle=shuffle,
                                multi_inputs=True)
      val_gen = DataGenerator(RGB_val,
                              D_val,
                              grasp_val, 
                              batch_size=batch_size,
                              shape=shape,
                              shuffle=shuffle,
                              multi_inputs=True)
                            
  elif branches=='two_rgb':
    RGB1_paths, RGB2_paths, grasp_paths = path_lists(branches=branches)
    n = len(RGB1_paths)
    RGB1_paths, RGB2_paths, grasp_paths = np.array(RGB1_paths), np.array(RGB2_paths), np.array(RGB3_paths), np.array(grasp_paths)
    RGB1_paths, RGB2_paths, grasp_paths = unison_shuffle(a=RGB1_paths, b=RGB2_paths, c=grasp_paths)
    RGB1_paths, RGB2_paths, grasp_paths = list(RGB1_paths), list(RGB2_paths), list(grasp_paths)
    RGB1_train, RGB1_val = RGB1_paths[int(n*factor):], RGB1_paths[:int(n*factor)]
    RGB2_train, RGB2_val = RGB2_paths[int(n*factor):], RGB2_paths[:int(n*factor)]
    grasp_train, grasp_val = grasp_paths[int(n*factor):], grasp_paths[:int(n*factor)]

    train_gen = DataGenerator2(RGB1_train,
                               RGB2_train, 
                               grasp_train,
                               batch_size=batch_size,
                               shape=shape,
                               shuffle=shuffle
                               )
    val_gen = DataGenerator2(RGB1_val,
                            RGB2_val, 
                            grasp_val,
                            batch_size=batch_size,
                            shape=shape,
                            shuffle=shuffle
                            )
  elif branches=='three_rgb':
    RGB1_paths, RGB2_paths, RGB3_paths, grasp_paths = path_lists(branches=branches)
    n = len(RGB1_paths)
    RGB1_paths, RGB2_paths, RGB3_paths, grasp_paths = np.array(RGB1_paths), np.array(RGB2_paths), np.array(RGB3_paths), np.array(grasp_paths)
    RGB1_paths, RGB2_paths, RGB3_paths, grasp_paths = unison_shuffle(a=RGB1_paths, b=RGB2_paths, c=RGB3_paths, d=grasp_paths)
    RGB1_paths, RGB2_paths, RGB3_paths, grasp_paths = list(RGB1_paths), list(RGB2_paths), list(RGB3_paths), list(grasp_paths)
    RGB1_train, RGB1_val = RGB1_paths[int(n*factor):], RGB1_paths[:int(n*factor)]
    RGB2_train, RGB2_val = RGB2_paths[int(n*factor):], RGB2_paths[:int(n*factor)]
    RGB3_train, RGB3_val = RGB3_paths[int(n*factor):], RGB3_paths[:int(n*factor)]
    grasp_train, grasp_val = grasp_paths[int(n*factor):], grasp_paths[:int(n*factor)]
    
    if aug:
      RGB1_train, RGB2_train, RGB3_train, grasp_train = 2*RGB1_train, 2*RGB2_train, 2*RGB3_train, 2*grasp_train
    
    RGB1_test, RGB2_test, RGB3_test, grasp_test = test_path_lists(branches=branches)

    train_gen = DataGenerator3(RGB1_train,
                               RGB2_train, 
                               RGB3_train,
                               grasp_train,
                               batch_size=batch_size,
                               shape=shape,
                               shuffle=shuffle,
                               aug_p=aug_p,
                               iso_p=0.8,
                               noise_p=0.5,
                               others_p=0.5
                               )
    val_gen = DataGenerator3(RGB1_val,
                            RGB2_val, 
                            RGB3_val,
                            grasp_val,
                            batch_size=batch_size,
                            shape=shape,
                            shuffle=shuffle,
                            aug_p=val_aug_p,
                            iso_p=0.8,
                            noise_p=0.5,
                            others_p=0.5
                            )

    test_gen = DataGenerator3(RGB1_test,
                              RGB2_test, 
                              RGB3_test,
                              grasp_test,
                              batch_size=batch_size,
                              shape=shape,
                              shuffle=False,
                              aug_p=0,
                              iso_p=0,
                              noise_p=0,
                              others_p=0
                              )
  # elif branches=='three_d':
  #   RGB_paths, D_paths, grasp_paths = path_lists(branches=branches)
  #   n = len(RGB_paths)
  #   RGB_paths, D_paths, grasp_paths = np.array(RGB_paths), np.array(D_paths), np.array(grasp_paths)
  #   RGB_paths, D_paths, grasp_paths = unison_shuffle(RGB_paths,D_paths,grasp_paths)
  #   RGB_paths, D_paths, grasp_paths = list(RGB_paths), list(D_paths), list(grasp_paths)
  #   RGB_train, RGB_val = RGB_paths[int(n*factor):], RGB_paths[:int(n*factor)]
  #   D_train, D_val = D_paths[int(n*factor):], D_paths[:int(n*factor)]
  #   grasp_train, grasp_val = grasp_paths[int(n*factor):], grasp_paths[:int(n*factor)]

  #   train_gen = DataGenerator(RGB_train, D_train, grasp_train, multi_inputs=True)
  #   val_gen = DataGenerator(RGB_val, D_val, grasp_val, multi_inputs=True)

  return train_gen, val_gen, test_gen
