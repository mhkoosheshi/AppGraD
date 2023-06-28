import numpy as np
from PIL import Image
import os
import math
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
import random
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
  '''
  Provide RGB or RGB-D data for your model
  '''
  # feeding network with keypoints under construction
  # z-parameter preprocessing under constrution
  # fix bounding box in albumentations
  def __init__(self,
              RGB_paths,
              D_paths,
              grasp_paths,
              batch_size=8,
              camera_mode="vs1",
              input_mode="RGB",
              shape=(224,224),
              param_mode=[1,1,1,1,1],
              bb_norm=False,
              shuffle=True,
              multi_inputs=False
              ):

    self.RGB_paths = RGB_paths
    self.D_paths = D_paths
    self.grasp_paths = grasp_paths
    self.batch_size = batch_size
    self.camera_mode = camera_mode
    self.input_mode = input_mode
    self.shape = shape
    self.param_mode = param_mode
    self.bb_norm = bb_norm
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
              camera_mode="vs1",
              input_mode="RGB",
              shape=(224,224),
              param_mode=[1,1,1,1,1],
              bb_norm=False,
              shuffle=True,
              multi_inputs=False
              ):

    self.RGB1_paths = RGB1_paths
    self.RGB2_paths = RGB2_paths
    self.grasp_paths = grasp_paths
    self.batch_size = batch_size
    self.camera_mode = camera_mode
    self.input_mode = input_mode
    self.shape = shape
    self.param_mode = param_mode
    self.bb_norm = bb_norm
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


def get_loader(batch_size=8,
              camera_mode=camera_mode,
              input_mode=input_mode,
              shape=shape,
              param_mode=param_mode,
              bb_norm=bb_norm,
              shuffle=True,
              factor=factor)
  

  factor = 0.15
  RGB_paths, D_paths, grasp_paths = path_lists(camera_mode = "vs1")
  n = len(RGB_paths)
  RGB_paths, D_paths, grasp_paths = np.array(RGB_paths), np.array(D_paths), np.array(grasp_paths)
  RGB_paths, D_paths, grasp_paths = unison_shuffle(RGB_paths,D_paths,grasp_paths)
  RGB_paths, D_paths, grasp_paths = list(RGB_paths), list(D_paths), list(grasp_paths)
  RGB_train, RGB_val = RGB_paths[int(n*factor):], RGB_paths[:int(n*factor)]
  D_train, D_val = D_paths[int(n*factor):], D_paths[:int(n*factor)]
  grasp_train, grasp_val = grasp_paths[int(n*factor):], grasp_paths[:int(n*factor)]

  train_gen = DataGenerator(RGB_train, D_train, grasp_train, multi_inputs=True)
  val_gen = DataGenerator(RGB_val, D_val, grasp_val, multi_inputs=True)





  factor = 0.15
  RGB1_paths, RGB2_paths, grasp_paths = path_lists(camera_mode = "vs5", branches=branches)
  n = len(RGB1_paths)
  RGB1_paths, RGB2_paths, grasp_paths = np.array(RGB1_paths), np.array(RGB2_paths), np.array(grasp_paths)
  RGB1_paths, RGB2_paths, grasp_paths = unison_shuffle(RGB1_paths, RGB2_paths, grasp_paths)
  RGB1_paths, RGB2_paths, grasp_paths = list(RGB1_paths), list(RGB2_paths), list(grasp_paths)
  RGB1_train, RGB1_val = RGB1_paths[int(n*factor):], RGB1_paths[:int(n*factor)]
  RGB2_train, RGB2_val = RGB2_paths[int(n*factor):], RGB2_paths[:int(n*factor)]
  grasp_train, grasp_val = grasp_paths[int(n*factor):], grasp_paths[:int(n*factor)]

  train_gen = DataGenerator2(RGB1_train, RGB2_train, grasp_train)
  val_gen = DataGenerator2(RGB1_val, RGB2_val, grasp_val)


  return train_gen, val_gen
