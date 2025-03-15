import numpy as np
from PIL import Image
import glob
import os
import cv2
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from google.colab.patches import cv2_imshow
from shapely.geometry import Polygon
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import datetime
import os
import time
from tensorflow.keras.models import load_model

def output_postprocess(pred, shape=(224,224)):

  pred_ = [0, 0, 0, 0, 0]
  pred_[0] = (355-155)*pred[0] + 155
  pred_[1] = (410-185)*pred[1] + 185
  pred_[2] = (0.08-0.01)*pred[2] + 0.01
  pred_[3] = (180)*pred[3] + 0
  pred_[4] = (105-25)*pred[4] + 25
  if pred_[3]<90:
      pred_[3] = pred_[3] + 90
  elif pred_[3]>90:
      pred_[3] = pred_[3] -270

  return pred_

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def trans(x, shift_x, shift_y):
    # shift_x = -20
    # shift_y = 650
    for i in range(x.shape[1] -1, shift_x, -1):
        x = np.roll(x, -1, axis=1)
        x[:, -1] = x[:, 0]

    return x

def rect2points(rectangle, shape=(512,512)):

    [x, y, z, t, w] = rectangle

    # now convert this to RANGE
    y_c = (((0.5*y)/512 -0.0442 - 0.125)/0.25)*shape[1]
    x_c = (((0.5*x)/512 - 0.125)/0.25)*shape[0]
    # x_c = x*shape[0]
    # y_c = y*shape[1]
    t = np.deg2rad(90-t)
    l = 15
    # w = w/2

    l = (l*shape[0]/512) *2
    w = (w*shape[0]/512)

    R1 = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    R2 = np.array([[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]])
    xp_c = (np.matmul(R1, np.array([[x_c],[y_c]])))[0][0]
    yp_c = (np.matmul(R1, np.array([[x_c],[y_c]])))[1][0]

    xp_1 = xp_c + l
    yp_1 = yp_c - w
    xp_2 = xp_1
    yp_2 = yp_c + w
    xp_3 = xp_c - l
    yp_3 = yp_2
    xp_4 = xp_3
    yp_4 = yp_1

    x1 = int((np.matmul(R2, np.array([[xp_1],[yp_1]])))[0][0])
    y1 = int((np.matmul(R2, np.array([[xp_1],[yp_1]])))[1][0])
    x2 = int((np.matmul(R2, np.array([[xp_2],[yp_2]])))[0][0])
    y2 = int((np.matmul(R2, np.array([[xp_2],[yp_2]])))[1][0])
    x3 = int((np.matmul(R2, np.array([[xp_3],[yp_3]])))[0][0])
    y3 = int((np.matmul(R2, np.array([[xp_3],[yp_3]])))[1][0])
    x4 = int((np.matmul(R2, np.array([[xp_4],[yp_4]])))[0][0])
    y4 = int((np.matmul(R2, np.array([[xp_4],[yp_4]])))[1][0])

    point1 = (x1,y1)
    point2 = (x2,y2)
    point3 = (x3,y3)
    point4 = (x4,y4)

    return point1, point2, point3, point4

def real_test(obj, i):

  main_path = '/content/drive/MyDrive/AppGraD/sim2real/new2'
  # main_path = '/content/drive/MyDrive/AppGraD'  
  # rgb1 = main_path + f"/vs10/rgb1016.png"
  # rgb2 = main_path + f"/vs11/rgb1016.png"
  # rgb3 = main_path + f"/vs12/rgb1016.png"
  
  rgb1 = main_path + f"/vs10_{obj}{i}.jpg"
  rgb2 = main_path + f"/vs11_{obj}{i}.jpg"
  rgb3 = main_path + f"/vs12_{obj}{i}.jpg"
  
  # img1 = np.random.rand(1,224,224,3)
  # img2 = np.random.rand(1,224,224,3)
  # img3 = np.random.rand(1,224,224,3)
  
  img1 = cv2.imread(rgb1)
  img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
  # mid_x, mid_y = int(1920/2), int(1080/2)
  # cw2, ch2 = int(1080/2), int(1080/2)
  # img1 = img1[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
  pimg = (Image.fromarray(img1)).resize((224,224))
  img1 = np.asarray(pimg)/255
  img1 = img1.reshape((1,224,224,3))
  # img1 = 1 - img1
  img1 = img1.astype(np.float32)
  
  img2 = cv2.imread(rgb2)
  img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
  # mid_x, mid_y = int(1920/2), int(1080/2)
  # cw2, ch2 = int(1080/2), int(1080/2)
  # img2 = img2[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
  pimg = (Image.fromarray(img2)).resize((224,224))
  img2 = np.asarray(pimg)/255
  img2 = img2.reshape((1,224,224,3))
  # img2 = 1 - img2
  img2 = img2.astype(np.float32)
  
  img3 = cv2.imread(rgb3)
  img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
  # mid_x, mid_y = int(1920/2), int(1080/2)
  # cw2, ch2 = int(1080/2), int(1080/2)
  # img3 = img3[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
  img3 = (Image.fromarray(img3)).resize((224,224))
  img3 = np.asarray(img3)/255
  img3 = img3.reshape((1,224,224,3))
  # img3 = 1 - img3
  img3 = img3.astype(np.float32)

  input = [img1, img2, img3]
  # model_path = "/content/drive/MyDrive/weights_mohokoo/sim2real_vgg16s2dsepconv_hardaug_20230831-042158.h5"
  model_path = "/content/drive/MyDrive/weights_mohokoo/a/vgg16s2dsepconv_20230729-080924.h5"
  model = load_model(model_path, custom_objects={"Huber": tf.keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss"), "logcosh": tf.keras.metrics.LogCoshError() })
  
  output = model.predict(input)
  output = output.tolist()
  output = output[0]

  output = output_postprocess(output, (224,224))
  p1, p2, p3, p4 = rect2points(output, (224,224))
  # pr, pl = [0,0], [0,0]
  # p1, p2, p3, p4 = list(p1), list(p2), list(p3), list(p4)
  # pl[0] = (p1[0]+p4[0])/2
  # pl[1] = (p1[1]+p4[1])/2
  # pr[0] = (p3[0]+p2[0])/2
  # pr[1] = (p3[1]+p2[1])/2
  # output_points = [tuple(pr), tuple(pl)]
  
  color1 = (0, 0, 255)
  color2 = (255, 0, 0)
  thickness = 2
  
  x = cv2.imread(main_path + f"/RANGE_{obj}{i}.jpg")
  x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
  x = cv2.resize(x, (224,224))
  
  image = cv2.line(x, p1, p2, color1, thickness)
  image = cv2.line(image, p3, p4, color1, thickness)
  image = cv2.line(image, p2, p3, color2, thickness)
  image = cv2.line(image, p4, p1, color2, thickness)
  # x = np.random.rand(3024,3024,3)
  # x = np.stack([x, x, x], axis=-1)
  # x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
  # # x = x.reshape((224,224))
  # x = rotate_image(x, -7)
  # x = trans(x, -20, 0)
  
  # x = cv2.circle(x, (int(output_points[0][0]),int(output_points[0][1])), radius=5, color=(255, 0, 0), thickness=30)
  # x = cv2.circle(x, (int(output_points[1][0]),int(output_points[1][1])), radius=5, color=(255, 0, 0), thickness=30)
  
  # x = cv2.resize(x, (224,224))
  plt.axis('off')
  plt.imshow(image)
  plt.savefig(main_path + f"/Results/RANGE_{obj}{i}.jpg")


if __name__=="__main__":

  objects = ['anbor', 'bowl', 'box', 'cable', 'cup', 'L', 'ring', 'stapler', 'watch', 'vase', 'screw']
  freq = [3, 2, 3, 3, 3, 3, 3, 3, 2, 4, 3]

  for obj, f in zip(objects, freq):
    for i in range(1, f+1):
      real_test(obj, i)
