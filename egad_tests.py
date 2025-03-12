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
from utils.data_loader import get_loader
# import osgeo
# import rasterio
# import rasterio.features
import seaborn as sns
# from keras.models import load_model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import model_from_json
import pandas as pd

def output_postprocess(pred, shape=(224,224)):

  pred_ = np.zeros(pred.shape)
  # pred_[:, 0] = (355-155)*pred[:, 0] + 155
  # pred_[:, 1] = (410-185)*pred[:, 1] + 185
  pred_[:, 0] = pred[:, 0]
  pred_[:, 1] = pred[:, 1]
  pred_[:, 2] = (0.08-0.01)*pred[:, 2] + 0.01
  pred_[:, 3] = (180)*pred[:, 3] + 0
  pred_[:, 4] = ((105-25)*pred[:, 4] + 25)*1.05

  for i in range(pred.shape[0]):
    if pred_[i, 3]<90:
      pred_[i, 3] = pred_[i, 3] + 90
    elif pred_[i, 3]>90:
      pred_[i, 3] = pred_[i, 3] -270

  return pred_

def rect2points(rectangle, shape=(512,512)):

    [x, y, z, t, w] = rectangle

    # now convert this to RANGE
    # y_c = (((0.5*y)/512 -0.0442 - 0.125)/0.25)*shape[1]
    # x_c = (((0.5*x)/512 - 0.125)/0.25)*shape[0]
    x_c = x*shape[0]
    y_c = y*shape[1]
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

    x1 = (np.matmul(R2, np.array([[xp_1],[yp_1]])))[0][0]
    y1 = (np.matmul(R2, np.array([[xp_1],[yp_1]])))[1][0]
    x2 = (np.matmul(R2, np.array([[xp_2],[yp_2]])))[0][0]
    y2 = (np.matmul(R2, np.array([[xp_2],[yp_2]])))[1][0]
    x3 = (np.matmul(R2, np.array([[xp_3],[yp_3]])))[0][0]
    y3 = (np.matmul(R2, np.array([[xp_3],[yp_3]])))[1][0]
    x4 = (np.matmul(R2, np.array([[xp_4],[yp_4]])))[0][0]
    y4 = (np.matmul(R2, np.array([[xp_4],[yp_4]])))[1][0]

    point1 = (x1,y1)
    point2 = (x2,y2)
    point3 = (x3,y3)
    point4 = (x4,y4)

    return point1, point2, point3, point4

# def render_mask(array, shape=(512,512)):
#     """
#     output an array of masks. input an array with rectangles
#     """
#     masks = []
#     for i in range(0,array.shape[0]):
#       rectangle = array[i,:]
#       rectangle = list(rectangle)
#       point1, point2, point3, point4 = rect2points(rectangle, shape=shape)
#       poly = Polygon([point1, point2, point3, point4])
#       mask = rasterio.features.rasterize([poly], out_shape=shape)
#       masks.append(mask)
#     masks = np.array(masks)
#     return masks

def render_mask(array, shape=(512,512)):
    """
    Generate an array of masks from an input array of rectangles.
    """
    masks = np.zeros((array.shape[0], *shape), dtype=np.uint8)

    for i in range(array.shape[0]):
        rectangle = array[i, :]
        point1, point2, point3, point4 = rect2points(rectangle, shape=shape)

        # Create a blank mask
        mask = np.zeros(shape, dtype=np.uint8)

        # Define the polygon and fill it
        pts = np.array([point1, point2, point3, point4], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)

        masks[i] = mask

    return masks

def jaccard(pred_masks, test_masks, shape=(512,512)):

  jaccard = []
  for i in range(pred_masks.shape[0]):
    rectangle = pred_masks[i]
    ground_truth = test_masks[i]
    axis=(0,1)
    smooth = 1e-5
    inse = np.sum(rectangle * ground_truth, axis=axis)
    l = np.sum(rectangle * rectangle, axis=axis)
    r = np.sum(ground_truth * ground_truth, axis=axis)
    j = (inse + smooth) / (l + r - inse + smooth)
    jaccard.append(j)
  jaccard = np.array(jaccard)

  return jaccard # an array with IoU in each cell for the samples

def score(jaccards, absangles):
  scores = []
  for i in range(0,jaccards.shape[0]):
    if jaccards[i,0]>=0.25 and absangles[i,0]<=30:
      scores.append(1)
    else:
      scores.append(0)

  scores = np.array(scores)

  return scores

def plot_predictions(images, pred, test, scores, shape=(512,512), plot_mode='points'):
  '''
  images: RANGE
  '''

  # works on batches
  columns = 5
  rows = 3
  for i in range(pred.shape[0]):

    pred_rect = list(pred[i,:])
    test_rect = list(test[i,:])

    pred_points = rect2points(pred_rect, shape=shape)
    test_points = rect2points(test_rect, shape=shape)

    p1, p2, p3, p4 = pred_points
    point1, point2, point3, point4 = (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), (int(p4[0]), int(p4[1]))
    pr, pl = [0,0], [0,0]
    p1, p2, p3, p4 = list(p1), list(p2), list(p3), list(p4)
    pl[0] = (p1[0]+p4[0])/2
    pl[1] = (p1[1]+p4[1])/2
    pr[0] = (p3[0]+p2[0])/2
    pr[1] = (p3[1]+p2[1])/2
    pred_points = [tuple(pr), tuple(pl)]

    p1, p2, p3, p4 = test_points
    point1_, point2_, point3_, point4_ = (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), (int(p4[0]), int(p4[1]))
    pr, pl = [0,0], [0,0]
    p1, p2, p3, p4 = list(p1), list(p2), list(p3), list(p4)
    pl[0] = (p1[0]+p4[0])/2
    pl[1] = (p1[1]+p4[1])/2
    pr[0] = (p3[0]+p2[0])/2
    pr[1] = (p3[1]+p2[1])/2
    test_points = [tuple(pr), tuple(pl)]

    im = images[i]
    img = im.copy()
    # img = im
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if plot_mode == 'points':
      img = cv2.circle(img, (int(pred_points[0][0]),int(pred_points[0][1])), radius=5, color=(0, 0, 255), thickness=-1)
      img = cv2.circle(img, (int(pred_points[1][0]),int(pred_points[1][1])), radius=5, color=(0, 0, 255), thickness=-1)
      img = cv2.circle(img, (int(test_points[0][0]),int(test_points[0][1])), radius=5, color=(255, 255, 255), thickness=-1)
      img = cv2.circle(img, (int(test_points[1][0]),int(test_points[1][1])), radius=5, color=(255, 255, 255), thickness=-1)

    elif plot_mode == 'rectangles':
      image = img
      color1 = (255, 0, 0)
      color2 = (0, 0, 255)

      image = cv2.line(image, point1_, point2_, (0, 0, 0), thickness=1)
      image = cv2.line(image, point3_, point4_, (0, 0, 0), thickness=1)
      image = cv2.line(image, point2_, point3_, (0, 0, 0), thickness=1)
      image = cv2.line(image, point4_, point1_, (0, 0, 0), thickness=1)

      image = cv2.line(image, point1, point2, color1, thickness=1)
      image = cv2.line(image, point3, point4, color1, thickness=1)
      image = cv2.line(image, point2, point3, color2, thickness=1)
      image = cv2.line(image, point4, point1, color2, thickness=1)

    elif plot_mode == 'lines':
      img = cv2.circle(img, (int(pred_points[0][0]),int(pred_points[0][1])), radius=5, color=(0, 0, 255), thickness=-1)
      img = cv2.circle(img, (int(pred_points[1][0]),int(pred_points[1][1])), radius=5, color=(0, 0, 255), thickness=-1)
      img = cv2.circle(img, (int(test_points[0][0]),int(test_points[0][1])), radius=5, color=(255, 255, 255), thickness=-1)
      img = cv2.circle(img, (int(test_points[1][0]),int(test_points[1][1])), radius=5, color=(255, 255, 255), thickness=-1)
      img = cv2.line(img, (int(pred_points[0][0]), int(pred_points[0][1])), (int(pred_points[1][0]),int(pred_points[1][1])), color=(0, 0, 255), thickness=-1)
      img = cv2.line(img, (int(test_points[0][0]),int(test_points[0][1])), (int(test_points[1][0]),int(test_points[1][1])), color=(255, 255, 255), thickness=-1)


    fig = plt.figure(figsize=(25,9))

    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    string = scores[i]*'success' + np.abs(scores[i]-1)*'failure'
    plt.title(f"{i+1}, {string}")
    plt.imshow(img)
    plt.savefig(f'{i}.png')
    # plt.savefig(f'egad/{i}.png')

  # plt.show()

def grasp_success_rate(model_path,
                       test_gen,
                       RANGE_imgs,
                       shape=(224,224),
                       plot_size=(224,224),
                       batch_size=10,
                       plot_mode='points'
                       ):
  h = 49*batch_size

  model = tf.keras.models.load_model(
                                      model_path,
                                      custom_objects={
                                          "Huber": tf.keras.losses.Huber,
                                          "logcosh": tf.keras.losses.LogCosh
                                      },
                                      compile=False  # safer first step
                                  )
  # model = load_model(model_path, compile=False)
  # model.load_weights(model_path)

  # model = keras.layers.TFSMLayer('/content/drive/MyDrive/a', call_endpoint='serving_default')
  pred_masks = [] # batch
  test_masks = [] # batch
  jaccards = np.zeros((h,1)) # total
  absangles = np.zeros((h,1)) #total
  scores_b = []
  scores = np.zeros((h,1)) # total
  scores_table = np.zeros((9,1))

  for i in range(0,test_gen.__len__()):

    pred_b = output_postprocess(model.predict(test_gen.__getitem__(i)[0][:][:]))
    test_b = output_postprocess(test_gen.__getitem__(i)[1][0][:])
    RANGE_imgs_b = RANGE_imgs[(i)*batch_size:(i+1)*batch_size]

    pred_masks = render_mask(pred_b, shape=shape)
    test_masks = render_mask(test_b, shape=shape)

    scores_b = score(jaccards[(i)*batch_size:(i+1)*batch_size],absangles[(i)*batch_size:(i+1)*batch_size])
    absangles[(i)*batch_size:(i+1)*batch_size] = np.reshape(np.abs(pred_b[:,3]-pred_b[:,3]),(batch_size,1))
    jaccards[(i)*batch_size:(i+1)*batch_size] = np.reshape(jaccard(pred_masks, test_masks, shape=shape),(batch_size,1))
    scores_b = score(jaccards[(i)*batch_size:(i+1)*batch_size],absangles[(i)*batch_size:(i+1)*batch_size])
    scores[(i)*batch_size:(i+1)*batch_size] = np.reshape(score(jaccards[(i)*batch_size:(i+1)*batch_size],absangles[(i)*batch_size:(i+1)*batch_size]),(batch_size,1))

    print(f"Image shape: {RANGE_imgs_b.shape}")
    print(f"Predictions shape: {pred_b.shape}")
    print(f"Test shape: {test_b.shape}")
    plot_predictions(RANGE_imgs_b, pred_b, test_b, scores_b, shape=shape, plot_mode=plot_mode)
    print()
    print(f"Scores = {scores[(i)*batch_size:(i+1)*batch_size]}")
    print(f"Score is {np.mean(scores[(i)*batch_size:(i+1)*batch_size])}")
    print(f"Mean Jaccard is {np.mean(jaccards[(i)*batch_size:(i+1)*batch_size])}")
    print(f"Mean absolute error of Thetta is {np.mean(absangles[(i)*batch_size:(i+1)*batch_size])}")
    print()

    scores_table[i] = np.mean(scores[(i)*batch_size:(i+1)*batch_size])

  scores_table = np.reshape(scores_table, (3,3))
  # sns.heatmap(data=scores_table, annot=True, cmap='Blues')
  print(f"Overall Score is {np.mean(scores_table)}")

  return scores_table, jaccards

def RANGE(shape=(224,224)):
  RANGE_imgs = [] # should be preprocessed
  paths = []
  for im_path in glob.glob('/content/drive/MyDrive/AppGraD/test/RANGE/rgb*.png'):
    paths.append(im_path)

  paths = sorted(paths)

  for im_path in paths:
    img = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
    pimg = (Image.fromarray(img)).resize(shape)
    img = np.asarray(pimg)
    # img = np.float32(img)
    RANGE_imgs.append(img)

  RANGE_imgs = np.array(RANGE_imgs)

  return RANGE_imgs


if __name__=='__main__':
  
  branches = 'three_rgb'
  shape = (224,224)
  factor = 0.15
  epochs = 100
  batch_size = 10
  lr = 1e-4
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.000001)
  early_stop = EarlyStopping(monitor='val_loss', patience=5)

  train_gen, val_gen, test_gen= get_loader(batch_size=batch_size,
                                branches=branches,
                                shape=shape,
                                shuffle=True,
                                factor=0.15)
  
  RANGE_img = RANGE(shape=(224,224))

  # directory_name = 'egad'
  # try:
  #     os.mkdir(directory_name)
  #     print(f"Directory '{directory_name}' created successfully.")
  # except FileExistsError:
  #     print(f"Directory '{directory_name}' already exists.")
  # except PermissionError:
  #     print(f"Permission denied: Unable to create '{directory_name}'.")
  # except Exception as e:
  #     print(f"An error occurred: {e}")


  results = grasp_success_rate(model_path="/content/drive/MyDrive/weights_mohokoo/a/vgg16s2dsepconv_20230729-080924.h5",
                                test_gen=test_gen,
                                RANGE_imgs=RANGE_img,
                                shape=(224,224),
                                plot_size=(224,224),
                                plot_mode = 'rectangles',
                                batch_size=batch_size
                                )
  
