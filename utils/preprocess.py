import numpy as np
import cv2
import glob


def path_lists(camera_mode="up"):
  # fix pers and iso labeling
  list_trainX = []
  list_trainy = []
  list_depthX = []

  for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/{camera_mode}/rgb*.png"):
    list_trainX.append(im_path)

  for c, grasp_path in enumerate(glob.glob("/content/drive/MyDrive/AppGraD/grasp/grasp*.txt")):
    list_trainy.append(grasp_path)

  for d_path in glob.glob(f"/content/drive/MyDrive/AppGraD/{camera_mode}/d*.png"):
    list_depthX.append(d_path)

  list_trainX = sorted(list_trainX)
  list_trainy = sorted(list_trainy)
  list_depthX = sorted(list_depthX)

  return list_trainX, list_depthX, list_trainy

def unison_shuffle(a,b,c):
  inx=np.random.permutation(a.shape[0])
  return a[inx],b[inx],c[inx]

def get_inputs(trainX, depthX, input_mode = "RBD"):

  if input_mode == "RBD":
    trainX = trainX[:,:,:,[0,2,1]]
    trainX[:,:,:,2] = depthX
  elif input_mode == "RGB":
    trainX = trainX
  return trainX

def ImageToFloatArray(path, scale_factor=None):
    DEFAULT_RGB_SCALE_FACTOR = 256000.0
    DEFAULT_GRAY_SCALE_FACTOR = {np.uint8: 100.0,
                                 np.uint16: 1000.0,
                                 np.int32: DEFAULT_RGB_SCALE_FACTOR}
    image = cv2.imread(r"{}".format(path))
    image_array = np.array(image)
    image_dtype = image_array.dtype
    image_shape = image_array.shape

    channels = image_shape[2] if len(image_shape) > 2 else 1
    assert 2 <= len(image_shape) <= 3
    if channels == 3:
        # RGB image needs to be converted to 24 bit integer.
        float_array = np.sum(image_array * [65536, 256, 1], axis=2)
        if scale_factor is None:
            scale_factor = DEFAULT_RGB_SCALE_FACTOR
    else:
        if scale_factor is None:
            scale_factor = DEFAULT_GRAY_SCALE_FACTOR[image_dtype.type]
        float_array = image_array.astype(np.float32)
    scaled_array = float_array / scale_factor
    return scaled_array

def get_params(trainy, bb, param_mode = [1,1,0,1,1], shape=(512,512), bb_norm=False):

  if bb_norm == False:
    # trainy[:, [0,1,4]] = shape[0]*trainy[:, [0,1,4]]/512 # did it while reading .txt files
    trainy[:, [0,1]] = np.round((np.floor(trainy[:, [0,1]]))/shape[0],3)
    trainy[:, 4] = np.round((np.floor(trainy[:, 4]))/(shape[0]*105/512),3)
    trainy[:,2] = trainy[:, 2]/1.43

  else :
    # trainy[:, [0,1,4]] = shape[0]*trainy[:, [0,1,4]]/512 # did it while reading .txt files
    trainy[:,0] = (trainy[:,0]-bb[:,0])/bb[:,2]
    trainy[:,1] = (trainy[:,1]-bb[:,1])/bb[:,3]
    trainy[:, 2] = trainy[:, 2]/1.43
    trainy[:, 4] = np.round((np.floor(trainy[:, 4]))/(shape[0]*105/512),3)
    # trainy[:, [0,1,4]] = np.round((np.floor(trainy[:, [0,1,4]]))/shape[0],2)
    # trainy[:,2] = np.round(trainy[:, 2]/1.43,4)

  y_ = np.zeros((trainy.shape[0],param_mode.count(1)+1))
  t = np.zeros((trainy.shape[0],2))
  t[:,0] = (np.sin(np.deg2rad(np.round(trainy[:,3],1)))+1)/2
  t[:,1] = (np.cos(np.deg2rad(np.round(trainy[:,3],1)))+1)/2
  t = np.round(t, 4)

  if param_mode == [1,1,1,1,1]:
    y_[:,[0,1,2]] = trainy[:,[0,1,2]]
    y_[:,[3,4]] = t[:,:]
    y_[:,[5]] = trainy[:,[4]]


  elif param_mode == [1,1,0,1,1]:
    y_[:,[0,1]] = trainy[:,[0,1]]
    y_[:,[2,3]] = t[:,:]
    y_[:,[4]] = trainy[:,[4]]


  elif param_mode == [1,1,1,1,0]:
    y_[:,[0,1,2]] = trainy[:,[0,1,2]]
    y_[:,[3,4]] = t[:,:]

  elif param_mode == [1,1,0,1,0]:
    y_[:,[0,1]] = trainy[:,[0,1]]
    y_[:,[2,3]] = t[:,:]

  trainy = y_

  return trainy