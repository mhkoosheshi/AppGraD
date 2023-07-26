import numpy as np
import cv2
import glob


def path_lists(branches='one'):
  # fix pers and iso labeling
  list_grasp = []
  list_RGB1 = []
  list_D1 = []
  list_RGB2 = []
  list_D2 = []
  list_RGB3 = []
  list_D3 = []

  if branches=='one' or branches=='two_rgbd':
    for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/vs1/rgb*.png"):
      list_RGB1.append(im_path)

    for c, grasp_path in enumerate(glob.glob("/content/drive/MyDrive/AppGraD/grasp/grasp*.txt")):
      list_grasp.append(grasp_path)

    for d_path in glob.glob(f"/content/drive/MyDrive/AppGraD/vs1/d*.png"):
      list_D1.append(d_path)

    list_RGB1 = sorted(list_RGB1)
    list_grasp = sorted(list_grasp)
    list_D1 = sorted(list_D1)

    return list_RGB1, list_D1, list_grasp

  elif branches== 'two_rgb':
    for im_path in glob.glob(f"/content/drive/MyDrive/gad4dof/vs5/rgb*.png"):
      list_RGB1.append(im_path)

    for im_path in glob.glob(f"/content/drive/MyDrive/gad4dof/vs6/rgb*.png"):
      list_RGB2.append(im_path)

    for c, grasp_path in enumerate(glob.glob("/content/drive/MyDrive/gad4dof/grasp/grasp*.txt")):
      list_grasp.append(grasp_path)

    list_RGB1 = sorted(list_RGB1)
    list_grasp = sorted(list_grasp)
    list_RGB2 = sorted(list_RGB2)

    return list_RGB1, list_RGB2, list_grasp

  elif branches== 'three_rgb':
    for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/vs10/rgb*.png"):
      list_RGB1.append(im_path)

    for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/vs11/rgb*.png"):
      list_RGB2.append(im_path)

    for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/vs12/rgb*.png"):
      list_RGB3.append(im_path)

    for c, grasp_path in enumerate(glob.glob("/content/drive/MyDrive/AppGraD/grasp/grasp*.txt")):
      list_grasp.append(grasp_path)

    list_RGB1 = sorted(list_RGB1)
    list_RGB2 = sorted(list_RGB2)
    list_RGB3 = sorted(list_RGB3)
    list_grasp = sorted(list_grasp)

    return list_RGB1, list_RGB2, list_RGB3, list_grasp

  elif branches== 'three_d':
    for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/vs10/d*.png"):
      list_D1.append(im_path)

    for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/vs11/d*.png"):
      list_D2.append(im_path)

    for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/vs12/d*.png"):
      list_D3.append(im_path)

    for c, grasp_path in enumerate(glob.glob("/content/drive/MyDrive/AppGraD/grasp/grasp*.txt")):
      list_grasp.append(grasp_path)

    list_D1 = sorted(list_D1)
    list_D2 = sorted(list_D2)
    list_D3 = sorted(list_D3)
    list_grasp = sorted(list_grasp)
    
    return list_D1, list_D2, list_D3, list_grasp

def test_path_lists(branches='one'):
  # fix pers and iso labeling
  list_grasp = []
  list_RGB1 = []
  list_D1 = []
  list_RGB2 = []
  list_D2 = []
  list_RGB3 = []
  list_D3 = []

  if branches=='one' or branches=='two_rgbd':
    for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/test/vs1/rgb*.png"):
      list_RGB1.append(im_path)

    for c, grasp_path in enumerate(glob.glob("/content/drive/MyDrive/AppGraD/test/grasp/grasp*.txt")):
      list_grasp.append(grasp_path)

    for d_path in glob.glob(f"/content/drive/MyDrive/AppGraD/test/vs1/d*.png"):
      list_D1.append(d_path)

    list_RGB1 = sorted(list_RGB1)
    list_grasp = sorted(list_grasp)
    list_D1 = sorted(list_D1)

    return list_RGB1, list_D1, list_grasp

  elif branches== 'two_rgb':
    for im_path in glob.glob(f"/content/drive/MyDrive/gad4dof/vs5/rgb*.png"):
      list_RGB1.append(im_path)

    for im_path in glob.glob(f"/content/drive/MyDrive/gad4dof/vs6/rgb*.png"):
      list_RGB2.append(im_path)

    for c, grasp_path in enumerate(glob.glob("/content/drive/MyDrive/gad4dof/grasp/grasp*.txt")):
      list_grasp.append(grasp_path)

    list_RGB1 = sorted(list_RGB1)
    list_grasp = sorted(list_grasp)
    list_RGB2 = sorted(list_RGB2)

    return list_RGB1, list_RGB2, list_grasp

  elif branches== 'three_rgb':
    for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/test/vs10/rgb*.png"):
      list_RGB1.append(im_path)

    for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/test/vs11/rgb*.png"):
      list_RGB2.append(im_path)

    for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/test/vs12/rgb*.png"):
      list_RGB3.append(im_path)

    for c, grasp_path in enumerate(glob.glob("/content/drive/MyDrive/AppGraD/test/grasp/grasp*.txt")):
      list_grasp.append(grasp_path)

    list_RGB1 = sorted(list_RGB1)
    list_RGB2 = sorted(list_RGB2)
    list_RGB3 = sorted(list_RGB3)
    list_grasp = sorted(list_grasp)

    return list_RGB1, list_RGB2, list_RGB3, list_grasp

  elif branches== 'three_d':
    for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/test/vs10/d*.png"):
      list_D1.append(im_path)

    for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/test/vs11/d*.png"):
      list_D2.append(im_path)

    for im_path in glob.glob(f"/content/drive/MyDrive/AppGraD/test/vs12/d*.png"):
      list_D3.append(im_path)

    for c, grasp_path in enumerate(glob.glob("/content/drive/MyDrive/AppGraD/test/grasp/grasp*.txt")):
      list_grasp.append(grasp_path)

    list_D1 = sorted(list_D1)
    list_D2 = sorted(list_D2)
    list_D3 = sorted(list_D3)
    list_grasp = sorted(list_grasp)
    
    return list_D1, list_D2, list_D3, list_grasp


def unison_shuffle(a, b, c, d=None):
  if d is None:
    inx=np.random.permutation(a.shape[0])
    return a[inx],b[inx],c[inx]
  elif d is not None:
    inx=np.random.permutation(a.shape[0])
    return a[inx], b[inx], c[inx], d[inx]

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