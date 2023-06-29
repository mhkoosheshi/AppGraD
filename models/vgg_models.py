import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, BatchNormalization, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.constraints import MaxNorm
from keras.layers import Lambda


def vgg16_double(trainable=True,
                shape=(224,224),
                N=5,
                ):

    inputsRGB = layers.Input((shape[0], shape[1], 3))
    # stack the depth to have (-,-,3)
    inputsD = layers.Input((shape[0], shape[1], 3))

    # RGB
    backbone = keras.applications.VGG16(weights="imagenet",
                                      include_top=False,
                                      input_shape=(shape[0], shape[1], 3))
    backbone._name='vgg1'
    backbone.trainable = True
    backbone = backbone(inputsRGB)
    outputs1 = backbone
    outputs1 = layers.SeparableConv2D(N*4, kernel_size=3, strides=1, activation="relu")(outputs1)
    outputs1 = layers.Flatten()(outputs1)
    outputs1 = layers.Dense(64)(outputs1)
    outputs1 = layers.Dense(64)(outputs1)

    model1 = keras.Model(inputsRGB, outputs1, name='rgb_net')

    # Depth
    backbone = keras.applications.VGG16(weights="imagenet",
                                      include_top=False,
                                      input_shape=(shape[0], shape[1], 3))
    backbone._name='vgg2'
    backbone.trainable = True
    backbone = backbone(inputsD)
    outputs2 = backbone
    outputs2 = layers.SeparableConv2D(N*4, kernel_size=3, strides=1, activation="relu")(outputs2)
    outputs2 = layers.Flatten()(outputs2)
    outputs2 = layers.Dense(64)(outputs2)
    outputs2 = layers.Dense(64)(outputs2)

    model2 = keras.Model(inputsD, outputs2, name='depth_net')

    # concatenate for the last MLP
    cnct = layers.concatenate([model1.output, model2.output])
    outputs = layers.Dense(128)(cnct)
    outputs = layers.Dense(128)(outputs)
    outputs = layers.Dense(128)(outputs)
    outputs = layers.Dense(128)(outputs)
    outputs = layers.Dense(N)(outputs)

    model = keras.Model([model1.input, model2.input], outputs, name='vgg16_double')

    return model


def vgg16_single(trainable=True,
                shape=(224,224),
                N=5,
                ):
    
    inputs = layers.Input((shape[0], shape[1], 3))

    backbone = keras.applications.VGG16(weights="imagenet",
                                        include_top=False,
                                        input_shape=(shape[0], shape[1], 3)
    )
    backbone.trainable = trainable
    backbone = backbone(inputs)
    outputs = backbone
    # outputs = layers.Dropout(0.3)(outputs)
    outputs = layers.SeparableConv2D(
        N*4, kernel_size=3, strides=1, activation="relu"
    )(outputs)
    # outputs = layers.SeparableConv2D(
    #     N, kernel_size=3, strides=1, activation="relu"
    # )(outputs)
    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(128)(outputs)
    outputs = layers.Dense(32)(outputs)
    outputs = layers.Dense(16)(outputs)
    outputs = layers.Dense(N)(outputs)


    # outputs = layers.SeparableConv2D(
    #     N, kernel_size=3, strides=1
    # )(outputs)
    model = keras.Model(inputs, outputs, name='vgg16_single')

    return model

def vgg16_s2d(n_freeze=10,
              n_pick=18,
              shape=(224,224)):

  '''
  A modified version of vgg16 block, including vgg16's kernels and s2d as its downsampling method

  n_freeze (int): the number of starting layers to freeze. max=18
  n_pick (int): the number of layers of vgg16 to pick
  '''

  vgg = VGG16(include_top=False, 
              weights='imagenet', 
              input_shape=(224,224,3)
              )
  vgg.trainable = True

  for layer in vgg.layers[1:n_freeze+1]:
    layer.trainable = False

  s2d = Lambda(lambda x:tf.nn.space_to_depth(x,2))
  x = tf.keras.Input(shape=(224,224,3))
  x1 = vgg.layers[1](x)
  x1 = vgg.layers[2](x1)
  x1 = s2d(x1)
  x1 = vgg.layers[4](x1)
  x1 = vgg.layers[5](x1)
  x1 = s2d(x1)
  x1 = vgg.layers[7](x1)
  x1 = vgg.layers[8](x1)
  x1 = vgg.layers[9](x1)
  x1 = s2d(x1)
  x1 = vgg.layers[11](x1)
  x1 = vgg.layers[12](x1)
  x1 = vgg.layers[13](x1)
  x1 = s2d(x1)
  x1 = vgg.layers[15](x1)
  x1 = vgg.layers[16](x1)
  x1 = vgg.layers[17](x1)
  x1 = s2d(x1)

  vgg_s2d = tf.keras.Model(inputs=x, outputs=x1)

  return vgg_s2d