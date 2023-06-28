import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, BatchNormalization, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.constraints import MaxNorm

def get_model(arch='simple',
              shape=(512,512),
              param_mode=[1,1,1,1,1],
              trainable = True):

  # N = param_mode.count(1)+1
  N = 5



  if arch=='simple':
    inputs = layers.Input((shape[0], shape[1], 3))

    backbone = layers.Conv2D(32, 5, strides=(3,3), activation="relu")(inputs)
    backbone = layers.Conv2D(64, 5, strides=(3,3), activation="relu")(backbone)
    backbone = layers.Conv2D(128, 5, strides=(2,2), activation="relu")(backbone)
    backbone = layers.Conv2D(256, 3, strides=(1,1), activation="relu")(backbone)
    backbone = layers.Conv2D(256, 3, strides=(1,1), activation="relu")(backbone)
    backbone = layers.Conv2D(512, 1, strides=(1,1), activation="relu")(backbone)
    outputs = backbone
    outputs = layers.Dropout(0.3)(outputs)
    outputs = layers.SeparableConv2D(
        N, kernel_size=2, strides=1, activation="relu"
    )(outputs)
    outputs = layers.SeparableConv2D(
        N, kernel_size=1, strides=1, activation='sigmoid'
    )(outputs)
    model = keras.Model(inputs, outputs, name=arch)

  elif arch=='VGG16':
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
    model = keras.Model(inputs, outputs, name=arch)


  elif arch=="RGBD":

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

    model = keras.Model([model1.input, model2.input], outputs, name=arch)

  elif arch=="ML_backbone_RGBD":

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
    outputs1 = layers.Flatten()(outputs1)

    model1 = keras.Model(inputsRGB, outputs1, name='rgb_net')

    # Depth
    backbone = keras.applications.VGG16(weights="imagenet",
                                      include_top=False,
                                      input_shape=(shape[0], shape[1], 3))
    backbone._name='vgg2'
    backbone.trainable = True
    backbone = backbone(inputsD)
    outputs2 = backbone
    outputs2 = layers.Flatten()(outputs2)

    model2 = keras.Model(inputsD, outputs2, name='depth_net')

    # concatenate for the last MLP
    cnct = layers.concatenate([model1.output, model2.output])

    model = keras.Model([model1.input, model2.input], cnct, name=arch)

  return model

model=get_model(arch='RGBD',
                shape=(224,224),
                param_mode=[1,1,1,1,1],
                trainable=False)
model.summary()