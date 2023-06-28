from models.vgg_models import vgg16_double, vgg16_single

def get_model(arch='simple',
              shape=(512,512),
              N=5,
              trainable = True):

  if arch=='vgg16_single':

    model = vgg16_single(trainable=True,
                        shape=shape,
                        N=N,
                        )


  elif arch=="vgg16_double":
    
    model = vgg16_double(trainable=True,
                        shape=shape,
                        N=N,
                        )

  return model