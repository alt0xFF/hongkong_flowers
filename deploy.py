import os
import os.path as osp
import subprocess
import tensorflow as tf
import keras

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils
from tensorflow.python.saved_model import utils as saved_model_utils
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from keras.models import load_model, model_from_config, Model
from keras import backend as K
import numpy as np
from options import Options

trained_model = 'model_latest_weights.h5'
model_dir = './checkpoints/2017-10-03_experiment_0/'
bucket_dir = 'gs://staging.dlhk-flower.appspot.com'

height = 244
width = 244
channels = 3

def build_signature(inputs, outputs):
    """Build the signature.
    Not using predic_signature_def in saved_model because it is replacing the
    tensor name, b/35900497.
    Args:
    inputs: a dictionary of tensor name to tensor
    outputs: a dictionary of tensor name to tensor
    Returns:
    The signature, a SignatureDef proto.
    """
    signature_inputs = {key: saved_model_utils.build_tensor_info(tensor) for key, tensor in inputs.items()}
    signature_outputs = {key: saved_model_utils.build_tensor_info(tensor) for key, tensor in outputs.items()}

    signature_def = signature_def_utils.build_signature_def(
        signature_inputs, signature_outputs,
        signature_constants.PREDICT_METHOD_NAME)

    return signature_def

def decode_and_resize(image_str_tensor):
  """Decodes jpeg string, resizes it and returns a uint8 tensor."""
  image = tf.image.decode_jpeg(image_str_tensor, channels=channels)
  # Note resize expects a batch_size, but tf_map supresses that index,
  # thus we have to expand then squeeze.  Resize returns float32 in the
  # range [0, uint8_max]
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(
      image, [height, width], align_corners=False)
  image = tf.squeeze(image, squeeze_dims=[0])
  image = tf.cast(image, dtype=tf.uint8)
  return image

if __name__ == '__main__':
    K.set_learning_phase(bool(0))

    tf_model_path = osp.join(model_dir, 'export')

    try:
        # remove old model file
        print('removing {}'.format(tf_model_path))
        os.rmdir(tf_model_path)
    except:
        pass

    builder = saved_model_builder.SavedModelBuilder(tf_model_path)

    image_str_tensor = tf.placeholder(tf.string, shape=[None])

    image = tf.map_fn(
        decode_and_resize, image_str_tensor, back_prop=False, dtype=tf.uint8)
    # convert_image_dtype, also scales [0, uint8_max] -> [0 ,1).
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # setting up, options contains all our params
    options = Options(library=0,    # use keras
                      configs=2,    # use resnet50 model
                      transform=1)  # use transform for resnet50

    # load the weight file
    options.load = True
    options.load_file = osp.join(model_dir, trained_model)

    # initialize model
    model = options.FlowerClassificationModel(options, input_tensor=image).model

    keys_placeholder = tf.placeholder(tf.string, shape=[None])
    inputs = {
        'key': keys_placeholder,
        'image_bytes': image_str_tensor
    }

    # To extract the id, we need to add the identity function.
    keys = tf.identity(keys_placeholder)
    outputs = {
        'key': keys,
        'prediction': model.output,
    }

    signature_def = build_signature(inputs=inputs, outputs=outputs)
    signature_def_map = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
    }

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=[tag_constants.SERVING],
                                             signature_def_map=signature_def_map)
        builder.save()
        print("Saved")

    subprocess.call(['gsutil', 'cp','-r', tf_model_path, bucket_dir])
