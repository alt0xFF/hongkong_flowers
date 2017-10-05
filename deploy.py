import os
import os.path as osp
import shutil
import subprocess
import argparse
import tensorflow as tf
import keras

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils
from tensorflow.python.saved_model import utils as saved_model_utils
from tensorflow.contrib.session_bundle import exporter
from keras import backend as K
import numpy as np
from options import Options

height = 244
width = 244
channels = 3

def preprocess_input(tf_image):
    r, g, b = tf.split(tf_image, num_or_size_splits=3, axis=3)

    # Zero-center by mean pixel
    b -= 103.939
    g -= 116.779
    r -= 123.68

    # 'RGB'->'BGR'
    data = tf.concat([b, g, r], 3)

    return data

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
    parser = argparse.ArgumentParser(description='Deploy model to gcloud')
    parser.add_argument('-t', '--trained_model', default='model_best_weights.h5',
                        type=str,
                        help='trained model location')
    parser.add_argument('-m', '--model_dir', default='./checkpoints/2017-10-04_experiment_0/',
                        type=str,
                        help='model directory')
    parser.add_argument('-b', '--bucket_dir', default='gs://staging.dlhk-flower.appspot.com',
                        type=str,
                        help='gcloud storage bucket')
    args = parser.parse_args()

    K.set_learning_phase(bool(0))

    tf_model_path = osp.join(args.model_dir, 'export')

    try:
        # remove old model file
        print('removing {}'.format(tf_model_path))
        shutil.rmtree(tf_model_path)
    except:
        pass

    builder = saved_model_builder.SavedModelBuilder(tf_model_path)

    image_str_tensor = tf.placeholder(tf.string, shape=[None])

    image = tf.map_fn(
        decode_and_resize, image_str_tensor, back_prop=False, dtype=tf.uint8)
    # convert_image_dtype, also scales [0, uint8_max] -> [0 ,1).
    image = tf.cast(image, dtype=tf.float32)

    # setting up, options contains all our params
    options = Options(library=0,    # use keras
                      configs=2,    # use resnet50 model
                      transform=1)  # use transform for resnet50

    # load the weight file
    options.load = True
    options.load_file = osp.join(args.model_dir, args.trained_model)

    input_tensor = preprocess_input(image)

    # initialize model
    model = options.FlowerClassificationModel(options, input_tensor=input_tensor)

    my_model = model.model

    keys_placeholder = tf.placeholder(tf.string, shape=[None])
    inputs = {
        'key': keys_placeholder,
        'image_bytes': image_str_tensor
    }

    # To extract the id, we need to add the identity function.
    keys = tf.identity(keys_placeholder)
    outputs = {
        'key': keys,
        'prediction': my_model.output,
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

    subprocess.call(['gsutil', 'cp','-r', tf_model_path, args.bucket_dir])
