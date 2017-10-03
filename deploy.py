import os
import os.path as osp
import subprocess
import tensorflow as tf
import keras

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from keras.models import load_model, model_from_config
from keras import backend as K
import numpy as np
from options import Options

trained_model = 'model_best_weights.h5'
model_dir = './checkpoints/2017-10-02_experiment_0/'
bucket_dir = 'gs://staging.dlhk-flower.appspot.com'

if __name__ == '__main__':
    K.set_learning_phase(bool(0))

    # setting up, options contains all our params
    options = Options(library=0,    # use keras
                      configs=2,    # use resnet50 model
                      transform=1)  # use transform for resnet50

    # load the weight file
    options.load = True
    options.load_file = osp.join(model_dir, trained_model)
    tf_model_path = osp.join(model_dir, 'export')

    # initialize model
    model = options.FlowerClassificationModel(options).model
    builder = saved_model_builder.SavedModelBuilder(tf_model_path)

    signature = predict_signature_def(inputs={'images': model.input},
                                      outputs={'preds': model.output})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=[tag_constants.SERVING],
                                             signature_def_map={'serving_default': signature})
        builder.save()
        print("Saved")

    subprocess.call(['gsutil', 'cp','-r', tf_model_path, bucket_dir])
