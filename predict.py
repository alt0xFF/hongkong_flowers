# -*- coding: utf-8 -*-
import sys
import json
import base64
import numpy as np

from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery

credentials = GoogleCredentials.get_application_default()
ml = discovery.build("ml", "v1", credentials=credentials)

img = load_img(sys.argv[1], target_size=(224, 224))  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 300, 300)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

datagen = ImageDataGenerator()

for batch in datagen.flow(x, batch_size=1, save_format='jpg'):
    data = {"instances": [{
        "image_bytes": {
            "b64": base64.b64encode(batch).decode('utf8')
        },
        "key": 0
    }]}
    req = ml.projects().predict(
        body=data, name="projects/{0}/models/{1}".format("dlhk-flower", "baseline")
    )
    result = req.execute()

    print(result)

    break
