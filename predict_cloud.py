# -*- coding: utf-8 -*-
import os
import sys
import json
import base64
import numpy as np
from PIL import Image

from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery

credentials = GoogleCredentials.get_application_default()
ml = discovery.build("ml", "v1", credentials=credentials)
classes = [i for i in os.listdir('./dataset/train/') if os.path.isdir('./dataset/train/' + i)]

img = open(sys.argv[1], 'rb')  # this is a PIL image

data = {"instances": [{
    "image_bytes": {
        "b64": base64.b64encode(img.read()).decode('utf8')
    },
    "key": "0"
}]}
req = ml.projects().predict(
    body=data, name="projects/{0}/models/{1}".format("dlhk-flower", "baseline")
)
result = req.execute()
preds = result['predictions'][0]['prediction']
pred = np.argmax(preds)

print(preds[pred])
print(pred, classes[pred])
