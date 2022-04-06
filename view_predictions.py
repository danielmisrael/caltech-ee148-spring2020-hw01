import os
import numpy as np
import json
from PIL import ImageDraw
from PIL import Image

# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions:
preds_path = '../data/hw01_preds'
os.makedirs(preds_path,exist_ok=True) # create directory if needed

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'r') as f:
    preds = json.load(f)

for i in range(len(file_names)):
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))

    for bounding_box in preds[file_names[i]]:
        draw = ImageDraw.Draw(I)
        draw.rectangle(bounding_box, outline='green')
    I.save(preds_path + '/pred_' + str(i), 'jpeg')
