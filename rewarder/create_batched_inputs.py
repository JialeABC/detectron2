import cv2
from torchvision.io import read_image
import os
def create_batched_inputs(inf_data):
    res = []
    for img in inf_data:
        temp = {}
        temp['file_name'] =img
        image = read_image(img)
        temp['image'] = image
        temp['height'] = image.shape[1]
        temp['width'] = image.shape[2]
        temp['image_id']= os.path.splitext(os.path.basename(img))[0]
        res.append(temp)
    return res