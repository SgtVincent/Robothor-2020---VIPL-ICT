from PIL import Image
import os
import sys

PATH_TO_STORE = "/home/ubuntu/Robothor_class_images/temp.jpg"

def vis(event):
    image = Image.fromarray(event.frame)
    image.save(PATH_TO_STORE)