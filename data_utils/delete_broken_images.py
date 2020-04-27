import os
from PIL import Image
import argparse
import numpy as np
import torchvision.transforms as transforms

def parse_arguments():
    parser = argparse.ArgumentParser(description="scrape all possible images from ai2thor scene")

    parser.add_argument(
        "--image_dir",
        type=str,
        # default="/home/ubuntu/Robothor_class_images/images",
        default="/home/ubuntu/download_images",
        help="path where object images stored",
    )
    args = parser.parse_args()
    return args

def resnet_input_transform(input_image, im_size=224):
    all_transforms = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(im_size),
        transforms.CenterCrop(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transformed_image = all_transforms(input_image)
    return transformed_image

def all_images(image_dir):
    for root, dirs, files in os.walk(args.image_dir):
        for f in files:
            path = os.path.join(root, f)
            yield path

if __name__ == '__main__':
    args = parse_arguments()
    scenes = os.listdir(args.image_dir)
    for img_path in all_images(args.image_dir):
        try:
            image = Image.open(img_path, 'r')
            image_tensor = resnet_input_transform(image)
            image.close()
        except:
            print("Image file {} broken, deleted...".format(img_path))
            os.remove(img_path)
    # for scene in scenes:
    #     obj_subdirs = [ os.path.join(args.image_dir, scene, obj)
    #         for obj in os.listdir(os.path.join(args.image_dir, scene))
    #     ]
    #     for obj_subdir in obj_subdirs:
    #         images = [ os.path.join(obj_subdir, img)
    #             for img in os.listdir(obj_subdir)
    #         ]
    #         for img in images:
    #             try:
    #                 image = Image.open(img, 'r')
    #                 img_arr = image.asarray(image)
    #                 image.close()
    #                 assert img_arr.shape[0] == 3
    #             except:
    #                 print("Image file {} broken, deleted...".format(img))
    #                 os.remove(img)
