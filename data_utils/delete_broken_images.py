import os
from PIL import Image
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="scrape all possible images from ai2thor scene")

    parser.add_argument(
        "--image_dir",
        type=str,
        default="/home/ubuntu/Robothor_class_images/images",
        help="path where object images stored",
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    scenes = os.listdir(args.image_dir)
    for scene in scenes:
        obj_subdirs = [ os.path.join(args.image_dir, scene, obj)
            for obj in os.listdir(os.path.join(args.image_dir, scene))
        ]
        for obj_subdir in obj_subdirs:
            images = [ os.path.join(obj_subdir, img)
                for img in os.listdir(obj_subdir)
            ]
            for img in images:
                try:
                    image = Image.open(img, 'r')
                    image.close()
                except:
                    print("Image file {} broken, deleted...".format(img))
                    os.remove(img)
