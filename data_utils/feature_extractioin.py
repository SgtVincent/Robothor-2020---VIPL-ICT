import numpy as np
import h5py
import torchvision.transforms as transforms
from PIL import Image
import pretrainedmodels
import torch
import argparse
import torch.multiprocessing as mp

ROBOTHOR_SCENES=["FloorPlan_Train{}_{}".format(i,j) for i in range(1, 13) for j in range(1,6)]
ITHOR_SCENES=["FloorPlan{}".format(i) for i in
    list(range(1,31)) +
    list(range(201,231)) +
    list(range(301,331)) +
    list(range(401,431))
]

ROBOTHOR_DIR="/home/chenjunting/ai2thor_data/Robothor_data"
ITHOR_DIR="/home/chenjunting/ai2thor_data/Ithor_data"

def parse_arguments():
    parser = argparse.ArgumentParser(description="scrape all possible images from ai2thor scene")

    parser.add_argument(
        "--scene_dir",
        type=str,
        help="path where ai2thor scene images stored",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="robothor",
        help="choice of scenes: {robothor, ithor}"
    )
    parser.add_argument(
        "--num_process",
        type=int,
        default=4,
        help="number of processes launched to scrape images parallelly",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="image encoder in pretrainded_models to extract features"
    )
    parser.add_argument(
        "-i",
        type=int,
        default=1
    )

    args = parser.parse_args()
    return args

def transform_visualize(input_image, im_size=224):
    all_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(im_size),
        transforms.CenterCrop(im_size),
    ])
    transformed_image = all_transforms(input_image)
    return transformed_image


# %%

def resnet_input_transform(input_image, im_size=224):
    all_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(im_size),
        transforms.CenterCrop(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transformed_image = all_transforms(input_image)
    return transformed_image

# save_shape default: resnet18: (512,7,7)
def extract_features(scene, scene_dir, model, save_shape=(512,7,7)):
    resnet18 = pretrainedmodels.__dict__[model](num_classes=1000, pretrained='imagenet')
    if torch.cuda.is_available():
        resnet18.cuda()


    images = h5py.File('{}/{}/images.hdf5'.format(scene_dir, scene), 'r')
    features = h5py.File('{}/{}/{}.hdf5'.format(scene_dir, scene, model), 'w')

    for k in images:
        frame = resnet_input_transform(images[k][:], 224)
        frame = torch.Tensor(frame)
        if torch.cuda.is_available():
            frame = frame.cuda()
        frame = frame.unsqueeze(0)

        v = resnet18.features(frame)
        v = v.view(*save_shape)

        v = v.cpu().detach().numpy()
        features.create_dataset(k, data=v)

    images.close()
    features.close()
    print("Feature data in {} has been saved".format(scene))

def mp_extract_features(queue, scene_dir, model, save_shape):
    while not queue.empty():
        try:
            scene = queue.get(timeout=3)
        except:
            return
        extract_features(scene, scene_dir, model, save_shape)

def Test():
    scene = 'FloorPlan1'
    scene_dir = '/home/chenjunting/Documents/ai2thor_datasets/thor_offline_data_with_images'
    extract_features(scene, scene_dir, 'resnet18_featuremap')

if __name__ == "__main__":

    args = parse_arguments()

    # prepare shared data
    if args.scenes == "robothor":
        scenes = ROBOTHOR_SCENES
    elif args.scenes == "ithor":
        scenes = ITHOR_SCENES
    elif args.scenes == "custom_robothor":
        scenes = ["FloorPlan_Train{}_{}".format(args.i, j) for j in range(1, 6)]
    else:
        raise Exception("--scenes value should be in {robothor, ithor, custom_robothor}")
    queue = mp.Queue()
    for x in scenes:
        queue.put(x)
    processes = []
    for rank in range(args.num_process):
        p = mp.Process(target=mp_extract_features, args=(queue, args.scene_dir, args.model, (512,7,7)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()




