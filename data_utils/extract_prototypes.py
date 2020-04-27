import argparse
import os

import pretrainedmodels
import torch
import multiprocessing as mp
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import h5py

def parse_arguments():
    parser = argparse.ArgumentParser(description="scrape all possible images from ai2thor scene")

    parser.add_argument(
        "--class_images_dir",
        type=str,
        default="/home/ubuntu/Robothor_class_images/images",
        help="path where object images stored",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="/home/ubuntu/chenjunting/savn/data/object_protos.hdf5",
        help="path to store prototypes",
    )
    parser.add_argument(
        "--num_process",
        type=int,
        default=8,
        help="number of processes launched to scrape images parallelly",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="image encoder in pretrainded_models to extract features"
    )

    parser.add_argument(
        "--object_types",
        nargs='*',
        default=[],
        help="target objects to save prototype, default save all objects in <class_images_dir>"
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="online",
        help="choose from {online, ai2thor}"
    )
    parser.add_argument(
        "--scenes",
        nargs='+',
        default=['FloorPlan_Train1_1', 'FloorPlan_Train1_2', 'FloorPlan_Train1_3', 'FloorPlan_Train1_4',
                 'FloorPlan_Train1_5', 'FloorPlan_Train2_1', 'FloorPlan_Train2_2', 'FloorPlan_Train2_3',
                 'FloorPlan_Train2_4', 'FloorPlan_Train2_5', 'FloorPlan_Train3_1', 'FloorPlan_Train3_2',
                 'FloorPlan_Train3_3', 'FloorPlan_Train3_4', 'FloorPlan_Train3_5', 'FloorPlan_Train4_1',
                 'FloorPlan_Train4_2', 'FloorPlan_Train4_3', 'FloorPlan_Train4_4', 'FloorPlan_Train4_5',
                 'FloorPlan_Train5_1', 'FloorPlan_Train5_2', 'FloorPlan_Train5_3', 'FloorPlan_Train5_4',
                 'FloorPlan_Train5_5', 'FloorPlan_Train6_1', 'FloorPlan_Train6_2', 'FloorPlan_Train6_3',
                 'FloorPlan_Train6_4', 'FloorPlan_Train6_5', 'FloorPlan_Train7_1', 'FloorPlan_Train7_2',
                 'FloorPlan_Train7_3', 'FloorPlan_Train7_4', 'FloorPlan_Train7_5', 'FloorPlan_Train8_1',
                 'FloorPlan_Train8_2', 'FloorPlan_Train8_3', 'FloorPlan_Train8_4', 'FloorPlan_Train8_5',
                 'FloorPlan_Train9_1', 'FloorPlan_Train9_2', 'FloorPlan_Train9_3', 'FloorPlan_Train9_4',
                 'FloorPlan_Train9_5', 'FloorPlan_Train10_1', 'FloorPlan_Train10_2', 'FloorPlan_Train10_3',
                 'FloorPlan_Train10_4', 'FloorPlan_Train10_5', 'FloorPlan_Train11_1', 'FloorPlan_Train11_2',
                 'FloorPlan_Train11_3', 'FloorPlan_Train11_4', 'FloorPlan_Train11_5', 'FloorPlan_Train12_1',
                 'FloorPlan_Train12_2', 'FloorPlan_Train12_3', 'FloorPlan_Train12_4', 'FloorPlan_Train12_5'],
        help="scenes to scrape"
    )

    parser.add_argument(
        "--gpus",
        nargs="+",
        default=[0]
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
    )
    args = parser.parse_args()
    return args


# def transform_visualize(input_image, im_size=224):
#     all_transforms = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize(im_size),
#         transforms.CenterCrop(im_size),
#     ])
#     transformed_image = all_transforms(input_image)
#     return transformed_image

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


def mean_O_n(images, gpu, model, batch_size=20):
    num_images = len(images)
    proto = None
    count = 0
    for i in range(0, num_images, batch_size):
        batch_images = [Image.open(img_path)
                        for img_path in images[i: i + batch_size]]
        batch_tensors = torch.stack([resnet_input_transform(image)
                                     for image in batch_images], dim=0).cuda(gpu)
        batch_features = model.features(batch_tensors)
        if count == 0:
            proto = torch.zeros(batch_features.shape[1:]).cuda(gpu)
        proto = (count * proto + batch_features.mean(dim=0) * batch_features.shape[0]) \
                / (count + batch_features.shape[0])

        for image in batch_images:
            image.close()

    return np.squeeze(proto.cpu().detach().numpy())


# extract prototype for object_type by calculating mean of all images of the class
def extract_prototypes(object_type, class_dir, scenes, batch_size, model, data_source,
                       proto_queue, save_shape=(512, 7, 7), gpu_id=0):
    torch.cuda.set_device(gpu_id)
    # TODO: add more models options
    resnet18 = pretrainedmodels.__dict__[model](num_classes=1000, pretrained='imagenet')
    if torch.cuda.is_available():
        resnet18.cuda(gpu_id)

    images = []
    if data_source == "online":
        images = [os.path.join(class_dir, object_type, img_file)
                       for img_file in os.listdir(os.path.join(class_dir, object_type))]
    elif data_source == "ai2thor":
        for scene in scenes:
            if not os.path.exists(os.path.join(class_dir, scene, object_type)):
                continue
            images += [os.path.join(class_dir, scene, object_type, img_file)
                       for img_file in os.listdir(os.path.join(class_dir, scene, object_type))]
    else:
        print("data source({}) not allowed, choose from: online, ai2thor".format(data_source))
        exit(0)

    print("start to calculate prototype for object {}".format(object_type))

    proto = mean_O_n(images, gpu_id, resnet18, batch_size)
    proto_queue.put({object_type.replace(" ",""):
        {
            "prototype":proto,
            "num_images":len(images),
        }
    })
    print("Prototype for object {} has been saved".format(object_type))
    return

# helper function for multiprocessing
def mp_extract_prototypes(rank, args, obj_queue, proto_queue, save_shape):
    while not obj_queue.empty():
        try:
            object_type = obj_queue.get(timeout=3)
        except:
            return
        gpu_id = args.gpus[rank % len(args.gpus)]
        # extract_features(scene, args.scene_dir, args.model, save_shape, gpu_id)
        extract_prototypes(object_type=object_type,
                           class_dir=args.class_images_dir,
                           scenes=args.scenes,
                           batch_size=args.batch_size,
                           model=args.model,
                           data_source=args.data_source,
                           proto_queue=proto_queue,
                           gpu_id=gpu_id)


if __name__ == '__main__':

    args = parse_arguments()
    # if args.scenes:
    #     scenes = args.scenes.split(',')
    # else:  # all scenes in robothor
    #     scenes = ["FloorPlan_Train{}_{}".format(i, j) for i in range(1, 13) for j in range(1, 6)]

    obj_queue = mp.Queue()
    proto_queue = mp.Queue()

    all_objects = []
    if args.object_types == []:
        if args.data_source == "ai2thor":
            for scene in args.scenes:
                if scene not in os.listdir(args.class_images_dir):
                    continue
                all_objects += os.listdir(os.path.join(args.class_images_dir, scene))
        else:
            all_objects = os.listdir(args.class_images_dir)
        all_objects = sorted(np.unique(np.array(all_objects)))
    else:
        all_objects = args.object_types
    # put all objects in queue
    for obj in all_objects:
        obj_queue.put(obj)

    processes = []
    for rank in range(args.num_process):
        p = mp.Process(target=mp_extract_prototypes, args=(rank, args, obj_queue, proto_queue, (512, 7, 7)))
        p.start()
        processes.append(p)
    proto_dict = {}

    for obj in all_objects:
        proto_dict.update(proto_queue.get())

    for p in processes:
        p.join()

    with h5py.File(args.out_path, 'w') as f:
        for key in proto_dict.keys():
            f.create_dataset(key, data=proto_dict[key]['prototype'])
            print("There are {} images of {}".format(proto_dict[key]['num_images'], key))
