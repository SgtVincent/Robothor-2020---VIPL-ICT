import ai2thor.controller
from itertools import product
import concurrent.futures
import ai2thor.controller
import multiprocessing as mp
import numpy as np
import cv2
import hashlib
import json
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="scrape all possible images of classes from ai2thor scene")

    parser.add_argument(
        "--out_dir",
        type=str,
        default='/home/ubuntu/Robothor_class_images',
        help="path to store scraped images",
    )

    parser.add_argument(
        "--num_process",
        type=int,
        default=12,
        help="number of processes launched to scrape images parallelly",
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="Ultra",
        help="High,High WebGL,Low,Medium,MediumCloseFitShadows,Ultra,Very High,Very Low"
    )
    parser.add_argument(
        "--player_size",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--zoom_size",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--rotate_by",
        type=int,
        default=30
    )
    parser.add_argument(
        "--min_visible_ratio",
        type=float,
        default = 0.5,
        help="object must be at least <min_visible_ratio> in view"
    )
    parser.add_argument(
        "--grid_size",
        type=float,
        default=0.125,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Robothor",
        help="{Robothor, Ithor}"
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=15,
        help="extra pixels for object in the scene"
    )
    args = parser.parse_args()
    return args

def class_dataset_images_for_scene(scene_name, args):

    quality = args.quality
    player_size = args.player_size
    zoom_size = args.zoom_size
    target_size = args.target_size
    rotations = list(range(0, 360, args.rotate_by))
    # TODO: customize horizon range
    horizons = [330, 0, 30]
    buffer = 15
    # object must be at least min_visible_ratio in view
    min_size = ((target_size * args.min_visible_ratio) / zoom_size) * player_size

    env = ai2thor.controller.Controller(
        width=args.player_size,
        height=args.player_size,
        agentMode='bot',
        gridSize=args.grid_size,
        rotateStepDegrees=args.rotate_by,
        renderObjectImage=True,
        renderClassImage=False,
        renderImage=False,
    )

    # env.start(width=player_size, height=player_size) # ai2thor <= 1.0.1
    event = env.reset(scene_name)
    # event = env.step(
    #     dict(
    #         action="Initialize",
    #         gridSize=args.grid_size,
    #         renderObjectImage=True,
    #         renderClassImage=False,
    #         renderImage=False,
    #     )
    # ) # ai2thor <= 1.0.1

    for o in event.metadata["objects"]:
        if o["receptacle"] and o["receptacleObjectIds"] and o["openable"]:
            print("opening %s" % o["objectId"])
            env.step(
                dict(action="OpenObject", objectId=o["objectId"], forceAction=True)
            )

    event = env.step(dict(action="GetReachablePositions", gridSize=args.grid_size))

    visible_object_locations = []
    for point in event.metadata["actionReturn"]:
        for rot, hor in product(rotations, horizons):
            exclude_colors = set(
                map(tuple, np.unique(event.instance_segmentation_frame[0], axis=0))
            )
            exclude_colors.update(
                set(
                    map(
                        tuple,
                        np.unique(event.instance_segmentation_frame[:, -1, :], axis=0),
                    )
                )
            )
            exclude_colors.update(
                set(
                    map(tuple, np.unique(event.instance_segmentation_frame[-1], axis=0))
                )
            )
            exclude_colors.update(
                set(
                    map(
                        tuple,
                        np.unique(event.instance_segmentation_frame[:, 0, :], axis=0),
                    )
                )
            )

            event = env.step(
                dict(
                    action="TeleportFull",
                    x=point["x"],
                    y=point["y"],
                    z=point["z"],
                    rotation=rot,
                    horizon=hor,
                    forceAction=True,
                ),
                raise_for_failure=True,
            )

            visible_objects = []

            for o in event.metadata["objects"]:

                # if o["visible"] and o["objectId"] and o["pickupable"]:
                if o["visible"] and o["objectId"]:
                    color = event.object_id_to_color[o["objectId"]]
                    mask = (
                        (event.instance_segmentation_frame[:, :, 0] == color[0])
                        & (event.instance_segmentation_frame[:, :, 1] == color[1])
                        & (event.instance_segmentation_frame[:, :, 2] == color[2])
                    )
                    points = np.argwhere(mask)

                    if len(points) > 0:
                        min_y = int(np.min(points[:, 0]))
                        max_y = int(np.max(points[:, 0]))
                        min_x = int(np.min(points[:, 1]))
                        max_x = int(np.max(points[:, 1]))
                        max_dim = max((max_y - min_y), (max_x - min_x))
                        if (
                            max_dim > min_size
                            and min_y > buffer
                            and min_x > buffer
                            and max_x < (player_size - buffer)
                            and max_y < (player_size - buffer)
                        ):
                            visible_objects.append(
                                dict(
                                    objectId=o["objectId"],
                                    min_x=min_x,
                                    min_y=min_y,
                                    max_x=max_x,
                                    max_y=max_y,
                                )
                            )
                            print(
                                "[%s] including object id %s %s"
                                % (scene_name, o["objectId"], max_dim)
                            )

            if visible_objects:
                visible_object_locations.append(
                    dict(point=point, rot=rot, hor=hor, visible_objects=visible_objects)
                )

    env.stop()
    env = ai2thor.controller.Controller(
        quality=args.quality,
        width=zoom_size,
        height=zoom_size,
        agentMode='bot',
        rotateStepDegrees=args.rotate_by,
        gridSize=args.grid_size
    )
    # env.start(width=zoom_size, height=zoom_size) # ai2thor <= 1.0.1
    env.reset(scene_name)
    # event = env.step(dict(action="Initialize", gridSize=args.grid_size)) # ai2thor <= 1.0.1

    for o in event.metadata["objects"]:
        if o["receptacle"] and o["receptacleObjectIds"] and o["openable"]:
            print("opening %s" % o["objectId"])
            env.step(
                dict(action="OpenObject", objectId=o["objectId"], forceAction=True)
            )

    image_metadata = {}

    for vol in visible_object_locations:
        point = vol["point"]

        event = env.step(
            dict(
                action="TeleportFull",
                x=point["x"],
                y=point["y"],
                z=point["z"],
                rotation=vol["rot"],
                horizon=vol["hor"],
                forceAction=True,
            ),
            raise_for_failure=True,
        )


        for v in vol["visible_objects"]:
            object_id = v["objectId"]
            min_y = int(round(v["min_y"] * (zoom_size / player_size)))
            max_y = int(round(v["max_y"] * (zoom_size / player_size)))
            max_x = int(round(v["max_x"] * (zoom_size / player_size)))
            min_x = int(round(v["min_x"] * (zoom_size / player_size)))
            delta_y = max_y - min_y
            delta_x = max_x - min_x
            scaled_target_size = max(delta_x, delta_y, target_size) + buffer * 2
            if min_x > (zoom_size - max_x):
                start_x = min_x - (scaled_target_size - delta_x)
                end_x = max_x + buffer
            else:
                end_x = max_x + (scaled_target_size - delta_x)
                start_x = min_x - buffer

            if min_y > (zoom_size - max_y):
                start_y = min_y - (scaled_target_size - delta_y)
                end_y = max_y + buffer
            else:
                end_y = max_y + (scaled_target_size - delta_y)
                start_y = min_y - buffer

            # print("max x %s max y %s min x %s  min y %s" % (max_x, max_y, min_x, min_y))
            # print("start x %s start_y %s end_x %s end y %s" % (start_x, start_y, end_x, end_y))
            print("storing %s " % object_id)
            # origin_img = event.cv2img
            img = event.cv2img[start_y:end_y, start_x:end_x, :]
            # seg_img = event.cv2img[min_y:max_y, min_x:max_x, :]
            # dst = cv2.resize(
            #     img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4
            # )

            object_type = object_id.split("|")[0].lower()
            origin_target_dir = os.path.join(args.out_dir, "origin_images", scene_name, object_type)
            target_dir = os.path.join(args.out_dir, "images", scene_name, object_type)
            cropped_target_dir = os.path.join(args.out_dir, "cropped_images", scene_name, object_type)

            h = hashlib.md5()
            h.update(json.dumps(point, sort_keys=True).encode("utf8"))
            h.update(json.dumps(v, sort_keys=True).encode("utf8"))

            # os.makedirs(origin_target_dir, exist_ok=True)
            os.makedirs(target_dir, exist_ok=True)
            # os.makedirs(cropped_target_dir, exist_ok=True)

            # cv2.imwrite(os.path.join(origin_target_dir, h.hexdigest() + ".png"), origin_img)
            cv2.imwrite(os.path.join(target_dir, h.hexdigest() + ".png"), img)
            # cv2.imwrite(os.path.join(cropped_target_dir,  h.hexdigest() + ".png"), seg_img)

            # save metadata for each image: bbox
            key = os.path.join(object_type, h.hexdigest() + ".png")
            # if object_type not in image_metadata:
            #             #     image_metadata[object_type]={
            #             #         key:{
            #             #             "bbox": (min_y, max_y, min_x, max_x)
            #             #         }
            #             #     }
            #             # else:
            #             #     image_metadata[object_type].update({
            #             #         key: {
            #             #             "bbox": (min_y, max_y, min_x, max_x)
            #             #         }
            #             #     })
    # save bbox data

    # with open(os.path.join("origin_images", scene_name, "meta_bbox.json"), "w") as f:
    #     json.dump(image_metadata, f)

    env.stop()

    return scene_name


# def build_class_dataset(max_workers=4, dataset="Test", quality="Ultra"):
if __name__ == '__main__':

    args = parse_arguments()

    mp.set_start_method("spawn")

    controller = ai2thor.controller.Controller()
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.num_process)
    futures = []

    if args.dataset == "Ithor":
        scene_names = controller.scene_names()
    elif args.dataset == "Robothor":
        scene_names = controller.robothor_scenes(types=["train"])
    else:
        print("Dataset {} not supported...".format(args.dataset))
        exit(0)

    for scene in scene_names:
        print("processing scene %s" % scene)
        futures.append(executor.submit(class_dataset_images_for_scene, scene, args))

    for f in concurrent.futures.as_completed(futures):
        scene = f.result()
        print("scene name complete: %s" % scene)


