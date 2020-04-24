import argparse
import json
import os
import re
import random
from pprint import pprint

import ai2thor.controller
import ai2thor.util.metrics as metrics


def parse_arguments():
    parser = argparse.ArgumentParser(description="scrape all possible images of classes from ai2thor scene")

    parser.add_argument(
        "--out_dir",
        type=str,
        default='/home/ubuntu/chenjunting/savn/data/curriculum_meta',
        help="directory to store curriculum meta",
    )

    parser.add_argument(
        "--num_process",
        type=int,
        default=12,
        help="number of processes launched to scrape images parallelly",
    )
    parser.add_argument(
        "--player_size",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--visibility_distance",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--rotate_by",
        type=int,
        default=30
    )
    parser.add_argument(
        "--grid_size",
        type=float,
        default=0.125,
    )
    parser.add_argument(
        "--desired_points",
        type=int,
        default=30
    )
    parser.add_argument(
        "--distance_upgrade_step",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--penalty_decay",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--num_ep_per_stage",
        type=int,
        default=300000,
        help="number of episodes for each curriculum training stage"
    )
    parser.add_argument(
        "--max_diff_level",
        type=int,
        default=10,
        help="max difficulty level, default set to 10 (10m)"
    )
    parser.add_argument(
        "--dump_each_scene",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=[
            "Apple", "Baseball Bat","BasketBall", "Bowl", "Garbage Can", "House Plant",
            "Laptop","Mug","Remote","Spray Bottle","Vase", "Alarm Clock","Television", "Pillow"]
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["FloorPlan_Train{}_{}".format(i,j) for i in range(1,13) for j in range(1,6)]
    )

    args = parser.parse_args()
    return args


def sqr_dist(a, b):
    x = a[0] - b[0]
    z = a[2] - b[2]
    return x * x + z * z


def sqr_dist_dict(a, b):
    x = a['x'] - b['x']
    z = a['z'] - b['z']
    return x * x + z * z

def key_sort_func(scene_name):
    m = re.search('FloorPlan_([a-zA-Z\-]*)([0-9]+)_([0-9]+)', scene_name)
    return m.group(1), int(m.group(2)), int(m.group(3))

def get_points(controller,
               object_type,
               scene,
               objects_types_in_scene,
               failed_points,
               grid_size,
               rotate_by,
               desired_points=30):
        print("Getting points in scene {}, with target object {}...: ".format(scene, object_type))
        controller.reset(scene)
        event = controller.step(
            dict(
                action='ObjectTypeToObjectIds',
                objectType=object_type.replace(" ", "")
            )
        )
        object_ids = event.metadata['actionReturn']

        if object_ids is None or len(object_ids) > 1 or len(object_ids) == 0:
            print("Object type '{}' not available in scene.".format(object_type))
            return None

        objects_types_in_scene.add(object_type)
        object_id = object_ids[0]

        event_reachable = controller.step(
            dict(
                action='GetReachablePositions',
                gridSize=grid_size
            )
        )

        target_position = controller.step(action='GetObjectPosition', objectId=object_id).metadata['actionReturn']

        reachable_positions = event_reachable.metadata['actionReturn']

        reachable_pos_set = set([
            (pos['x'], pos['y'], pos['z']) for pos in reachable_positions
            # if sqr_dist_dict(pos, target_position) >= visibility_distance * visibility_multiplier_filter
        ])

        def filter_points(selected_points, point_set, minimum_distance):
            result = set()
            for selected in selected_points:
                if selected in point_set:
                    result.add(selected)
                    remove_set = set(
                        [p for p in point_set if sqr_dist(p, selected) <= minimum_distance * minimum_distance]
                    )
                    point_set = point_set.difference(remove_set)
            return result


        points = random.sample(reachable_pos_set, desired_points * 4)

        final_point_set = filter_points(points, reachable_pos_set, grid_size * 2)

        print("Total number of points: {}".format(len(final_point_set)))

        print("Id {}".format(event.metadata['actionReturn']))

        point_objects = []

        eps = 0.0001
        counter = 0
        for (x, y, z) in final_point_set:
            possible_orientations = list(range(0, 360, rotate_by))
            pos_unity = dict(x=x, y=y, z=z)
            try:
                path = metrics.get_shortest_path_to_object(
                    controller,
                    object_id,
                    pos_unity,
                    {'x': 0, 'y': 0, 'z': 0}
                )
                minimum_path_length = metrics.path_distance(path)

                rotation_allowed = False
                while not rotation_allowed:
                    if len(possible_orientations) == 0:
                        break
                    roatation_y = random.choice(possible_orientations)
                    possible_orientations.remove(roatation_y)
                    evt = controller.step(
                        action="TeleportFull",
                        x=pos_unity['x'],
                        y=pos_unity['y'],
                        z=pos_unity['z'],
                        rotation=dict(x=0, y=roatation_y, z=0)
                    )
                    rotation_allowed = evt.metadata['lastActionSuccess']
                    if not evt.metadata['lastActionSuccess']:
                        print(evt.metadata['errorMessage'])
                        print("--------- Rotation not allowed! for pos {} rot {} ".format(pos_unity, roatation_y))

                if minimum_path_length > eps and rotation_allowed:
                    m = re.search('FloorPlan_([a-zA-Z\-]*)([0-9]+)_([0-9]+)', scene)
                    point_id = "{}_{}_{}_{}_{}".format(
                        m.group(1),
                        m.group(2),
                        m.group(3),
                        object_type,
                        counter
                    )
                    point_objects.append({
                        'id': point_id,
                        'scene': scene,
                        'object_type': object_type,
                        'object_id': object_id,
                        'target_position': target_position,
                        'initial_position': pos_unity,
                        'initial_orientation': roatation_y,
                        'shortest_path': path,
                        'shortest_path_length': minimum_path_length
                    })
                    counter += 1

            except ValueError:
                print("-----Invalid path discarding point...")
                failed_points.append({
                    'scene': scene,
                    'object_type': object_type,
                    'object_id': object_id,
                    'target_position': target_position,
                    'initial_position': pos_unity
                })

        # sorted_objs = sorted(point_objects,
        #                      key=lambda m: sqr_dist_dict(m['initial_position'], m['target_position']))

        sorted_objs = sorted(point_objects,key=lambda m: m['shortest_path_length'])
        return sorted_objs

# @points: sorted points for one object
# @episodes: list of [points], points in one sub-list has same difficulty
# @max_diff_level: max splits number
def split_by_difficulty(points, episodes, distance_upgrade_step, max_diff_level=10):

    difficulty = 1
    split_index = [[] for i in range(max_diff_level)]
    start = 0
    for i, obj in enumerate(points):
        difficulty = round(obj['shortest_path_length'] // distance_upgrade_step) + 1
        if difficulty > 10: difficulty = 10
        points[i]['difficulty'] = difficulty
        split_index[difficulty-1].append(i)
    for i in range(max_diff_level):
        episodes[i] += [points[index] for index in split_index[i]]

    return


def create_robothor_dataset(
        args,
        width=300,
        height=300,
        objects_filter=None,
        scene_filter=None,
        filter_file=None
):
    """
    Creates a dataset for the robothor challenge in `intermediate_directory`
    named `robothor-dataset.json`
    """

    scene = 'FloorPlan_Train1_1'
    angle = args.rotate_by
    grid_size = args.grid_size
    visibility_distance = args.visibility_distance
    targets = args.targets
    desired_points = args.desired_points
    rotations = list(range(0, 360, args.rotate_by))
    # Restrict points visibility_multiplier_filter * visibility_distance away from the target object
    visibility_multiplier_filter = 2

    # scene_object_filter = {}
    # if filter_file is not None:
    #     with open(filter_file, 'r') as f:
    #         scene_object_filter = json.load(f)
    #         print("Filter:")
    #         pprint(scene_object_filter)

    print("Visibility distance: {}".format(visibility_distance))
    controller = ai2thor.controller.Controller(
        width=width,
        height=height,
        scene=scene,
        # Unity params
        gridSize=grid_size,
        rotateStepDegrees=angle,
        agentMode='bot',
        visibilityDistance=visibility_distance,
    )

    failed_points = []

    if objects_filter is not None:
        obj_filter = set([o for o in objects_filter.split(",")])
        targets = [o for o in targets if o.replace(" ", "") in obj_filter]


    # event = controller.step(
    #     dict(
    #         action='GetScenesInBuild',
    #     )
    # )
    # scenes_in_build = event.metadata['actionReturn']

    objects_types_in_scene = set()

    curriculum_meta = {
        "num_ep_per_stage": args.num_ep_per_stage,
        "distance_upgrade_step": args.distance_upgrade_step,
        "penalty_decay": args.penalty_decay,
        "episodes":{}
    }

    episodes = {}
    # dataset_flat = []

    if not os.path.exists(args.out_dir ):
        os.makedirs(args.out_dir)


    scenes = args.scenes

    # scenes = sorted(
    #     [scene for scene in scenes_in_build if 'physics' not in scene],
    #     key=key_sort_func
    # )
    # if scene_filter is not None:
    #     scene_filter_set = set([o for o in scene_filter.split(",")])
    #     scenes = [s for s in scenes if s in scene_filter_set]

    print("Sorted scenes: {}".format(scenes))
    for scene in scenes:
        scene_episodes = [ [] for i in range(args.max_diff_level)]
        for objectType in targets:

            points = get_points(controller, objectType, scene, objects_types_in_scene,
                             failed_points, grid_size, args.rotate_by, desired_points)
            if points is not None:
                split_by_difficulty(points, scene_episodes, args.distance_upgrade_step, max_diff_level=args.max_diff_level)

        if args.dump_each_scene:
            scene_out_file = "{}_curriculum_{}_{}_{}.json".format(
                scene,
                args.num_ep_per_stage,
                args.distance_upgrade_step,
                args.penalty_decay
            )
            if not os.path.exists(os.path.join(args.out_dir, scene, scene_out_file)):
                os.makedirs(os.path.join(args.out_dir, scene))

            with open(os.path.join(args.out_dir, scene, scene_out_file), 'w') as f:
                scene_curriculum_meta = {
                    "num_ep_per_stage": args.num_ep_per_stage,
                    "distance_upgrade_step": args.distance_upgrade_step,
                    "penalty_decay": args.penalty_decay,
                    "episodes": {scene:scene_episodes}
                }

                json.dump(scene_curriculum_meta, f, indent=4)

        # collect episodes for this scene
        episodes[scene] = scene_episodes

    curriculum_meta['episodes'] = episodes

    out_file_name = "curriculum_{}_{}_{}.json".format(
        args.num_ep_per_stage,
        args.distance_upgrade_step,
        args.penalty_decay
    )
    with open(os.path.join(args.out_dir, out_file_name), 'w') as f:
        json.dump(curriculum_meta, f, indent=4)
    print("Object types in scene union: {}".format(objects_types_in_scene))
    print("Total unique objects: {}".format(len(objects_types_in_scene)))
    print("Total scenes: {}".format(len(scenes)))
    print("Scene datapoints: ")
    for scene in scenes:
        print("    {}: {}".format(scene, len(
            [ep for diff_list in curriculum_meta['episodes'][scene]
             for ep in diff_list]
        )))
    # print("Total datapoints: {}".format(len(dataset_flat)))

    print(failed_points)
    failed_file_name = "failed_{}_{}_{}.json".format(
        args.num_ep_per_stage,
        args.distance_upgrade_step,
        args.penalty_decay
    )
    with open(os.path.join(args.out_dir, failed_file_name), 'w') as f:
        json.dump(failed_points, f, indent=4)


if __name__ == '__main__':
    args = parse_arguments()
    create_robothor_dataset(args, args.player_size, args.player_size)
