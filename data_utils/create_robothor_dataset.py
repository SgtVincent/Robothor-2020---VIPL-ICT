import ai2thor.controller
import ai2thor.util.metrics as metrics
import json
from pprint import pprint

def create_robothor_dataset(
        local_build=False, # Whether to use local executable file for controller
        editor_mode=False, # Whether to start unity program to edit scene
        width=300,
        height=300,
        output='robothor-dataset.json',
        intermediate_directory='.', # dir to store dataset for robothor challenge
        visibility_distance=1.0,
        objects_filter=None,
        scene_filter=None,
        filter_file=None
    ):
    """
    Creates a dataset for the robothor challenge in `intermediate_directory`
    named `robothor-dataset.json`
    """

    scene = 'FloorPlan_Train1_1'
    angle = 45
    gridSize = 0.25
    # Restrict points visibility_multiplier_filter * visibility_distance away from the target object
    visibility_multiplier_filter = 2

    scene_object_filter = {}
    if filter_file is not None:
        with open(filter_file, 'r') as f:
            scene_object_filter = json.load(f)
            print("Filter:")
            pprint(scene_object_filter)

    print("Visibility distance: {}".format(visibility_distance))
    controller = ai2thor.controller.Controller(
        width=width,
        height=height,
        local_executable_path=_local_build_path() if local_build else None,
        start_unity=False if editor_mode else True,
        scene=scene,
        port=8200,
        host='127.0.0.1',
        # Unity params
        gridSize=gridSize,
        fieldOfView=60,
        rotateStepDegrees=angle,
        agentMode='bot',
        visibilityDistance=visibility_distance,
    )

    targets = [
        "Apple",
        "Baseball Bat",
        "BasketBall",
        "Bowl",
        "Garbage Can",
        "House Plant",
        "Laptop",
        "Mug",
        "Remote",
        "Spray Bottle",
        "Vase",
        "Alarm Clock",
        "Television",
        "Pillow"

    ]
    failed_points = []

    if objects_filter is not None:
        obj_filter = set([o for o in objects_filter.split(",")])
        targets = [o for o in targets if o.replace(" ", "") in obj_filter]

    desired_points = 30
    event = controller.step(
        dict(
            action='GetScenesInBuild',
        )
    )
    scenes_in_build = event.metadata['actionReturn']

    objects_types_in_scene = set()

    def sqr_dist(a, b):
        x = a[0] - b[0]
        z = a[2] - b[2]
        return x * x + z * z

    def sqr_dist_dict(a, b):
        x = a['x'] - b['x']
        z = a['z'] - b['z']
        return x * x + z * z

    def get_points(contoller, object_type, scene):
        print("Getting points in scene: '{}'...: ".format(scene))
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
                gridSize=0.25
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

        import random
        points = random.sample(reachable_pos_set, desired_points * 4)

        final_point_set = filter_points(points, reachable_pos_set, gridSize * 2)

        print("Total number of points: {}".format(len(final_point_set)))

        print("Id {}".format(event.metadata['actionReturn']))

        point_objects = []

        eps = 0.0001
        counter = 0
        for (x, y, z) in final_point_set:
            possible_orientations = [0, 90, 180, 270]
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


        sorted_objs = sorted(point_objects,
                             key=lambda m: sqr_dist_dict(m['initial_position'], m['target_position']))
        third = int(len(sorted_objs) / 3.0)

        for i, obj in enumerate(sorted_objs):
            if i < third:
                level = 'easy'
            elif i < 2 * third:
                level = 'medium'
            else:
                level = 'hard'

            sorted_objs[i]['difficulty'] = level

        return sorted_objs

    dataset = {}
    dataset_flat = []

    if intermediate_directory is not None:
        if intermediate_directory != '.':
            if os.path.exists(intermediate_directory):
                shutil.rmtree(intermediate_directory)
            os.makedirs(intermediate_directory)
    import re

    def key_sort_func(scene_name):
        m = re.search('FloorPlan_([a-zA-Z\-]*)([0-9]+)_([0-9]+)', scene_name)
        return m.group(1), int(m.group(2)), int(m.group(3))

    scenes = sorted(
        [scene for scene in scenes_in_build if 'physics' not in scene],
                    key=key_sort_func
                    )

    if scene_filter is not None:
        scene_filter_set = set([o for o in scene_filter.split(",")])
        scenes = [s for s in scenes if s in scene_filter_set]

    print("Sorted scenes: {}".format(scenes))
    for scene in scenes:
        dataset[scene] = {}
        dataset['object_types'] = targets
        objects = []
        for objectType in targets:

            if filter_file is None or (objectType in scene_object_filter and scene in scene_object_filter[objectType]):
                dataset[scene][objectType] = []
                obj = get_points(controller, objectType, scene)
                if obj is not None:

                    objects = objects + obj

        dataset_flat = dataset_flat + objects
        if intermediate_directory != '.':
            with open(os.path.join(intermediate_directory, '{}.json'.format(scene)), 'w') as f:
                json.dump(objects, f, indent=4)


    with open(os.path.join(intermediate_directory, output), 'w') as f:
        json.dump(dataset_flat, f, indent=4)
    print("Object types in scene union: {}".format(objects_types_in_scene))
    print("Total unique objects: {}".format(len(objects_types_in_scene)))
    print("Total scenes: {}".format(len(scenes)))
    print("Total datapoints: {}".format(len(dataset_flat)))

    print(failed_points)
    with open(os.path.join(intermediate_directory, 'failed.json'), 'w') as f:
        json.dump(failed_points, f, indent=4)
