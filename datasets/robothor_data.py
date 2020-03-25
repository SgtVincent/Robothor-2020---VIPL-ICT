from .constants import (
    ROBOTHOR_ORIGINAL_CLASS_LIST
)
# TODO: change back to all after all data scraped
scene_types = ['FloorPlan_Train1', 'FloorPlan_Train2', 'FloorPlan_Train3',
               'FloorPlan_Train4', 'FloorPlan_Train5', 'FloorPlan_Train6',
               'FloorPlan_Train7', 'FloorPlan_Train8', 'FloorPlan_Train9',
               'FloorPlan_Train10', 'FloorPlan_Train11', 'FloorPlan_Train12']

# scene_types = ['FloorPlan_Train2',
#                'FloorPlan_Train6',
#                'FloorPlan_Train7', 'FloorPlan_Train8',
#                'FloorPlan_Train10', 'FloorPlan_Train11', 'FloorPlan_Train12']

def get_scenes(scene_type):
    # scene_type: "FloorPlan_TrainX" or "FloorPlan_ValY"
    return [scene_type + "_{}".format(i) for i in range(1,6)]

# TODO: modify code relative to these two functions in test_val_episode.py and nonadaptivea3c_val.py
def name_to_num(name):
    return scene_types.index(name)


def num_to_name(num):
    return scene_types[num-1]


def get_data(scene_types):
    idx = []
    for j in range(len(scene_types)):
        idx.append(scene_types.index(scene_types[j]))

    scenes = [
        get_scenes(scene_type) for  scene_type in scene_types
    ]

    possible_targets = ROBOTHOR_ORIGINAL_CLASS_LIST
    # dump code since object class for all scene type are the same
    # TODO: modify this code when using customized targets ...
    targets = [ROBOTHOR_ORIGINAL_CLASS_LIST] * 12

    return scenes, possible_targets, [targets[i] for i in idx]
