from .constants import (
    ROBOTHOR_ORIGINAL_CLASS_LIST
)
import re
import os
import json
import networkx
import h5py
import numpy as np

scene_types = ['FloorPlan_Train1', 'FloorPlan_Train2', 'FloorPlan_Train3',
               'FloorPlan_Train4', 'FloorPlan_Train5', 'FloorPlan_Train6',
               'FloorPlan_Train7', 'FloorPlan_Train8', 'FloorPlan_Train9',
               'FloorPlan_Train10', 'FloorPlan_Train11', 'FloorPlan_Train12']

DIFFICULTY = ['easy', 'medium', 'hard']

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

def preload_metadata(args, scene_types,
                     train_scenes="[1-5]",
                     grid_file_name="grid.json",
                     graph_file_name="graph.json",
                     metadata_file_name="visible_object_map.json",
                     ):

    metadata = {}
    i,j = re.findall(r"\d+", train_scenes)
    # load all metadata to dictionary
    for scene_type in scene_types:
        for scene_name in [scene_type + "_{}".format(k) for k in range(int(i), int(j)+1)]:
            metadata[scene_name] = {}

            with open(os.path.join(args.offline_data_dir, scene_name, grid_file_name),"r",) as f:
                metadata[scene_name]['grid'] = json.load(f)

            with open(os.path.join(args.offline_data_dir, scene_name, graph_file_name),"r") as f:
                graph_json = json.load(f)
            metadata[scene_name]['graph_json'] = graph_json
            metadata[scene_name]['graph'] = networkx.readwrite.node_link_graph(graph_json).to_directed()

            with open(os.path.join(args.offline_data_dir, scene_name, metadata_file_name),"r") as f:
                metadata[scene_name]['metadata'] = json.load(f)
            with  h5py.File(os.path.join(args.offline_data_dir, scene_name, args.images_file_name), "r") as images:
                metadata[scene_name]['all_states'] = list(images.keys())

    return metadata

def get_curriculum_meta(args, scenes):
    scenes = np.array(scenes).reshape(-1)
    curriculum_meta = {}
    with open(args.episode_file) as f:
        episodes = json.loads(f.read())
        # Build a dictionary of the dataset indexed by scene->difficulty->object_type
    for e in episodes:

            scene = e['scene']
            difficulty = e['difficulty']
            object_type = e['object_type']

            # omit scene not in scenes
            if scene not in scenes:
                continue

            if scene not in curriculum_meta:
                curriculum_meta[scene] = {}
            if difficulty not in curriculum_meta[scene]:
                curriculum_meta[scene][difficulty] = {}
            if object_type not in curriculum_meta[scene][difficulty]:
                curriculum_meta[scene][difficulty][object_type] = [e]
            else:
                curriculum_meta[scene][difficulty][object_type].append(e)

    return curriculum_meta






