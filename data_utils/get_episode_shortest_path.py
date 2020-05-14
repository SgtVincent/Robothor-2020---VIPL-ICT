import json
import os
import re
import numpy as np
import argparse
from concurrent import futures

import sys
sys.path.append("./")
from agents.navigation_agent import NavigationAgent
from utils.flag_parser import parse_arguments
from datasets.robothor_data import get_curriculum_meta, preload_metadata
from datasets.constants import ROBOTHOR_ORIGINAL_CLASS_LIST
from datasets.glove import Glove
from datasets.prototypes import Prototype
from runners.train_util import new_episode
from datasets.thor_agent_state import ThorAgentState
from utils.class_finder import episode_class
from models.basemodel import BaseModel

def init_episode(
        args,
        scenes,
        possible_targets=None,
        targets=None,
        keep_obj=False,
        glove=None,
        protos=None,
        pre_metadata=None,
        curriculum_meta=None):

        episode_constructor = episode_class(args.episode_type)
        episode = episode_constructor(args, gpu_id, args.strict_done)
        episode.new_episode(args=args,
                           scenes=scenes,
                           possible_targets=possible_targets,
                           targets=targets,
                           keep_obj=keep_obj,
                           glove=glove,
                           protos=protos,
                           pre_metadata=pre_metadata,
                           curriculum_meta=curriculum_meta)
        return episode


def get_shorest_path_for_scene(rank, args, scene, gpu_id):
    
    print("processing scene %s" % scene)
    glove=[]
    protos=[]
    curriculum_meta = get_curriculum_meta(args, [scene])
    scene_type, scene_num = re.match('([a-zA-Z_]+\d+)_(\d+)',scene).groups()
    pre_metadata = preload_metadata(args, [scene_type],
                                    train_scenes='[{}-{}]'.format(scene_num,scene_num))
    if args.glove_file:
        glove = Glove(args.glove_file)
    if args.proto_file:
        protos = Prototype(args.proto_file)

    #
    episode_object = init_episode(args=args,
                                  scenes=[scene],
                                  possible_targets=ROBOTHOR_ORIGINAL_CLASS_LIST,
                                  targets=ROBOTHOR_ORIGINAL_CLASS_LIST,
                                  glove=glove,
                                  protos=protos,
                                  pre_metadata=pre_metadata,
                                  curriculum_meta=curriculum_meta)

    shortest_path_lens = {}
    count = 0
    for diff_eps in curriculum_meta[scene]:
        for ep in diff_eps:

            start_state = ThorAgentState(**ep['initial_position'],rotation=ep['initial_orientation'],
                                   horizon=0, state_decimal=args.state_decimal)
            obj_id = ep['object_id']
            _, len, _  = episode_object.environment.controller.shortest_path_to_target(
                start_state, obj_id, False
            )
            shortest_path_lens[ep['id']] = len
            print("episode id {}, difficulty {}, shortest_path_len is {}".format(ep['id'], ep['difficulty'], len))
            count += 1

    with open(os.path.join(args.curriculum_meta_dir, scene, "shortest_path_len.json"), 'w') as out_file:
        json.dump(shortest_path_lens, out_file)
    print("finished scene {}: calculated {} episodes in total".format(scene, count))
    return scene

if __name__ == '__main__':
    args = parse_arguments()
    scenes = []
    if args.data_source == "ithor":
        from datasets.ithor_data import get_data
        scenes, possible_targets, targets = get_data(args.scene_types, args.train_scenes)

    elif args.data_source == "robothor":
        from datasets.robothor_data import get_data
        scenes, possible_targets, targets = get_data(args.scene_types)

    scenes = np.array(scenes).reshape(-1)
    fs = []

    # sequential run, for debug only
    # for i, scene in enumerate(scenes):
    #     gpu_id = args.gpu_ids[i % len(args.gpu_ids)]
    #     print("processing scene %s" % scene)
    #     get_shorest_path_for_scene(i, args, scene, gpu_id)

    # parallel run
    with futures.ProcessPoolExecutor(max_workers=args.workers) as executor:

        for i, scene in enumerate(scenes):
            gpu_id = args.gpu_ids[i % len(args.gpu_ids)]
            
            fs.append(executor.submit(
                get_shorest_path_for_scene, i, args, scene, gpu_id))
    for future in futures.as_completed(fs):
        pass