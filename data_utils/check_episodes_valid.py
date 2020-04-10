import os
import numpy as np
import h5py
import argparse
import json
import sys
sys.path.append('.')
from datasets.thor_agent_state import  ThorAgentState


def parse_arguments():
    parser = argparse.ArgumentParser(description="scrape all possible images from ai2thor scene")

    parser.add_argument(
        "--episode_file_path",
        type=str,
        default="./data/train.json",
        help="path where episodes file stored",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/ubuntu/Robothor_data",
        help="path where ai2thor scene images stored",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="",
        help="specify scenes to scrape, in the format of 'scene1,scene2,...'"
    )
    parser.add_argument(
        "--state_decimal",
        type=int,
        default=3,
        help="decimal of key in state data: e.g. images.hdf5"
    )
    parser.add_argument(
        "--grid_size",
        type=float,
        default=0.125,
        help="grid size of controller"
    )
    parser.add_argument(
        "check_in_ai2thor",
        action="store_true",
        default=False,
        help="if check all error_episodes in ai2thor"
    )
    # parser.add_argument(
    #     "--num_process",
    #     type=int,
    #     default=4,
    #     help="number of processes launched to scrape images parallelly",
    # )
    args = parser.parse_args()
    return args


def get_episodes(file_path, scenes):
    scenes = np.array(scenes).reshape(-1)
    curriculum_meta = {}
    with open(file_path) as f:
        episodes = json.loads(f.read())
        # Build a dictionary of the dataset indexed by scene->difficulty->object_type
    for e in episodes:

            scene = e['scene']
            object_type = e['object_type']

            # omit scene not in scenes
            if scene not in scenes:
                continue
            if scene not in curriculum_meta:
                curriculum_meta[scene] = {}
            if object_type not in curriculum_meta[scene]:
                curriculum_meta[scene][object_type] = [e]
            else:
                curriculum_meta[scene][object_type].append(e)

    return curriculum_meta

def check_episode(e, scene_states, state_decimal):
    # check state
    state = ThorAgentState(**e['initial_position'], rotation=e['initial_orientation'],
                           horizon=0, state_decimal=state_decimal)
    scene = e['scene']
    if str(state) not in scene_states:
        print("Episode {} not in scene {}, with initial state {}".format(e['id'], scene, str(state)))
        return False
    return True

if __name__ == '__main__':
    args = parse_arguments()
    if args.scenes == "":
        scenes = ["FloorPlan_Train{}_{}".format(i, j) for i in range(1,13) for j in range(1,6)]
    else:
        scenes = args.scenes.split(',')
    episodes = get_episodes(args.episode_file_path, scenes)

    # get all sta
    all_states = {}
    for scene in scenes:
        with h5py.File(os.path.join(args.data_dir, scene, "resnet18.hdf5"), 'r') as f:
            states = list(f.keys())
            all_states[scene] = states

    error_episodes = []
    for scene in scenes:
        print("checking episodes in scene {}".format(scene))
        for obj in episodes[scene].keys():
            for e in episodes[scene][obj]:
                if not check_episode(e, all_states[scene], args.state_decimal):
                    error_episodes.append(e)

    # if args.check_in_ai2thor:
    #     from ai2thor.controller import Controller
    #     controller = Controller(scene='FloorPlan_Train1_1', agentMode='bot', gridSize=0.125)
    #     for err_ep in error_episodes:
    #         scene =

