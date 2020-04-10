import json
import os
import sys
sys.path.append(".") # Assume script run in project root directory
from multiprocessing import Process, Queue
import argparse

from ai2thor.controller import BFSController
from datasets.offline_sscontroller import SSController

def parse_arguments():
    parser = argparse.ArgumentParser(description="scrape all possible images from ai2thor scene")

    parser.add_argument(
        "--out_dir",
        type=str,
        default='/home/chenjunting/Robothor_data',
        help="path to store scraped images",
    )

    parser.add_argument(
        "--num_process",
        type=int,
        default=12,
        help="number of processes launched to scrape images parallelly",
    )

    parser.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="specify scenes to scrape, in the format of 'scene1,scene2,...'"
    )
    parser.add_argument("--state_decimal", type=int, default=3, help="decimal of key in state data: e.g. images.hdf5")

    args = parser.parse_args()
    return args

def search_and_save(in_queue, out_dir, rank, gpus=[0,1,2,3]):

    gpu_id = gpus[rank % len(gpus)]
    os.system("export CUDA_VISIBLE_DEVICES={}".format(gpu_id))

    while not in_queue.empty():
        try:
            scene_name = in_queue.get(timeout=3)
        except:
            return
        c = None
        # try:
        sub_out_dir = os.path.join(out_dir, scene_name)
        if not os.path.exists(sub_out_dir):
            os.mkdir(sub_out_dir)

        print('starting:', scene_name)

        c = SSController(
            grid_size=0.125,
            grid_file=os.path.join(sub_out_dir, 'grid.json'),
            graph_file=os.path.join(sub_out_dir, 'graph.json'),
            metadata_file=os.path.join(sub_out_dir, 'metadata.json'),
            images_file=os.path.join(sub_out_dir, 'images.hdf5'),
            # depth_file=os.path.join(sub_out_dir, 'depth.hdf5'), # no depth data allowed in robothor-challenge
            grid_assumption=False,
            rotate_by=30,
            state_decimal=3,
            ai2thor_args={
                'start_unity':True,
                'width':640,
                'height':480,
                'agentMode':'bot',
                'gridSize':0.125,
            })
        # c.start()
        c.search_all_closed(scene_name)
        c.stop()
        # except AssertionError as e:
        #     print('Error is', e)
        #     print('Error in scene {}'.format(scene_name))
        #     if c is not None:
        #         c.stop()
        #     continue


def main():

    args = parse_arguments()
    out_dir = args.out_dir
    num_processes = args.num_process

    queue = Queue()
    if args.scenes:
        scenes = args.scenes.split(',')
    else: # all scenes in robothor
        scenes = ["FloorPlan_Train{}_{}".format(i, j) for i in range(1,13) for j in range(1,6)]
    for scene in scenes:
        queue.put(scene)

    processes = []
    for i in range(num_processes):
        p = Process(target=search_and_save, args=(queue, out_dir, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()