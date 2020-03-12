import json
import os
import time
import warnings
from collections import deque
from math import gcd
from multiprocessing import Process, Queue
import argparse

from ai2thor.controller import BFSController
from datasets.my_bfscontroller import ExhaustiveBFSController

PATH_TO_STORE = '/data/robothor'


def parse_arguments():
    parser = argparse.ArgumentParser(description="scrape all possible images from ai2thor scene")

    parser.add_argument(
        "--out_dir",
        type=str,
        default=PATH_TO_STORE,
        help="path to store scraped images",
    )

    parser.add_argument(
        "--num_process",
        type=int,
        default=1,
        help="number of processes launched to scrape images parallelly",
    )

    args = parser.parse_args()
    return args

def search_and_save(in_queue, out_dir):
    while not in_queue.empty():
        try:
            scene_name = in_queue.get(timeout=3)
        except:
            return
        c = None
        # try:
        out_dir = os.path.join(out_dir, scene_name)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        print('starting:', scene_name)
        c = ExhaustiveBFSController(
            grid_size=0.25,
            fov=90.0,
            grid_file=os.path.join(out_dir, 'grid.json'),
            graph_file=os.path.join(out_dir, 'graph.json'),
            metadata_file=os.path.join(out_dir, 'metadata.json'),
            images_file=os.path.join(out_dir, 'images.hdf5'),
            # depth_file=os.path.join(out_dir, 'depth.hdf5'), # no depth data allowed in robothor-challenge
            grid_assumption=False)
        c.start()
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
    scene_names = ["FloorPlan_Train1_1"] # Robothor

    # scene_names = ["FloorPlan1"] # Ithor
    for x in scene_names:
        queue.put(x)

    processes = []
    for i in range(num_processes):
        p = Process(target=search_and_save, args=(queue, out_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()