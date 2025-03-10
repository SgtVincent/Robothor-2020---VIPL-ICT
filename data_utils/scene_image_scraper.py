import json
import os
import sys
from collections import deque
from math import gcd
from multiprocessing import Process, Queue
import argparse
sys.path.append(".") # Assume script run in project root directory
from ai2thor.controller import BFSController
from datasets.offline_controller_with_small_rotation_backup_0320 import ExhaustiveBFSController

PATH_TO_STORE = ''


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
        default=8,
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
        try:
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
                depth_file=os.path.join(out_dir, 'depth.hdf5'),
                grid_assumption=False,
                rotate_by=45,
                state_decimal=2
            )
            c.start()
            c.search_all_closed(scene_name)
            c.stop()
        except AssertionError as e:
            print('Error is', e)
            print('Error in scene {}'.format(scene_name))
            if c is not None:
                c.stop()
            continue


def main():

    args = parse_arguments()
    out_dir = args.out_dir
    num_processes = args.num_process

    queue = Queue()
    scene_names = []
    for i in range(2):
        for j in range(30):
            if i == 0:
                scene_names.append("FloorPlan" + str(j + 1))
            else:
                scene_names.append("FloorPlan" + str(i + 1) + '%02d' % (j + 1))
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