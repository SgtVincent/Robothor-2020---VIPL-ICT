from __future__ import print_function, division

import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.multiprocessing as mp

import time
import numpy as np
import random
import json
from tqdm import tqdm
import cProfile

from utils.net_util import ScalarMeanTracker
from runners import nonadaptivea3c_val, savn_val
from pandas import Series, DataFrame

def prof_target(target, rank, args, model_to_open, create_shared_model, init_agent,
                res_queue, max_val_ep, scene_type):
    cProfile.runctx('target(rank, args, model_to_open, create_shared_model, init_agent,res_queue, max_val_ep, scene_type)',
                    globals(), locals(), 'prof_result/prof{}.prof'.format(rank))

def main_eval(args, create_shared_model, init_agent):
    # 设置随即数种子i
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

    model_to_open = args.load_model

    processes = []

    res_queue = mp.Queue()

    if args.model == "SAVN":
        args.learned_loss = True
        args.num_steps = 6
        target = savn_val
    else:
        args.learned_loss = False
        args.num_steps = args.max_episode_length
        target = nonadaptivea3c_val


    rank = 0
    for scene_type in args.scene_types:
        p = mp.Process(
            target=prof_target,
            args=(
                target,
                rank,
                args,
                model_to_open,
                create_shared_model,
                init_agent,
                res_queue,
                args.max_val_ep,
                scene_type,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
        rank += 1

    count = 0
    end_count = 0
    all_train_scalars = ScalarMeanTracker()
    # analyze performance for each scene_type
    scene_train_scalars = {scene_type:ScalarMeanTracker() for scene_type in args.scene_types}
    # analyze performance for each difficulty level
    if args.curriculum_learning:
        diff_train_scalars = {}

    proc = len(args.scene_types)
    # pbar = tqdm(total=args.max_val_ep * proc)

    try:
        while end_count < proc:
            train_result = res_queue.get()
            # pbar.update(1)
            count += 1
            print("{} episdoes evaluated...".format(count))
            if "END" in train_result:
                end_count += 1
                continue
            # analysis performance for each difficulty split
            if args.curriculum_learning:
                diff = train_result['difficulty']
                if diff not in diff_train_scalars:
                    diff_train_scalars[diff] = ScalarMeanTracker()
                diff_train_scalars[diff].add_scalars(train_result)
            # analysis performance for each scene_type
            scene_train_scalars[train_result["scene_type"]].add_scalars(train_result)
            all_train_scalars.add_scalars(train_result)

        all_tracked_means = all_train_scalars.pop_and_reset()
        scene_tracked_means = {scene_type: scene_train_scalars[scene_type].pop_and_reset()
                             for scene_type in args.scene_types}
        if args.curriculum_learning:
            diff_tracked_means = {diff: diff_train_scalars[diff].pop_and_reset()
                                  for diff in diff_train_scalars}

    finally:
        for p in processes:
            time.sleep(0.1)
            p.join()

    if args.curriculum_learning:
        result = {"all_result":all_tracked_means,
                  "diff_reult":diff_tracked_means,
                  "scene_result":scene_tracked_means}
    else:
        result = {"all_result":all_tracked_means,
                  "scene_result":scene_tracked_means}

    with open(args.results_json, "w") as fp:
        json.dump(result, fp, sort_keys=True, indent=4)

    print("\n\n\nall_result:\n")
    print(Series(all_tracked_means))
    print("\n\n\nscene_result:\n")
    print(DataFrame(scene_tracked_means))
    if args.curriculum_learning:
        print("\n\n\ndiff_result:\n")
        print(DataFrame(diff_tracked_means))
