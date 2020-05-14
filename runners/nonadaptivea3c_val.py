from __future__ import division

import time
import torch
import setproctitle
import copy
from datasets.glove import Glove
from datasets.prototypes import Prototype

from datasets.robothor_data import (
    preload_metadata,
    get_curriculum_meta,
    load_offline_shortest_path_data
)

from models.model_io import ModelOptions
import os

from .train_util import (
    compute_loss,
    new_episode,
    run_episode,
    end_episode,
    reset_player,
    compute_spl,
    get_bucketed_metrics,
)


def nonadaptivea3c_val(
    rank,
    args,
    model_to_open,
    model_create_fn,
    initialize_agent,
    res_queue,
    max_count,
    scene_type,
):

    glove = None
    protos = None
    pre_metadata = None
    curriculum_meta = None
    scene_types = [scene_type]
    offline_shortest_data = None


    if args.glove_file:
        glove = Glove(args.glove_file)
    if args.proto_file:
        protos = Prototype(args.proto_file)

    if args.data_source == "ithor":

        from datasets.ithor_data import get_data, name_to_num

        scenes, possible_targets, targets = get_data(scene_types, args.val_scenes)
        num = name_to_num(scene_type)
        scenes = scenes[0]
        targets = targets[0]

    elif args.data_source == "robothor":

        from datasets.robothor_data import get_data
        # TODO: design a flexible scene allocating strategy

        pre_metadata = preload_metadata(args, scene_types)

        scenes, possible_targets, targets = get_data(scene_types)
        scenes = scenes[0]
        targets = targets[0]

        if args.curriculum_learning:
            curriculum_meta = get_curriculum_meta(args, scenes)
            if args.offline_shortest_data:
                offline_shortest_data = load_offline_shortest_path_data(args, scenes)


    setproctitle.setproctitle("Val Agent: {}".format(rank))

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    shared_model = model_create_fn(args)

    if model_to_open != "":
        saved_state = torch.load(
            model_to_open, map_location=lambda storage, loc: storage
        )
        shared_model.load_state_dict(saved_state)

    player = initialize_agent(model_create_fn, args, rank, gpu_id=gpu_id)
    player.sync_with_shared(shared_model)
    count = 0

    model_options = ModelOptions()

    while count < max_count:

        # Get a new episode.
        total_reward = 0
        player.eps_len = 0
        scene = new_episode(args, player, scenes, possible_targets, targets, glove=glove, protos=protos,
            pre_metadata=pre_metadata, curriculum_meta=curriculum_meta)
        if scene == None: # iteration stopped
            break

        player_start_state = copy.deepcopy(player.environment.controller.state)
        player_start_time = time.time()

        # Train on the new episode.
        while not player.done:

            # Make sure model is up to date.
            # player.sync_with_shared(shared_model)
            # Run episode for num_steps or until player is done.
            total_reward = run_episode(player, args, total_reward, model_options, False)
            # Compute the loss.
            # loss = compute_loss(args, player, gpu_id, model_options)
            if not player.done:
                reset_player(player)

        # for k in loss:
        #     loss[k] = loss[k].item()
        if offline_shortest_data: # assume data_source == robothor and curriculum_learning is True
            scene = player.environment.scene_name
            episode_id = player.episode.episode_id
            best_path_length = offline_shortest_data[scene][episode_id]
            spl = player.success * (best_path_length / float(player.eps_len))
        else:
            spl, best_path_length = compute_spl(player, player_start_state)

        bucketed_spl = get_bucketed_metrics(spl, best_path_length, player.success)
        if args.curriculum_learning:
            end_episode(
                player,
                res_queue,
                total_time=time.time() - player_start_time,
                total_reward=total_reward,
                spl=spl,
                **bucketed_spl,
                scene_type=scene_type,
                difficulty=player.episode.difficulty
            )   
        else:
            end_episode(
                player,
                res_queue,
                total_time=time.time() - player_start_time,
                total_reward=total_reward,
                spl=spl,
                **bucketed_spl,
                scene_type=scene_type,
            )

        count += 1
        reset_player(player)

    player.exit()
    res_queue.put({"END":True, "scene_type":scene_type, "total_episodes": count})
