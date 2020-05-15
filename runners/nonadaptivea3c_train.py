from __future__ import division

import time


from datasets.glove import Glove
from datasets.prototypes import Prototype

import setproctitle

from models.model_io import ModelOptions

from agents.random_agent import RandomNavigationAgent

import random

from datasets.robothor_data import preload_metadata, get_curriculum_meta

from .train_util import (
    compute_loss,
    new_episode,
    run_episode,
    transfer_gradient_from_player_to_shared,
    end_episode,
    reset_player
)
from demo_robothor.trajectory import *


def nonadaptivea3c_train(
    rank,
    args,
    create_shared_model,
    shared_model,
    initialize_agent,
    optimizer,
    res_queue,
    end_flag,
    global_ep,
):

    glove = None
    protos = None
    pre_metadata = None
    curriculum_meta = None
    scene_types = args.scene_types

    if args.glove_file:
        glove = Glove(args.glove_file)
    if args.proto_file:
        protos = Prototype(args.proto_file)

    if args.data_source == "ithor":
        from datasets.ithor_data import get_data
        scenes, possible_targets, targets = get_data(scene_types, args.train_scenes)

    elif args.data_source == "robothor":

        from datasets.robothor_data import get_data

        # check if use pinned_scene mode
        if args.pinned_scene:
            # TODO: design a flexible scene allocating strategy
            scene_types = [scene_types[(rank % len(scene_types))]]
            pre_metadata = preload_metadata(args, scene_types)

        scenes, possible_targets, targets = get_data(scene_types)

        if args.curriculum_learning:
            curriculum_meta = get_curriculum_meta(args, scenes)


    # is pinned_scene set to True, pre-load all metadata for controller
    # constructed in new_episode()


    random.seed(args.seed + rank)
    idx = list(range(len(scene_types)))
    random.shuffle(idx)

    setproctitle.setproctitle("Training Agent: {}".format(rank))

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]

    import torch

    torch.cuda.set_device(gpu_id)

    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    player = initialize_agent(create_shared_model, args, rank, gpu_id=gpu_id)
    compute_grad = not isinstance(player, RandomNavigationAgent)

    model_options = ModelOptions()

    j = 0

    while not end_flag.value:

        # Get a new episode.
        total_reward = 0
        player.eps_len = 0
        # new_episode(args, player, scenes[idx[j]], possible_targets, targets[idx[j]],glove=glove, protos=protos,
        #     pre_metadata=pre_metadata, curriculum_meta=curriculum_meta, total_ep=global_ep.value)
        scene = new_episode(args, player, scenes[idx[j]], possible_targets, targets[idx[j]],glove=glove, protos=protos,
            pre_metadata=pre_metadata, curriculum_meta=curriculum_meta)
        player_start_time = time.time()

        # Train on the new episode.
        while not player.done:
            # Make sure model is up to date.
            player.sync_with_shared(shared_model)
            # Run episode for num_steps or until player is done.
            total_reward = run_episode(player, args, total_reward, model_options, True)

            # plot trajectory , by wuxiaodong
            if args.demo_trajectory and global_ep.value % args.demo_trajectory_freq == 0:
                print(len(player.episode.episode_trajectories))
                # todo delete
                # scene = 'FloorPlan_Train1_1'
                trajectory_pil = get_trajectory(scene,
                                                [str(loc) for loc in player.episode.episode_trajectories],
                                                birdview_root='./demo_robothor/data/birdview/',
                                                init_loc_str=player.episode.init_pos_str,
                                                target_loc_str=player.episode.target_pos_str,
                                                actions=player.episode.actions_taken,
                                                success=player.success, target_name=player.episode.target_object)
                demo_out_dir = os.path.join(args.log_dir, '../output_trajecgtory', args.title)
                if not os.path.exists(demo_out_dir):
                    os.makedirs(demo_out_dir)
                trajectory_pil.save(os.path.join(demo_out_dir, '{}_init_{}_target_{}_iter{}.png'.format(
                    player.episode.object_type,
                    player.episode.init_pos_str,
                    player.episode.target_pos_str,
                    global_ep.value
                )))
                print('ploting {}_init_{}_target_{}_iter{}.png'.format(
                    player.episode.object_type,
                    player.episode.init_pos_str,
                    player.episode.target_pos_str,
                    global_ep.value
                ))

            # Compute the loss.
            loss = compute_loss(args, player, gpu_id, model_options)
            if compute_grad:
                # Compute gradient.
                player.model.zero_grad()
                loss["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(player.model.parameters(), 100.0)
                # Transfer gradient to shared model and step optimizer.
                transfer_gradient_from_player_to_shared(player, shared_model, gpu_id)
                optimizer.step()
            # Clear actions and repackage hidden.
            if not player.done:
                reset_player(player)

        # print("Training Agent {}: finished episodes on {}, local loss {}".format(
        #     rank, scene, loss.cpu().detach().numpy() ))

        for k in loss:
            loss[k] = loss[k].item()

        end_episode(
            player,
            res_queue,
            title=scene_types[idx[j]],
            total_time=time.time() - player_start_time,
            total_reward=total_reward,
            policy_loss=loss['policy_loss'],
            value_loss=loss['value_loss']
        )

        reset_player(player)

        j = (j + 1) % len(scene_types)

    player.exit()
