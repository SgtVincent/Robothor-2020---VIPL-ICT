from __future__ import print_function, division

import os
import random
import ctypes
import setproctitle
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from utils import flag_parser

from utils.class_finder import model_class, agent_class, optimizer_class
from utils.net_util import ScalarMeanTracker
from main_eval import main_eval
from runners.train_util import load_checkpoint

from runners import nonadaptivea3c_train, nonadaptivea3c_val, savn_train, savn_val


os.environ["OMP_NUM_THREADS"] = "1"


def main():
    # 设置进程名称
    setproctitle.setproctitle("Train/Test Manager")

    # 获取命令行参数
    args = flag_parser.parse_arguments()

    if args.model == "BaseModel" or args.model == "GCN":
        args.learned_loss = False
        args.num_steps = 50
        target = nonadaptivea3c_val if args.eval else nonadaptivea3c_train
    else:
        args.learned_loss = True
        args.num_steps = 6
        target = savn_val if args.eval else savn_train

    # 检查pinned_scene 和 data_source 是否冲突
    if args.data_source == "ithor" and args.pinned_scene == True:
        raise Exception("Cannot set pinned_scene to true when using ithor dataset")

    # 获取模型对象类别， 未创建对象 e.g. <class 'models.basemodel.BaseModel'>
    create_shared_model = model_class(args.model)
    # 获取agent类别，未创建对象 default <class 'agents.navigation_agent.NavigationAgent'>
    init_agent = agent_class(args.agent_type)
    # 获取优化器对象类别，未创建对象 default <class 'optimizers.shared_adam.SharedAdam'>
    optimizer_type = optimizer_class(args.optimizer)
########################  测试阶段 ################################
    if args.eval:
        main_eval(args, create_shared_model, init_agent)
        return
####################### 训练阶段 #################################
    start_time = time.time()
    local_start_time_str = time.strftime(
        "%Y-%m-%d_%H:%M:%S", time.localtime(start_time)
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)


    # 设置日志参数
    if args.log_dir is not None:
        tb_log_dir = args.log_dir + "/" + args.title + "-" + local_start_time_str
        log_writer = SummaryWriter(log_dir=tb_log_dir)
    else:
        log_writer = SummaryWriter(comment=args.title)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method("spawn")

    # 创建一个 torch.nn.Module的子类对象
    shared_model = create_shared_model(args)

    optimizer = optimizer_type(
        filter(lambda p: p.requires_grad, shared_model.parameters()), args
    )
    # 加载预先保存的模型
    train_total_ep, n_frames = load_checkpoint(args, shared_model, optimizer)

    # 总训练episode
    # train_total_ep = 0
    # n_frames = 0

    if shared_model is not None:
        # 模型在多进程间共享参数 这个参数是torch.mutiprocessing 调用fork之前必须调用的方法
        shared_model.share_memory()
        # 创建一个 torch.optim.Optimizer的子类对象
        # filter 函数把model中所有需要梯度更新的变量 作为参数送到optimizer的constructor中

        optimizer.share_memory()
        print(shared_model)
    else:
        assert (
            args.agent_type == "RandomNavigationAgent"
        ), "The model is None but agent is not random agent"
        optimizer = None

    processes = []

    end_flag = mp.Value(ctypes.c_bool, False)
    # 多进程共享资源队列
    train_res_queue = mp.Queue()
    # 创建多进程
    # target 进程执行目标函数
    #
    for rank in range(0, args.workers):
        p = mp.Process(
            target=target,
            args=(
                rank,
                args,
                create_shared_model,
                shared_model,
                init_agent,
                optimizer,
                train_res_queue,
                end_flag,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)

    print("Train agents created.")

    train_thin = args.train_thin
    train_scalars = ScalarMeanTracker()

    # 主线程
    try:
        while train_total_ep < args.max_ep:

            train_result = train_res_queue.get()
            train_scalars.add_scalars(train_result)
            train_total_ep += 1
            n_frames += train_result["ep_length"]
            if (train_total_ep % train_thin) == 0:
                log_writer.add_scalar("n_frames", n_frames, train_total_ep)
                tracked_means = train_scalars.pop_and_reset()
                for k in tracked_means:
                    log_writer.add_scalar(
                        k + "/train", tracked_means[k], train_total_ep
                    )

            if (train_total_ep % args.ep_save_freq) == 0:

                print(n_frames)
                if not os.path.exists(args.save_model_dir):
                    os.makedirs(args.save_model_dir)
                state_to_save = shared_model.state_dict()
                save_path = os.path.join(
                    args.save_model_dir,
                    "{0}_{1}_{2}_{3}.dat".format(
                        args.title, n_frames, train_total_ep, local_start_time_str
                    ),
                )
                torch.save(state_to_save, save_path)

            if (train_total_ep % args.ep_save_ckpt) == 0:
                print("save check point at episode {}".format(train_total_ep))
                checkpoint = {
                    'train_total_ep': train_total_ep,
                    'n_frames': n_frames,
                    'shared_model': shared_model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                checkpoint_path = os.path.join(args.save_model_dir, "checkpoint.dat")
                torch.save(checkpoint, checkpoint_path)

    finally:
        log_writer.close()
        end_flag.value = True
        for p in processes:
            time.sleep(0.1)
            p.join()


if __name__ == "__main__":
    main()
