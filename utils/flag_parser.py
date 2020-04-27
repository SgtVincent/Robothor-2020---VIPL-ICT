import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="SAVN.")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="learning rate (default: 0.0001)",
    )

    parser.add_argument(
        "--inner_lr",
        type=float,
        default=0.0001,
        metavar="ILR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--goal_success_reward",
        type=float,
        default=5,
        help="episode success reward (default: 0.01)",
    )
    parser.add_argument(
        "--step_penalty",
        type=float,
        default=-0.01,
        help="episode success reward (default: -0.01)",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.00,
        metavar="T",
        help="parameter for GAE (default: 1.00)",
    )
    parser.add_argument(
        "--beta", type=float, default=1e-2, help="entropy regularization term"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="W",
        help="how many training processes to use (default: 4)",
    )

    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=150,
        metavar="M",
        help="maximum length of an episode (default: 150)",
    )

    parser.add_argument(
        "--meta_train_episodes",
        type=int,
        default=1,
        help="how many meta-train episodes.",
    )

    parser.add_argument(
        "--meta_test_episodes", type=int, default=1, help="how many meta-test episodes"
    )
    parser.add_argument(
        "--shared-optimizer",
        default=True,
        metavar="SO",
        help="use an optimizer with shared statistics.",
    )
    parser.add_argument(
        "--load_model", type=str, default="", help="Path to load a saved model."
    )

    parser.add_argument(
        "--ep_save_freq",
        type=int,
        default=1e5,
        help="save model after this # of training episodes (default: 1e+4)",
    )
    parser.add_argument(
        "--optimizer",
        default="SharedAdam",
        metavar="OPT",
        help="shared optimizer choice of SharedAdam or SharedRMSprop",
    )
    parser.add_argument(
        "--save-model-dir",
        default="trained_models/",
        metavar="SMD",
        help="folder to save trained navigation",
    )
    parser.add_argument(
        "--log-dir", default="runs/", metavar="LG", help="folder to save logs"
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        default=-1,
        nargs="+",
        help="GPUs to use [-1 CPU only] (default: -1)",
    )
    parser.add_argument(
        "--amsgrad", default=True, metavar="AM", help="Adam optimizer amsgrad parameter"
    )
    parser.add_argument(
        "--grid_size",
        type=float,
        default=0.25,
        metavar="GS",
        help="The grid size used to discretize AI2-THOR maps.",
    )
    parser.add_argument(
        "--docker_enabled", action="store_true", help="Whether or not to use docker."
    )
    parser.add_argument(
        "--x_display", type=str, default=None, help="The X display to target, if any."
    )
    parser.add_argument(
        "--test_timeout",
        type=int,
        default=10,
        help="The length of time to wait in between test runs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="If true, output will contain more information.",
    )
    parser.add_argument(
        "--max_ep", type=int, default=6000000, help="maximum # of episodes"
    )

    parser.add_argument("--model",
                        type=str,
                        default="BaseModel",
                        help="Model to use.{BaseModel, GCN, SAVN, RelnetModel}")

    parser.add_argument(
        "--train_thin", type=int, default=1000, help="How often to print"
    )
    parser.add_argument(
        "--local_executable_path",
        type=str,
        default=None,
        help="a path to the local thor build.",
    )
    parser.add_argument(
        "--hindsight_replay",
        type=bool,
        default=False,
        help="whether or not to use hindsight replay.",
    )
    parser.add_argument(
        "--enable_test_agent",
        action="store_true",
        help="Whether or not to have a test agent.",
    )
    parser.add_argument(
        "--title", type=str, default="default_title", help="Info for logging."
    )

    parser.add_argument(
        "--train_scenes", type=str, default="[1-20]", help="scenes for training."
    )
    parser.add_argument(
        "--val_scenes",
        type=str,
        default="[21-30]",
        help="old validation scenes before formal split.",
    )

    parser.add_argument(
        "--possible_targets",
        type=str,
        default="FULL_OBJECT_CLASS_LIST",
        help="all possible objects.",
    )
    # if none use all dest objects
    parser.add_argument(
        "--train_targets",
        type=str,
        default=None,
        help="specific objects for this experiment from the object list.",
    )
    parser.add_argument(
        "--target_dim",
        type=int,
        default=300,
        help="which dimension of the glove vector to use",
    )
    parser.add_argument(
        "--action_space", type=int, default=7, help="space of possible actions, valid value: {4,6,7}."
    )

    parser.add_argument(
        "--hidden_state_sz", type=int, default=512, help="size of hidden state of LSTM."
    )

    parser.add_argument("--compute_spl", action="store_true", help="compute the spl.")

    parser.add_argument("--eval", action="store_true", help="run the test code")

    parser.add_argument(
        "--offline_data_dir",
        type=str,
        default="./data/thor_offline_data",
        help="where dataset is stored.",
        # for robothor: /home/chenjunting/ai2thor_data/Robothor_data
    )
    parser.add_argument(
        "--glove_dir",
        type=str,
        default="./data/thor_glove",
        help="where the glove files are stored.",
    )

    parser.add_argument(
        "--images_file_name",
        type=str,
        default="resnet18_featuremap.hdf5",
        help="Where the controller looks for images. Can be switched out to real images or Resnet features.",
    )

    parser.add_argument(
        "--disable-strict_done", dest="strict_done", action="store_false"
    )
    parser.set_defaults(strict_done=True)

    parser.add_argument(
        "--results_json", type=str, default="metrics.json", help="Write the results."
    )

    parser.add_argument(
        "--agent_type",
        type=str,
        default="NavigationAgent",
        help="Which type of agent. Choices are NavigationAgent or RandomAgent.",
    )

    parser.add_argument(
        "--episode_type",
        type=str,
        default="BasicEpisode",
        help="Which type of agent. Choices are NavigationAgent or RandomAgent.",
    )

    parser.add_argument(
        "--fov", type=float, default=100.0, help="The field of view to use."
    )

    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.25,
        help="The dropout ratio to use (default is no dropout).",
    )
    parser.add_argument(
        "--scene_types",
        nargs="+",
        default=["kitchen", "living_room", "bedroom", "bathroom"],
        # default for robothor
        #               ['FloorPlan_Train1', 'FloorPlan_Train2', 'FloorPlan_Train3',
        #                'FloorPlan_Train4', 'FloorPlan_Train5', 'FloorPlan_Train6',
        #                'FloorPlan_Train7', 'FloorPlan_Train8', 'FloorPlan_Train9',
        #                'FloorPlan_Train10', 'FloorPlan_Train11', 'FloorPlan_Train12']
    )

    parser.add_argument(
        "--gradient_limit",
        type=int,
        default=4,
        help="How many gradient steps allowed for MAML.",
    )

    parser.add_argument("--test_or_val", default="val", help="test or val")

################## new arguments for robothor data #################
    parser.add_argument("--data_source", type=str, default="robothor", help="selected from {ithor, robothor}" )

    # parser.add_argument("--graph_check", dest="graph_check", default=True, action="store_true",
    #                     help="whether use graph.json to check the validity of movement")
    parser.add_argument("--no_graph_check", dest="graph_check", action="store_false", default=True)
    # parser.set_defaults(graph_check=True)

    parser.add_argument("--rotate_by", type=int, default=30, help="rotation degree for RotateLeft/RotateRight, valid value:{30, 45}")

    parser.add_argument("--state_decimal", type=int, default=3, help="decimal of key in state data: e.g. images.hdf5")

    parser.add_argument("--load_checkpoint", type=str, default="", help="file_name of checkpoint to load")

    parser.add_argument("--ep_save_ckpt", type=int, default=100000, help="frequency to save checkpoint")

    parser.add_argument("--pinned_scene", dest="pinned_scene", action="store_true", default=False,
                        help="ONLY valied when data_source is robothor")
    # parser.set_defaults(pinned_scene=False)

    parser.add_argument("--curriculum_learning", dest="curriculum_learning", default=False, action="store_true")
    # parser.set_defaults(curriculum_learning=False)

    # parser.add_argument("--episode_file", type=str, default="./data/train.json", help="path to curriculum training episodes file")
    parser.add_argument("--curriculum_meta_dir", type=str, default="./data/curriculum_meta", help="directory to store curriculum meta files")
    parser.add_argument("--meta_pattern", type=str, default="curriculum_300000_1.0_0.8.json")

    parser.add_argument("--distance_upgrade_step",type=float,default=1.0)
    parser.add_argument("--penalty_decay",type=float,default=0.8)
    parser.add_argument(
        "--num_ep_per_stage",
        type=int,
        default=300000,
        help="number of episodes for each curriculum training stage"
    )

    parser.add_argument(
        "--proto_file",
        type=str,
        default="./data/object_protos_online.hdf5",
        help="file to store prototypes of target objects"
    )

################### new argumentes end #############################

    args = parser.parse_args()

    args.glove_file = "{}/glove_map{}d.hdf5".format(args.glove_dir, args.target_dim)

    return args
