""" Contains the Episodes for Navigation. """
from datasets.environment import Environment
from datasets.thor_agent_state import ThorAgentState
from utils.net_util import gpuify, toFloatTensor
from .basic_episode import BasicEpisode


class RobothorTestValEpisode(BasicEpisode):
    """ Episode for Navigation. """

    def __init__(self, args, gpu_id, strict_done=False):
        super(RobothorTestValEpisode, self).__init__(args, gpu_id, strict_done)

        self.curriculum_meta_iter = None
        self.episode_count = 0

    def _new_episode(self,
                     args,
                     episode,
                     glove=None,
                     protos=None,
                     pre_metadata=None):
        """ New navigation episode. """
        if episode == None:
            return None
        scene = episode["scene"]

        if self._env is None:
            self._env = Environment(
                offline_data_dir=args.offline_data_dir,
                use_offline_controller=True,
                grid_size=args.grid_size,
                images_file_name=args.images_file_name,
                local_executable_path=args.local_executable_path,
                rotate_by=args.rotate_by,
                state_decimal=args.state_decimal,
                pinned_scene=args.pinned_scene,
                pre_metadata=pre_metadata,
                actions=self.actions
            )
            self._env.start(scene)
        else:
            self._env.reset(scene)

        self.environment.controller.state = ThorAgentState(**episode['initial_position'],
                                                           rotation=episode['initial_orientation'],
                                                           horizon=0, state_decimal=args.state_decimal)

        self.task_data = [episode["object_id"]]
        object_type = episode["object_type"]
        self.target_object = object_type
        self.difficulty = episode["difficulty"]
        self.episode_id = episode['id']

        if args.glove_file != "":
            self.glove_embedding = toFloatTensor(glove.glove_embeddings[object_type][:], self.gpu_id)
        if args.proto_file != "":
            self.prototype = toFloatTensor(protos.protos[object_type.lower()][:], self.gpu_id)

        if args.verbose:
            print("Scene", scene, "Navigating towards:", self.target_object)

        return scene

    def _init_curriculum_meta_iter(
            self, scenes, curriculum_meta, max_ep_per_target=10):
        for scene in scenes:
            for diff_episodes in curriculum_meta[scene]:
                for ep in diff_episodes[:max_ep_per_target]:
                    yield ep

    def new_episode(
            self,
            args,
            scenes,
            possible_targets=None,
            targets=None,
            keep_obj=False,
            glove=None,
            protos=None,
            pre_metadata=None,
            curriculum_meta=None,
            total_ep=0,
            max_episode_per_obj=50,  # not implemented yet
    ):
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self.prev_frame = None
        self.current_frame = None

        targets = [target for target in targets if target in possible_targets]

        if args.curriculum_learning ==False:
            raise Exception("Error: Using robothor dataset, only curriculum_learing mode supported yet!")

        if self.curriculum_meta_iter == None:
            self.curriculum_meta_iter = self._init_curriculum_meta_iter(scenes, curriculum_meta, args.max_ep_per_diff)

        episode = None
        while True:  # iterate curriculum_meta until a valid episode found
            try:
                episode = next(self.curriculum_meta_iter)
            except StopIteration as e: # Iteration stopped
                episode = None
                break
            object_type = episode['object_type'].replace(" ","")
            if object_type in targets:
                self.episode_count += 1
                break

        return self._new_episode(args,
                                 episode=episode,
                                 glove=glove,
                                 protos=protos,
                                 pre_metadata=pre_metadata)
# unit test
# if __name__ == '__main__':
#     import sys
#     sys.path.append("./")
#     from utils import flag_parser
#     args = flag_parser.parse_arguments()
#     ep_obj = TestValEpisode(args, args.gpus[0])
#     ep_obj.new_episode()