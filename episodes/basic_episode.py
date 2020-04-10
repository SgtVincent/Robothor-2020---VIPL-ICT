""" Contains the Episodes for Navigation. """
import random
import os
import torch

from datasets.constants import DONE
from datasets.environment import Environment
from datasets.robothor_data import DIFFICULTY
from datasets.thor_agent_state import ThorAgentState

from utils.net_util import gpuify, toFloatTensor
from utils.action_util import get_actions
from utils.net_util import gpuify
from .episode import Episode


class BasicEpisode(Episode):
    """ Episode for Navigation. """

    def __init__(self, args, gpu_id, strict_done=False):
        super(BasicEpisode, self).__init__()

        self._env = None

        self.gpu_id = gpu_id
        self.strict_done = strict_done
        self.task_data = None
        self.glove_embedding = None
        self.actions = get_actions(args)
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self._last_action_embedding_idx = 0
        self.target_object = None
        self.prev_frame = None
        self.current_frame = None
        self.grid_size = args.grid_size
        self.goal_success_reward = args.goal_success_reward
        self.step_penalty = args.step_penalty


        self.scene_states = []
        if args.eval:
            random.seed(args.seed)

    @property
    def environment(self):
        return self._env

    @property
    def actions_list(self):
        return [{"action": a} for a in self.actions]

    def reset(self):
        self.done_count = 0
        self.duplicate_count = 0
        self._env.back_to_start()

    def state_for_agent(self):
        return self.environment.current_frame

    def current_agent_position(self):
        """ Get the current position of the agent in the scene. """
        return self.environment.current_agent_position

    def step(self, action_as_int):

        action = self.actions_list[action_as_int]

        if action["action"] != DONE:
            self.environment.step(action)
        else:
            self.done_count += 1

        reward, terminal, action_was_successful = self.judge(action)
        return reward, terminal, action_was_successful

    def judge(self, action):
        """ Judge the last event. """
        reward = self.step_penalty

        # Thresholding replaced with simple look up for efficiency.
        if self.environment.controller.state in self.scene_states:
            if action["action"] != DONE:
                if self.environment.last_action_success:
                    self.duplicate_count += 1
                else:
                    self.failed_action_count += 1
        else:
            self.scene_states.append(self.environment.controller.state)

        done = False

        if action["action"] == DONE:
            action_was_successful = False
            for id_ in self.task_data:
                if self.environment.object_is_visible(id_):
                    reward = self.goal_success_reward
                    done = True
                    action_was_successful = True
                    break
        else:
            action_was_successful = self.environment.last_action_success

        return reward, done, action_was_successful

    # Set the target index.
    @property
    def target_object_index(self):
        """ Return the index which corresponds to the target object. """
        return self._target_object_index

    @target_object_index.setter
    def target_object_index(self, target_object_index):
        """ Set the target object by specifying the index. """
        self._target_object_index = gpuify(
            torch.LongTensor([target_object_index]), self.gpu_id
        )

    def _new_random_episode(
        self, args, scenes, possible_targets, targets=None,
            keep_obj=False, glove=None, pre_metadata=None):
        """ New navigation episode. """
        #random episode
        scene = None
        retry = 0
        while scene not in os.listdir(args.offline_data_dir):
            scene = random.choice(scenes)
            retry += 1
            if retry >= 1000:
                raise Exception("No scenes found in {}".format(args.offline_data_dir))

        if self._env is None:
            self._env = Environment(
                offline_data_dir=args.offline_data_dir,
                use_offline_controller=True,
                grid_size=self.grid_size,
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

        # Randomize the start location.
        self._env.randomize_agent_location()
        objects = self._env.all_objects()

        visible_objects = [obj.split("|")[0] for obj in objects]
        intersection = [obj for obj in visible_objects if obj in targets]

        self.task_data = []

        idx = random.randint(0, len(intersection) - 1)
        goal_object_type = intersection[idx]
        self.target_object = goal_object_type

        for id_ in objects:
            type_ = id_.split("|")[0]
            if goal_object_type == type_:
                self.task_data.append(id_)

        if args.verbose:
            print("Scene", scene, "Navigating towards:", goal_object_type)

        self.glove_embedding = None
        self.glove_embedding = toFloatTensor(
            glove.glove_embeddings[goal_object_type][:], self.gpu_id
        )
        return scene

    # curriculum_meta: episodes indexed by scene, difficulty, object_type in order
    def _new_curriculum_episode(
        self, args, scenes, possible_targets, targets=None,
            keep_obj=False, glove=None, pre_metadata=None, curriculum_meta=None, total_ep=0):
        """ New navigation episode. """
        # choose a scene
        scene = None
        retry = 0

        flag_episode_valid = False
        while not flag_episode_valid:
            # choose a scene
            valid_scenes = os.listdir(args.offline_data_dir)
            intersection_scenes = [scene for scene in scenes if scene in valid_scenes]
            scene = random.choice(intersection_scenes)

            # choose difficulty
            if total_ep < args.difficulty_upgrade:
                diff = DIFFICULTY[0]
            elif total_ep < 2 * args.diffculty_upgrade:
                diff = random.choice(DIFFICULTY[:2])
            else:
                diff = random.choice(DIFFICULTY[:3])

            # choose object
            visible_objects = curriculum_meta[scene][diff].keys()
            intersection_objs = [obj for obj in visible_objects if obj in targets]
            object_type = random.choice(intersection_objs)

            episode = random.choice(curriculum_meta[scene][diff][object_type])
            # TODO: Present validity checking method breaks the principle of tiered-design and decoupling
            # TODO: Find a better way to check the validity of an episode  by junting, 2020-04-10

            state = ThorAgentState(**episode['initial_position'],rotation=episode['initial_orientation'],
                                   horizon=0, state_decimal=args.state_decimal)
            if str(state) in pre_metadata[scene]['all_states']:
                flag_episode_valid = True
            else:
                print("Episode ID {} not valid for its initial state missing from all_states".format(episode['id']) )


        if self._env is None:
            self._env = Environment(
                offline_data_dir=args.offline_data_dir,
                use_offline_controller=True,
                grid_size=self.grid_size,
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

        # initialize the start location.

        self._env.initialize_agent_location(**episode['initial_position'],
                                            rotation=episode['initial_orientation'],
                                            horizon=0)
        self.task_data = []
        self.target_object = object_type
        self.task_data.append(episode['object_id'])

        if args.verbose:
            print("Episode: Scene ", scene," Difficulty ",diff, " Navigating towards: ", object_type)

        self.glove_embedding = None
        self.glove_embedding = toFloatTensor(
            glove.glove_embeddings[object_type][:], self.gpu_id
        )
        return scene

    def new_episode(
        self,
        args,
        scenes,
        possible_targets=None,
        targets=None,
        keep_obj=False,
        glove=None,
        pre_metadata=None,
        curriculum_meta=None,
        total_ep=0,
    ):
        self.done_count = 0
        self.duplicate_count = 0
        self.failed_action_count = 0
        self.prev_frame = None
        self.current_frame = None

        if args.curriculum_learning:
            return self._new_curriculum_episode(args, scenes, possible_targets, targets,
                                keep_obj, glove, pre_metadata, curriculum_meta, total_ep)

        return self._new_random_episode(args, scenes, possible_targets, targets,
                                keep_obj, glove, pre_metadata)

