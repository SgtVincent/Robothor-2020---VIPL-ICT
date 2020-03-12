""" Exhaustive BFSController to search for reachable scene """

from collections import deque
import json
import copy
import time
import random
import os

from ai2thor.controller import Controller, distance
from .base_controller import BaseController

class ThorAgentState:
    """ Representation of a simple state of a Thor Agent which includes
        the position, horizon and rotation. """

    def __init__(self, x, y, z, rotation, horizon):
        self.x = round(x, 2)
        self.y = y
        self.z = round(z, 2)
        self.rotation = round(rotation)
        self.horizon = round(horizon)

    @classmethod
    def get_state_from_evenet(cls, event, forced_y=None):
        """ Extracts a state from an event. """
        state = cls(
            x=event.metadata["agent"]["position"]["x"],
            y=event.metadata["agent"]["position"]["y"],
            z=event.metadata["agent"]["position"]["z"],
            rotation=event.metadata["agent"]["rotation"]["y"],
            horizon=event.metadata["agent"]["cameraHorizon"],
        )
        if forced_y != None:
            state.y = forced_y
        return state

    def __eq__(self, other):
        """ If we check for exact equality then we get issues.
            For now we consider this 'close enough'. """
        if isinstance(other, ThorAgentState):
            return (
                self.x == other.x
                and
                # self.y == other.y and
                self.z == other.z
                and self.rotation == other.rotation
                and self.horizon == other.horizon
            )
        return NotImplemented

    def __str__(self):
        """ Get the string representation of a state. """
        """
        return '{:0.2f}|{:0.2f}|{:0.2f}|{:d}|{:d}'.format(
            self.x,
            self.y,
            self.z,
            round(self.rotation),
            round(self.horizon)
        )
        """
        return "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
            self.x, self.z, round(self.rotation), round(self.horizon)
        )

    def position(self):
        """ Returns just the position. """
        return dict(x=self.x, y=self.y, z=self.z)


class ExhaustiveBFSController(Controller):
    """ A much slower and more exhaustive version of the BFSController.
        This may be helpful if you wish to find the shortest path to an object.
        The usual BFSController does not consider things like rotate or look down
        when you are navigating towards an object. Additionally, there is some
        rare occurances of positions which you can only get to in a certain way.
        This ExhaustiveBFSController introduces the safe_teleport method which
        ensures that all states will be covered. 
        Strongly recomend having a seperate directory for each scene. See 
        OfflineControllerWithSmallRotation for more information on how the generated data may be used. """

    def __init__(
        self,
        grid_size=0.25,
        fov=90.0,
        grid_file=None,
        graph_file=None,
        metadata_file=None,
        images_file=None,
        seg_file=None,
        class_file=None,
        depth_file=None,
        debug_mode=True,
        grid_assumption=False,
        local_executable_path=None,
        actions=["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"],
        cameraY=0.2,
    ):

        super(ExhaustiveBFSController, self).__init__()
        # Allowed rotations.
        self.rotations = [0, 45, 90, 135, 180, 225, 270, 315]
        # Allowed horizons.
        self.horizons = [0, 30]

        self.allow_enqueue = True
        self.queue = deque()
        self.seen_points = []
        self.grid_points = []
        self.seen_states = []
        self.bad_seen_states = []
        self.visited_seen_states = []
        self.grid_states = []
        self.grid_size = grid_size
        self._check_visited = False
        self.scene_name = None
        self.fov = fov
        self.y = None
        self.cameraY = cameraY

        self.local_executable_path = local_executable_path

        # distance_threshold to be consistent with BFSController in generating grid.
        self.distance_threshold = self.grid_size / 5.0
        self.debug_mode = debug_mode
        self.actions = actions
        self.grid_assumption = grid_assumption

        self.grid_file = grid_file
        self.metadata_file = metadata_file
        self.graph_file = graph_file
        self.images_file = images_file
        self.seg_file = seg_file
        self.class_file = class_file
        self.depth_file = depth_file

        # Optionally make a gird (including x,y,z points that are reachable)
        self.make_grid = grid_file is not None

        # Optionally store the metadata of each state.
        self.make_metadata = metadata_file is not None

        # Optionally make a directed of (s,t) where exists a in self.actions
        # such that t is reachable via s via a.
        self.make_graph = graph_file is not None

        # Optionally store an hdf5 file which contains the frame for each state.
        self.make_images = images_file is not None

        self.make_seg = seg_file is not None
        self.make_class = class_file is not None

        self.make_depth = self.depth_file is not None

        self.metadata = {}
        self.classdata = {}

        self.graph = None
        if self.make_graph:
            import networkx as nx

            self.graph = nx.DiGraph()

        if self.make_images:
            import h5py

            self.images = h5py.File(self.images_file, "w")

        if self.make_seg:
            import h5py

            self.seg = h5py.File(self.seg_file, "w")

        if self.make_depth:
            import h5py

            self.depth = h5py.File(self.depth_file, "w")

    def safe_teleport(self, state):
        """ Approach a state from all possible directions if the usual teleport fails. """
        self.step(dict(action="Rotate", rotation=0))
        event = self.step(dict(action="Teleport", x=state.x, y=state.y, z=state.z))
        if event.metadata["lastActionSuccess"]:
            return event

        # Approach from the left.
        event = self.step(
            dict(action="Teleport", x=(state.x - self.grid_size), y=state.y, z=state.z)
        )
        if event.metadata["lastActionSuccess"]:
            self.step(dict(action="Rotate", rotation=90))
            event = self.step(dict(action="MoveAhead"))
            if event.metadata["lastActionSuccess"]:
                return event

        # Approach from the right.
        event = self.step(
            dict(action="Teleport", x=(state.x + self.grid_size), y=state.y, z=state.z)
        )
        if event.metadata["lastActionSuccess"]:
            self.step(dict(action="Rotate", rotation=270))
            event = self.step(dict(action="MoveAhead"))
            if event.metadata["lastActionSuccess"]:
                return event

        # Approach from the back.
        event = self.step(
            dict(action="Teleport", x=state.x, y=state.y, z=state.z - self.grid_size)
        )
        if event.metadata["lastActionSuccess"]:
            self.step(dict(action="Rotate", rotation=0))
            event = self.step(dict(action="MoveAhead"))
            if event.metadata["lastActionSuccess"]:
                return event

        # Approach from the front.
        event = self.step(
            dict(action="Teleport", x=state.x, y=state.y, z=state.z + self.grid_size)
        )
        if event.metadata["lastActionSuccess"]:
            self.step(dict(action="Rotate", rotation=180))
            event = self.step(dict(action="MoveAhead"))
            if event.metadata["lastActionSuccess"]:
                return event

        print(self.scene_name)
        print(str(state))
        raise Exception("Safe Teleport Failed")

    def teleport_to_state(self, state):
        """ Only use this method when we know the state is valid. """
        event = self.safe_teleport(state)
        assert event.metadata["lastActionSuccess"]
        event = self.step(dict(action="Rotate", rotation=state.rotation))
        assert event.metadata["lastActionSuccess"]
        event = self.step(dict(action="Look", horizon=state.horizon))
        assert event.metadata["lastActionSuccess"]

        if self.debug_mode:
            # Sanity check that we have teleported to the correct state.
            new_state = self.get_state_from_event(event)
            if state != new_state:
                print(state)
                print(new_state)
            assert state == new_state
        return event

    def get_state_from_event(self, event):
        return ThorAgentState.get_state_from_evenet(event, forced_y=self.y)

    def get_point_from_event(self, event):
        return event.metadata["agent"]["position"]

    def get_next_state(self, state, action, copy_state=False):
        """ Guess the next state when action is taken. Note that
            this will not predict the correct y value. """
        if copy_state:
            next_state = copy.deepcopy(state)
        else:
            next_state = state
        if action == "MoveAhead":
            if next_state.rotation == 0:
                next_state.z += self.grid_size
            elif next_state.rotation == 90:
                next_state.x += self.grid_size
            elif next_state.rotation == 180:
                next_state.z -= self.grid_size
            elif next_state.rotation == 270:
                next_state.x -= self.grid_size
            elif next_state.rotation == 45:
                next_state.z += self.grid_size
                next_state.x += self.grid_size
            elif next_state.rotation == 135:
                next_state.z -= self.grid_size
                next_state.x += self.grid_size
            elif next_state.rotation == 225:
                next_state.z -= self.grid_size
                next_state.x -= self.grid_size
            elif next_state.rotation == 315:
                next_state.z += self.grid_size
                next_state.x -= self.grid_size
            else:
                raise Exception("Unknown Rotation")
        elif action == "RotateRight":
            next_state.rotation = (next_state.rotation + 45) % 360
        elif action == "RotateLeft":
            next_state.rotation = (next_state.rotation - 45) % 360
        elif action == "LookUp":
            if abs(next_state.horizon) <= 1:
                return None
            next_state.horizon = next_state.horizon - 30
        elif action == "LookDown":
            if abs(next_state.horizon - 60) <= 1 or abs(next_state.horizon - 30) <= 1:
                return None
            next_state.horizon = next_state.horizon + 30
        return next_state

    def add_edge(self, curr_state, next_state):
        self.graph.add_edge(str(curr_state), str(next_state))

    def enqueue_state(self, state):
        """ Returns true if state is valid. """
        # ensure there are no dup states.
        if state in self.seen_states:
            return True

        if state in self.bad_seen_states:
            return False

        # ensure state is a legal rotation and horizon.
        if (
            round(state.horizon) not in self.horizons
            or round(state.rotation) not in self.rotations
        ):
            self.bad_seen_states.append(state)
            return False

        self.seen_states.append(state)
        self.queue.append(state)
        return True

    def enqueue_states(self, agent_state):

        if not self.allow_enqueue:
            return

        # Take all action in self.action and enqueue if they are valid.
        for action in self.actions:

            next_state_guess = self.get_next_state(agent_state, action, True)

            if next_state_guess is None:
                continue

            # # Bug.
            # if (
            #     self.scene_name == "FloorPlan208_physics"
            #     and next_state_guess.x == 0
            #     and next_state_guess.z == 1.75
            # ):
            #     self.teleport_to_state(agent_state)
            #     continue

            # Grid assumption is meant to make things faster and should not
            # be used in practice. In general it does not work when the y
            # values fluctuate in a scene. It circumvents using the actual controller.
            if self.grid_assumption:
                if next_state_guess in self.seen_states:
                    if self.make_graph:
                        self.add_edge(agent_state, next_state_guess)
                    continue

            event = self.step(
                dict(
                    action="Teleport",
                    x=next_state_guess.x,
                    y=next_state_guess.y,
                    z=next_state_guess.z,
                )
            )
            if not event.metadata["lastActionSuccess"]:
                self.teleport_to_state(agent_state)
                continue
            event = self.step(dict(action="Rotate", rotation=next_state_guess.rotation))
            if not event.metadata["lastActionSuccess"]:
                self.teleport_to_state(agent_state)
                continue
            event = self.step(dict(action="Look", horizon=next_state_guess.horizon))
            if not event.metadata["lastActionSuccess"]:
                self.teleport_to_state(agent_state)
                continue

            next_state = self.get_state_from_event(event)

            if next_state != next_state_guess:
                print(next_state)
                print(next_state_guess)
            assert next_state == next_state_guess

            if self.enqueue_state(next_state) and self.make_graph:
                self.add_edge(agent_state, next_state)

            # Return back to agents initial location.
            self.teleport_to_state(agent_state)

    def search_all_closed(self, scene_name):
        """ Runs the ExhaustiveBFSController on scene_name. """
        self.allow_enqueue = True
        self.queue = deque()
        self.seen_points = []
        self.visited_seen_points = []
        self.grid_points = []
        self.seen_states = []
        self.visited_seen_states = []
        self.scene_name = scene_name
        event = self.reset(scene_name)

        if self.make_seg or self.make_class:
            event = self.step(
                dict(
                    action="Initialize",
                    gridSize=self.grid_size,
                    fieldOfView=self.fov,
                    renderClassImage=True,
                    renderObjectImage=True,
                    renderDepthImage=True,
                    cameraY=self.cameraY,
                )
            )
        else:
            event = self.step(
                dict(
                    action="Initialize",
                    renderDepthImage=True,
                    gridSize=self.grid_size,
                    fieldOfView=self.fov,
                    cameraY=self.cameraY,
                )
            )
        self.y = event.metadata["agent"]["position"]["y"]

        # get all reachable positions
        event = self.step(dict(action="GetReachablePositions", gridSize=0.25))
        self.reachable_points = event.metadata["actionReturn"]

        self.enqueue_state(self.get_state_from_event(event))

        while self.queue:
            self.queue_step()

        if self.make_grid:
            with open(self.grid_file, "w") as outfile:
                json.dump(self.grid_points, outfile)
        if self.make_graph:
            from networkx.readwrite import json_graph

            with open(self.graph_file, "w") as outfile:
                data = json_graph.node_link_data(self.graph)
                json.dump(data, outfile)
        if self.make_metadata:
            with open(self.metadata_file, "w") as outfile:
                json.dump(self.metadata, outfile)
        if self.make_images:
            self.images.close()
        if self.make_seg:
            self.seg.close()
        if self.make_depth:
            self.depth.close()

        if self.make_class:
            with open(self.class_file, "w") as outfile:
                json.dump(self.classdata, outfile)

        print("Finished :", self.scene_name)

    def queue_step(self):
        search_state = self.queue.popleft()
        event = self.teleport_to_state(search_state)

        # if search_state.y > 1.3:
        #    raise Exception("**** got big point ")

        self.enqueue_states(search_state)
        self.visited_seen_states.append(search_state)

        if self.make_grid and not any(
            map(
                lambda p: distance(p, search_state.position())
                < self.distance_threshold,
                self.grid_points,
            )
        ):
            self.grid_points.append(search_state.position())

        if self.make_metadata:
            self.metadata[str(search_state)] = event.metadata

        if self.make_class:
            class_detections = event.class_detections2D
            for k, v in class_detections.items():
                class_detections[k] = str(v)
            self.classdata[str(search_state)] = class_detections

        if self.make_images and str(search_state) not in self.images:
            self.images.create_dataset(str(search_state), data=event.frame)

        if self.make_seg and str(search_state) not in self.seg:
            self.seg.create_dataset(
                str(search_state), data=event.class_segmentation_frame
            )

        if self.make_depth and str(search_state) not in self.depth:
            self.depth.create_dataset(str(search_state), data=event.depth_frame)

        elif str(search_state) in self.images:
            print(self.scene_name, str(search_state))
