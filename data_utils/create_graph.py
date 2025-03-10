from concurrent import futures
import json
import copy
import os
import h5py
import networkx as nx
from networkx.readwrite import json_graph
import sys
sys.path.append(".") # Assume script run in project root directory
from datasets.offline_sscontroller import ThorAgentState
import argparse
from concurrent import futures


STATE_DECIMAL=3 # Robothor

def convert_key_to_state(state_key):
    def _convert_list_to_state(l):
        return float(l[0]), float(l[1]), int(l[2]), int(l[3])
    try:
        x, z, r, h = _convert_list_to_state(state_key.split('|'))
        return ThorAgentState(x, 0.0, z, r, h, STATE_DECIMAL)

    except:
        raise Exception("state_key '{}' not valid".format(state_key))

class GraphMaker:
    def __init__(self,
                 image_dir,
                 image_file="images.hdf5",
                 graph_dir=None,
                 graph_file="graph.json"):

        self.image_dir = image_dir
        self.image_file = image_file
        if graph_dir == None:
            self.graph_dir = image_dir
        else:
            self.graph_dir = graph_dir
        self.graph_file = graph_file

        images = h5py.File(
            os.path.join(self.image_dir, self.image_file), "r"
        )
        self.all_states = list(images.keys())
        images.close()

        self.graph = nx.DiGraph()


    def start(self):
        # traverse all sub connected graphs
        for state_key in self.all_states:
            self.step(state_key)


        with open(os.path.join(self.graph_dir, self.graph_file), "w") as f:
            data = json_graph.node_link_data(self.graph)
            json.dump(data, f)

    def step(self, state):
        raise  NotImplementedError()

class GraphMakerDegree30(GraphMaker):

    def __init__(
            self,
            image_dir,
            image_file="images.hdf5",
            graph_dir=None,
            graph_file="graph.json",
            grid_size=0.125,
            actions=["MoveAhead","MoveBack", "RotateLeft", "RotateRight", "LookUp", "LookDown"]
    ):
        super(GraphMakerDegree30, self).__init__(
            image_dir, image_file, graph_dir, graph_file)
        self.actions = actions
        self.grid_size = grid_size

    def _get_next_state_30(self, state, action, copy_state=False):
        if copy_state:
            next_state = copy.deepcopy(state)
        else:
            next_state = state
        if action == "MoveAhead":
            if next_state.rotation == 0:
                next_state.z += 2 * self.grid_size
            elif next_state.rotation == 30:
                next_state.z += 2 * self.grid_size
                next_state.x += self.grid_size
            elif next_state.rotation == 60:
                next_state.z += self.grid_size
                next_state.x += 2 * self.grid_size
            elif next_state.rotation == 90:
                next_state.x += 2 * self.grid_size
            elif next_state.rotation == 120:
                next_state.z -= self.grid_size
                next_state.x += 2 * self.grid_size
            elif next_state.rotation == 150:
                next_state.z -= 2 * self.grid_size
                next_state.x += self.grid_size
            elif next_state.rotation == 180:
                next_state.z -= 2 * self.grid_size
            elif next_state.rotation == 210:
                next_state.z -= 2 * self.grid_size
                next_state.x -= self.grid_size
            elif next_state.rotation == 240:
                next_state.z -= self.grid_size
                next_state.x -= 2 * self.grid_size
            elif next_state.rotation == 270:
                next_state.x -= 2 * self.grid_size
            elif next_state.rotation == 300:
                next_state.z += self.grid_size
                next_state.x -= 2 * self.grid_size
            elif next_state.rotation == 330:
                next_state.z += 2 * self.grid_size
                next_state.x -= self.grid_size
            else:
                raise Exception("Unknown Rotation")
        if action == "MoveBack":
            if next_state.rotation == 0:
                next_state.z -= 2 * self.grid_size
            elif next_state.rotation == 30:
                next_state.z -= 2 * self.grid_size
                next_state.x -= self.grid_size
            elif next_state.rotation == 60:
                next_state.z -= self.grid_size
                next_state.x -= 2 * self.grid_size
            elif next_state.rotation == 90:
                next_state.x -= 2 * self.grid_size
            elif next_state.rotation == 120:
                next_state.z += self.grid_size
                next_state.x -= 2 * self.grid_size
            elif next_state.rotation == 150:
                next_state.z += 2 * self.grid_size
                next_state.x -= self.grid_size
            elif next_state.rotation == 180:
                next_state.z += 2 * self.grid_size
            elif next_state.rotation == 210:
                next_state.z += 2 * self.grid_size
                next_state.x += self.grid_size
            elif next_state.rotation == 240:
                next_state.z += self.grid_size
                next_state.x += 2 * self.grid_size
            elif next_state.rotation == 270:
                next_state.x += 2 * self.grid_size
            elif next_state.rotation == 300:
                next_state.z -= self.grid_size
                next_state.x += 2 * self.grid_size
            elif next_state.rotation == 330:
                next_state.z -= 2 * self.grid_size
                next_state.x += self.grid_size
            else:
                raise Exception("Unknown Rotation")

        elif action == "RotateRight":
            next_state.rotation = (next_state.rotation + 30) % 360
        elif action == "RotateLeft":
            next_state.rotation = (next_state.rotation - 30) % 360
        elif action == "LookUp":
            if next_state.horizon <= -30:
                return None
            next_state.horizon = next_state.horizon - 30
        elif action == "LookDown":
            if next_state.horizon >= 30:
                return None
            next_state.horizon = next_state.horizon + 30
        return next_state


    def step(self, state_key):

        state = convert_key_to_state(state_key)
        for action in self.actions:

            next_state = self._get_next_state_30(state, action, copy_state=True)
            next_state_key = str(next_state)

            if next_state_key in self.all_states:
                self.graph.add_edge(state_key, next_state_key)

def build_graph_for_scene(
    scene,
    dir="/home/chenjunting/ai2thor_data/Robothor_data",
    rotation=30
    ):

    if rotation == 30:
        graph_maker = GraphMakerDegree30(os.path.join(dir, scene))
        graph_maker.start()
        return scene
    else:
        print("Error: given rotation {}, supported rotation: [30]".format(rotation))
        return None


def parse_arguments():
    parser = argparse.ArgumentParser(description="scrape all possible images from ai2thor scene")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="+",
        default=["FloorPlan_Train{}_{}".format(i,j) for i in range(1,13) for j in range(1,6)]
    )
    parser.add_argument(
        "--num_process",
        type=int,
        default=15
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # make graph data for 30 degree case with parallel processes


    args = parse_arguments()
    data_dir = args.data_dir
    scenes = args.scenes
    num_process = args.num_process

    executor = futures.ProcessPoolExecutor(max_workers=args.num_process)
    fs = []

    for scene in scenes:
        try:
            build_graph_for_scene(scene, data_dir)
            print("scene {} has finished".format(scene))
        except Exception as e:
            print(e)
            print("scene {} has FAILED!".format(scene))
    with futures.ProcessPoolExecutor(max_workers=num_process) as executor:

        fs = [executor.submit(build_graph_for_scene, scene, data_dir)
              for scene in scenes]

    for future in futures.as_completed(fs):
        scene = future.result()
        print("scene {} has finished".format(scene))
