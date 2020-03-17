from invoke.context import Context
from tasks import create_robothor_dataset
from tasks import benchmark
from tasks import shortest_path_to_object
from tasks import visualize_shortest_paths
from tasks import build_class_dataset


def test_create_robothor_dataset():

    # context,
    # local_build = False,
    # editor_mode = False,
    # width = 300,
    # height = 300,
    # output = 'robothor-dataset.json',
    # intermediate_directory = '.',
    # visibility_distance = 1.0,
    # objects_filter = None,
    # scene_filter = None,
    # filter_file = None
    create_robothor_dataset(Context())

if __name__ == "__main__":

    test_create_robothor_dataset()