import os
import h5py
import json
import sys
sys.path.append('.')
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="scrape all possible images from ai2thor scene")

    parser.add_argument("--data_dir", type=str, required=True)

    parser.add_argument("--scenes",
                        type=str,
                        nargs="+",
                        default=["FloorPlan_Train{}_{}".format(i,j) for i in range(1,13) for j in range(1,6)])

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    scenes = args.scenes

    for scene in scenes:
        try:
            with open('{}/{}/metadata.json'.format(args.data_dir, scene), 'r') as f:
                metadata_list = json.load(f)

            visible_map = {}
            for k in metadata_list:
                metadata = metadata_list[k]
                for obj in metadata['objects']:
                    if obj['visible']:
                        objId = obj['objectId']
                        if objId not in visible_map:
                            visible_map[objId] = []
                        visible_map[objId].append(k)

            with open('{}/{}/visible_object_map.json'.format(args.data_dir, scene), 'w') as f:
                json.dump(visible_map, f)
            print("Finished visible_object_map.json for scene {}".format(scene))
        except Exception as e:
            print(scene, e)