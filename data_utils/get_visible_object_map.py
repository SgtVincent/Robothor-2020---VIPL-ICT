import os
import h5py
import json
import sys
sys.path.append('.')
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="scrape all possible images from ai2thor scene")

    parser.add_argument("--data_dir", type=str, required=True)

    parser.add_argument("--scenes", type=str, default="", help="scenes separated by ',': Train1,Train2,... ")

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    if not args.scenes:
        scenes = os.listdir(args.data_dir)
    else:
        scenes = args.scenes.split(',')

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