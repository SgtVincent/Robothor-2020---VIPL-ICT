

for scene in scenes:
    try:
        with open('{}/{}/metadata.json'.format(data_dir, scene), 'r') as f:
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

        with open('{}/{}/visible_object_map.json'.format(data_dir, scene), 'w') as f:
            json.dump(visible_map, f)
    except Exception as e:
        print(scene, e)