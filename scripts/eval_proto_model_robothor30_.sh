#!/usr/bin/env bash
python main.py \
  --eval \
  --test_or_val test \
  --episode_type RobothorTestValEpisode \
  --load_model trained_models/trained_proto_sp1e-3_decay08_0504/proto_train_sp1e-3_decay08_89023897_6000000_2020-04-30_08:43:23.dat \
  --model ProtoModel \
  --scene_types FloorPlan_Val1 FloorPlan_Val2 FloorPlan_Val3 \
  --offline_data_dir /home/ubuntu/Robothor_data_val \
  --curriculum_meta_dir ./data/valmeta_1 \
  --offline_shortest_data shortest_path_len.json \
  --images_file_name resnet18.hdf5 \
  --proto_file ./data/object_protos_thor.hdf5 \
  --gpu-ids 0 \
  --seed 10 \
  --max_val_ep 10000000 \
  --max_ep_per_diff 10000 \
  --action_space 7 \
  --curriculum_learning \
  --pinned_scene \
  --results_json results/proto_test.json

cat ./results/proto_test.json