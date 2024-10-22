#!/usr/bin/env bash
python main.py \
  --eval \
  --test_or_val test \
  --episode_type RobothorTestValEpisode \
  --load_model trained_sp1e-3_decay08_1m/base_train_sp1e-3_decay08_133207532_10000000_2020-04-28_10_58_43.dat \
  --model BaseModel \
  --scene_types FloorPlan_Val1 FloorPlan_Val2 FloorPlan_Val3 \
  --offline_data_dir /home/ubuntu/Robothor_data_val \
  --curriculum_meta_dir ./data/valmeta_1 \
  --offline_shortest_data "shortest_path_len.json" \
  --images_file_name resnet18.hdf5 \
  --gpu-ids 0 \
  --seed 10 \
  --max_val_ep 10000000 \
  --max_ep_per_diff 10000 \
  --action_space 7 \
  --curriculum_learning \
  --pinned_scene \
  --results_json results/base_test.json

cat ./results/base_test.json