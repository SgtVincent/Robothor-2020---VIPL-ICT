#!/usr/bin/env bash
python cprofile_main.py \
  --eval \
  --test_or_val test \
  --title cprofile_val \
  --episode_type RobothorTestValEpisode \
  --model BaseModel \
  --offline_data_dir /home/ubuntu/Robothor_data_val \
  --gpu-ids  0 \
  --save-model-dir temp_cprofile_train \
  --data_source robothor \
  --grid_size 0.125 --rotate_by 30 --state_decimal 3 \
  --images_file_name resnet18.hdf5 \
  --train_scenes [1-5] \
  --seed 10 \
  --lr 0.0001 \
  --step_penalty -0.001 \
  --max-episode-length 50 \
  --max_val_ep 30 \
  --max_ep_per_diff 1 \
  --curriculum_learning \
  --pinned_scene \
  --curriculum_meta_dir ./data/valmeta_1 \
  --load_model trained_sp1e-3_decay08_1m/base_train_sp1e-3_decay08_133207532_10000000_2020-04-28_10_58_43.dat \
  --scene_types FloorPlan_Val1 \
  --results_json base_model_test_robothor.json






