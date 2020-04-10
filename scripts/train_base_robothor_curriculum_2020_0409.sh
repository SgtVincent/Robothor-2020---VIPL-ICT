#!/usr/bin/env bash
python main.py \
  --title base_train \
  --model BaseModel \
  --offline_data_dir /home/chenjunting/Robothor_data \
  --gpu-ids  0 1 2 3 \
  --workers 18 \
  --save-model-dir trained_base_2020_0409 \
  --data_source robothor \
  --grid_size 0.125 --rotate_by 30 --state_decimal 3 \
  --images_file_name resnet18.hdf5 \
  --train_scenes [1-5] \
  --seed 10 \
  --lr 0.0001 \
  --step_penalty -0.005 \
  --max-episode-length 50 \
  --max_ep 3000000 \
  --verbose \
  --curriculum_learning \
  --pinned_scene \
  --scene_types FloorPlan_Train1 FloorPlan_Train2 FloorPlan_Train3


