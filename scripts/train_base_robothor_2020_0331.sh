#!/usr/bin/env bash
python main.py \
  --title base_train \
  --model BaseModel \
  --offline_data_dir /data/robothor/Robothor_data \
  --gpu-ids  1 2 \
  --workers 12 \
  --save-model-dir trained_base_2020_0331 \
  --data_source robothor \
  --grid_size 0.125 --rotate_by 30 --state_decimal 3 \
  --images_file_name resnet18.hdf5 \
  --train_scenes [1-5] \
  --seed 10 \
  --scene_types FloorPlan_Train1 FloorPlan_Train2 FloorPlan_Train3


