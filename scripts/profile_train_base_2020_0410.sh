#!/usr/bin/env bash
python cprofile_main.py \
  --title cprofile_train \
  --model BaseModel \
  --offline_data_dir /home/ubuntu/Robothor_data \
  --gpu-ids  0 \
  --workers 4 \
  --save-model-dir temp_cprofile_train \
  --data_source robothor \
  --grid_size 0.125 --rotate_by 30 --state_decimal 3 \
  --images_file_name resnet18.hdf5 \
  --train_scenes [1-5] \
  --seed 10 \
  --lr 0.0001 \
  --step_penalty -0.005 \
  --max-episode-length 50 \
  --max_ep 1000 \
  --curriculum_learning \
  --pinned_scene \
  --scene_types FloorPlan_Train1 FloorPlan_Train2 FloorPlan_Train3 FloorPlan_Train4 FloorPlan_Train5 FloorPlan_Train6 FloorPlan_Train7 FloorPlan_Train8 FloorPlan_Train8 FloorPlan_Train10 FloorPlan_Train11 FloorPlan_Train12

  # --verbose \

