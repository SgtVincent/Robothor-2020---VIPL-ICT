#!/usr/bin/env bash


python data_utils/extract_scene_features.py \
  --scene_dir="/home/ubuntu/Robothor_data" \
  --scenes "FloorPlan_Train1_1,FloorPlan_Train1_2" \
  --num_process 5 \
  --model resnet18




