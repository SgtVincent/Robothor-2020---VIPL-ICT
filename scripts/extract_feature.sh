#!/usr/bin/env bash

func(){
  python data_utils/feature_extractioin.py \
    --scene_dir="/home/chenjunting/Robothor_data" \
    --scenes custom_robothor -i $1 \
    --num_process 5 \
    --model resnet18
}

CUDA_VISIBLE_DEVICES=1 func 1&
#CUDA_VISIBLE_DEVICES=1 func 2&
CUDA_VISIBLE_DEVICES=2 func 3&
#func 4&
#func 5&
#func 6&
#func 7&
#func 8&
#func 9&
#func 10&
#func 11&
#func 12&

