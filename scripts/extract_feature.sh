#!/usr/bin/env bash

func(){
  python utils/feature_extractioin.py \
    --scene_dir="/home/chenjunting/ai2thor_data/Robothor_data" \
    --scenes custom_robothor -i $1 \
    --num_process 1 \
    --model resnet18
}

func 1&
func 2&
func 3&
func 4&
func 5&
func 6&
func 7&
func 8&
func 9&
func 10&
func 11&
func 12&

