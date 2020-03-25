#!/usr/bin/env bash
python main.py \
    --title base_train \
    --model BaseModel \
    --gpu-ids 3 \
    --workers 1 \
    --offline_data_dir /home/chenjunting/ai2thor_data/Robothor_data \
    --val_scenes [1-5]
    --save-model-dir trained_base_robothor_2020_0323