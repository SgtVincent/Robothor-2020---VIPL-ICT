#!/usr/bin/env bash
python main.py \
  --title train_base_1e-3 \
  --model BaseModel \
  --offline_data_dir /data/robothor/Robothor_data \
  --gpu-ids  0 1 2 \
  --workers 12 \
  --lr 1e-3 \
  --save-model-dir trained_base_1e-3 \
  --data_source robothor \
  --grid_size 0.125 --rotate_by 30 --state_decimal 3 \
  --images_file_name resnet18.hdf5 \
  --train_scenes [1-5] \
  --seed 1 \
  --load_model pretrained_models/nonadaptivea3c_pretrained.dat \
  --pinned_scene \
  --scene_types FloorPlan_Train1 FloorPlan_Train2 FloorPlan_Train3

  #  --load_checkpoint checkpoint.dat \

