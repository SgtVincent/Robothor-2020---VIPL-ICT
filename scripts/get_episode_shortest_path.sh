#!/usr/bin/env bash

python data_utils/get_episode_shortest_path.py \
  --offline_data_dir /home/ubuntu/Robothor_data_val \
  --curriculum_meta_dir ./data/valmeta_1 \
  --images_file_name resnet18.hdf5 \
  --model BaseModel \
  --episode_type BasicEpisode \
  --gpu-ids 0 \
  --workers 8 \
  --seed 10 \
  --load_model pretrained_models/nonadaptivea3c_pretrained.dat \
  --scene_types FloorPlan_Val1 FloorPlan_Val2 FloorPlan_Val3 \
  --train_scenes [1-5]