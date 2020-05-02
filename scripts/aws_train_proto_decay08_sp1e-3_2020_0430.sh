#!/usr/bin/env bash
python main.py \
  --title proto_train_sp1e-3_decay08 \
  --model ProtoModel \
  --offline_data_dir /home/ubuntu/Robothor_data \
  --gpu-ids  0 1 2 3 \
  --workers 12 \
  --save-model-dir trained_proto_sp1e-3_decay08 \
  --data_source robothor \
  --grid_size 0.125 --rotate_by 30 --state_decimal 3 \
  --images_file_name resnet18.hdf5 \
  --train_scenes [1-5] \
  --seed 10 \
  --lr 0.0001 \
  --step_penalty -0.001 \
  --max-episode-length 100 \
  --max_ep 6000000 \
  --num_ep_per_stage 600000 \
  --curriculum_learning \
  --pinned_scene \
  --scene_types FloorPlan_Train1 FloorPlan_Train2 FloorPlan_Train3 FloorPlan_Train4 FloorPlan_Train5 FloorPlan_Train6 FloorPlan_Train7 FloorPlan_Train8 FloorPlan_Train8 FloorPlan_Train10 FloorPlan_Train11 FloorPlan_Train12\
  --meta_pattern "curriculum_300000_1.0_0.8.json"\
  --penalty_decay 0.8 \
  --proto_file ./data/object_protos_thor.hdf5
  # --verbose \

