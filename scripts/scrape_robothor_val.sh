#!/usr/bin/env bash
python data_utils/scrape_scene_image_Robothor.py \
  --num_process 15 \
  --out_dir /home/ubuntu/Robothor_data_val \
  --scenes "FloorPlan_Val1_1,FloorPlan_Val1_2,FloorPlan_Val1_3,FloorPlan_Val1_4,FloorPlan_Val1_5,FloorPlan_Val2_1,FloorPlan_Val2_2,FloorPlan_Val2_3,FloorPlan_Val2_4,FloorPlan_Val2_5,FloorPlan_Val3_1,FloorPlan_Val3_2,FloorPlan_Val3_3,FloorPlan_Val3_4,FloorPlan_Val3_5" \
  | tee run_log/scrape_robothor_val_0502.log