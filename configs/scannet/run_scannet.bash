#!/bin/bash

cd /storage2/datasets/jkarhade/gaussian_slam/4DTrack

for seed in 0 1 2
do
    SEED=${seed}
    export SEED
    echo "Running scene number ${SCENE_NUM} with seed ${SEED}"
    python3 -u scripts/slam.py configs/scannet/scannet_eval.py
done