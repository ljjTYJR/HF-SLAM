#!/bin/bash

SCENE=$1
export SCENE

cd /home/airlab/nkeetha/4d/4DTrack

echo "Running scene number ${SCENE} with seed 0"
python3 -u scripts/slam.py configs/scannetpp/slam.py