#!/usr/bin/env bash
set -euo pipefail
# This script is an example usage of `convert_to_mcap.py` to convert the nuScenes mini-v1.0 dataset to MCAP.

docker build -t uto_nuscenes2mcap_tools .
mkdir -p output
mkdir -p data

