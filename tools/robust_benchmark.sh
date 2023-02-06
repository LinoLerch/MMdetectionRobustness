#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
OUT=$3
#CORR=${CORR:-'fog'}
#SEV=${SEV:-2 3}

#CORR=${CORR:-'holdout'}
#CORR=${CORR:-'benchmark'}
#CORR=${CORR:-'shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression'}

# FF-baseline  fb_add
#CORR=${CORR:-'motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression'}

# fb_mod
CORR=${CORR:-'snow frost fog brightness contrast elastic_transform pixelate jpeg_compression'}

SEV=${SEV:-0 1 2 3 4 5}

#pwd; ls -l  > dir_pwd.txt
#ls -l $(dirname "$0") > dir_0.txt
source /opt/bwhpc/common/devel/miniconda3/etc/profile.d/conda.sh
conda activate cvproj

python -Wignore tools/analysis_tools/test_robustness.py \
    $CONFIG \
    $CHECKPOINT \
    --out=$OUT \
    --corruptions $CORR  \
    --eval bbox \
    --summaries True \
    --severities $SEV \
    --final-prints P mPC rPC
    
#    --launcher slurm