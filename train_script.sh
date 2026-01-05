#!/bin/bash
# Experiment Name
python experiment_config.py \
    --EXPERIMENT_NAME LUNA25-aux-film-atten-nonlinear-with-aux \
    --force

# Set new config
python experiment_config.py \
    --set USE_AUX_MODEL=true \
    --set USE_SEG_GATE=false \
    --set USE_CLINICAL_GATE=false \
    --set CLS_HEAD_TYPE=atten_head \
    --set CLS_TAIL_TYPE=nonlinear \
    --force

# Run Training
python train_aux_film.py

# ---------- ---------- ---------- ---------- ---------- #

# Experiment Name
python experiment_config.py \
    --EXPERIMENT_NAME LUNA25-aux-film-atten-nonlinear-with-aux-seg-gate \
    --force

# Set new config
python experiment_config.py \
    --set USE_AUX_MODEL=true \
    --set USE_SEG_GATE=true \
    --set USE_CLINICAL_GATE=false \
    --set CLS_HEAD_TYPE=atten_head \
    --set CLS_TAIL_TYPE=nonlinear \
    --force

# Run Training
python train_aux_film.py

# ---------- ---------- ---------- ---------- ---------- #

# Experiment Name
python experiment_config.py \
    --EXPERIMENT_NAME LUNA25-aux-film-atten-nonlinear-with-aux-seg-clinical-gate \
    --force

# Set new config
python experiment_config.py \
    --set USE_AUX_MODEL=true \
    --set USE_SEG_GATE=true \
    --set USE_CLINICAL_GATE=true \
    --set CLS_HEAD_TYPE=atten_head \
    --set CLS_TAIL_TYPE=nonlinear \
    --force

# Run Training
python train_aux_film.py

# ---------- ---------- ---------- ---------- ---------- #

# Experiment Name
python experiment_config.py \
    --EXPERIMENT_NAME LUNA25-aux-film-atten-nonlinear-with-aux-clinical-gate \
    --force

# Set new config
python experiment_config.py \
    --set USE_AUX_MODEL=true \
    --set USE_SEG_GATE=false \
    --set USE_CLINICAL_GATE=true \
    --set CLS_HEAD_TYPE=atten_head \
    --set CLS_TAIL_TYPE=nonlinear \
    --force

# Run Training
python train_aux_film.py

# ---------- ---------- ---------- ---------- ---------- #

# Experiment Name
python experiment_config.py \
    --EXPERIMENT_NAME LUNA25-aux-film-atten-nonlinear-clinical-gate \
    --force

# Set new config
python experiment_config.py \
    --set USE_AUX_MODEL=false \
    --set USE_SEG_GATE=false \
    --set USE_CLINICAL_GATE=true \
    --set CLS_HEAD_TYPE=atten_head \
    --set CLS_TAIL_TYPE=nonlinear \
    --force

# Run Training
python train_aux_film.py