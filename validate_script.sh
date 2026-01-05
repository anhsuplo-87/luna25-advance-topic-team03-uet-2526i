#!/bin/bash
# baseline
python ./inference_aux_film.py --model_name LUNA25-aux-film-baseline_mask_mclab_split-multitask-3D-20251217 --valid
python ./inference_aux_film.py --model_name LUNA25-aux-film-baseline-with-aux_mask_mclab_split-multitask-3D-20251217 --valid
python ./inference_aux_film.py --model_name LUNA25-aux-film-baseline-with-aux-seg-gate_mask_mclab_split-multitask-3D-20251217 --valid
python ./inference_aux_film.py --model_name LUNA25-aux-film-baseline-with-aux-seg-clinical-gate_mask_mclab_split-multitask-3D-20251217 --valid
python ./inference_aux_film.py --model_name LUNA25-aux-film-baseline-with-aux-clinical-gate_mask_mclab_split-multitask-3D-20251218 --valid
python ./inference_aux_film.py --model_name LUNA25-aux-film-baseline-clinical-gate_mask_mclab_split-multitask-3D-20251218 --valid

# avg-nonlinear
python ./inference_aux_film.py --model_name LUNA25-aux-film-avg-nonlinear_mask_mclab_split-multitask-3D-20251220 --valid
python ./inference_aux_film.py --model_name LUNA25-aux-film-avg-nonlinear-with-aux_mask_mclab_split-multitask-3D-20251220 --valid
python ./inference_aux_film.py --model_name LUNA25-aux-film-avg-nonlinear-with-aux-seg-gate_mask_mclab_split-multitask-3D-20251221 --valid
python ./inference_aux_film.py --model_name LUNA25-aux-film-avg-nonlinear-with-aux-seg-clinical-gate_mask_mclab_split-multitask-3D-20251221 --valid
python ./inference_aux_film.py --model_name LUNA25-aux-film-avg-nonlinear-with-aux-clinical-gate_mask_mclab_split-multitask-3D-20251221 --valid
python ./inference_aux_film.py --model_name LUNA25-aux-film-avg-nonlinear-clinical-gate_mask_mclab_split-multitask-3D-20251221 --valid

# max head
python ./inference_aux_film.py --model_name LUNA25-aux-film-max-linear_mask_mclab_split-multitask-3D-20251220 --valid
python ./inference_aux_film.py --model_name LUNA25-aux-film-max-nonlinear_mask_mclab_split-multitask-3D-20251220 --valid

# atten head
python ./inference_aux_film.py --model_name LUNA25-aux-film-atten-linear_mask_mclab_split-multitask-3D-20251220 --valid
python ./inference_aux_film.py --model_name LUNA25-aux-film-atten-nonlinear_mask_mclab_split-multitask-3D-20251220 --valid
python ./inference_aux_film.py --model_name LUNA25-aux-film-atten-nonlinear_mask_mclab_split-multitask-3D-20251221 --valid

# multi head
python ./inference_aux_film.py --model_name LUNA25-aux-film-multi-linear_mask_mclab_split-multitask-3D-20251220 --valid
python ./inference_aux_film.py --model_name LUNA25-aux-film-multi-nonlinear_mask_mclab_split-multitask-3D-20251220 --valid