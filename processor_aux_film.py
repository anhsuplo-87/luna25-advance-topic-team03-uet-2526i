# ------------------- Processor ------------------- 

"""
Inference script for predicting malignancy of lung nodules
"""
import numpy as np
import pandas as pd
import dataloader
import dataloader_mask
import torch
import torch.nn as nn
from torchvision import models

import random

from models.mclab.model_2d import ResNet18
from models.mclab.model_3d import I3D
from models.multi_task_film_model import *

import os
import math
import logging
from pathlib import Path

from experiment_config import Configuration
from dataloader_mask import get_data_loader

import sklearn.metrics as metrics
from tqdm import tqdm

from train_visualizer import TrainVisualizer

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

# define processor
class MalignancyProcessor:
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule
    """

    def __init__(
            self, 
            suppress_logs=False, 
            model_name="LUNA25-baseline-2D",         # Experiment name
            model_dir="./results",
            device="cuda"
        ):

        self.model_dir = model_dir
        self.model_name = model_name
        self.suppress_logs = suppress_logs

        self.config = Configuration(Path(self.model_dir) / self.model_name / "config.json")

        self.size_px = self.config.SIZE_PX
        self.size_mm = self.config.SIZE_MM
        self.patch_size = self.config.PATCH_SIZE
        self.mode = self.config.MODE

        if not self.suppress_logs:
            logging.info("Initializing the deep learning system")

        self.device = torch.device(device)

        # Feature Extractor
        if self.mode == "2D":
            feature_extractor = ResNet18(weights=None).to(self.device)

        elif self.mode == "3D":
            feature_extractor = I3D(
                num_classes=1,
                input_channels=3,
                pre_trained=True, 
                dropout_prob=self.config.DROP_RATE, 
                freeze_bn=True, 
                extract_feature=True, 
            ).to(self.device)

        # Auxiliary Model
        aux_model = SegmentationHead(
            in_channels=1024, 
            out_channels=1, 
            target_size=(64,64,64)
        ).to(self.device)

        # Build main Model
        if 'baseline' in self.config.EXPERIMENT_NAME \
            and "-aux-clinical-gate" not in self.config.EXPERIMENT_NAME \
            and "-baseline-clinical-gate" not in self.config.EXPERIMENT_NAME:
            self.model = MultiTaskFiLMModel_baseline(
                feature_extractor, 
                aux_model=aux_model, 
                aux_task=self.config.AUX_TASK,
                use_aux_model=self.config.USE_AUX_MODEL,
                use_seg_gate=self.config.USE_SEG_GATE,
                use_clinical_gate=self.config.USE_CLINICAL_GATE,
                clinical_dim=self.config.CLINICAL_DIM
            ).to(self.device)
        else: # head - implementation [<head_type>-<tail_type>]
            self.model = MultiTaskFiLMModel(
                feature_extractor, 
                aux_model=aux_model, 
                aux_task=self.config.AUX_TASK,
                use_aux_model=self.config.USE_AUX_MODEL,
                use_seg_gate=self.config.USE_SEG_GATE,
                use_clinical_gate=self.config.USE_CLINICAL_GATE,
                clinical_dim=self.config.CLINICAL_DIM,
                cls_head_type=self.config.CLS_HEAD_TYPE,
                cls_tail_type=self.config.CLS_TAIL_TYPE
            ).to(self.device)

        self.model_root = Path(model_dir) / model_name 

    def _load_ckpt(self):
        model = self.model
        ckpt = torch.load(
            os.path.join(
                self.model_root,
                "best_multitask_model.pth",
            ),
            map_location=self.device
        )

        if 'ema_state_dict' in ckpt:
            model.load_state_dict(ckpt['ema_state_dict'])
        # If the standard 'model_state_dict' already contains the EMA weights (sometimes this is the case)
        elif 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        # If the ckpt only saved the state_dict directly
        else:
            model.load_state_dict(ckpt)

        # model.load_state_dict(ckpt)

        return model

    
    # Inference (for real-world input)

    def define_inputs(self, image, age, gender, header, coords):
        self.image = image
        self.age = age
        self.gender = gender
        self.header = header
        self.coords = coords

    def extract_patch(self, coord, output_shape, mode):

        patch = dataloader.extract_patch(
            CTData=self.image,
            coord=coord,
            srcVoxelOrigin=self.header["origin"],
            srcWorldMatrix=self.header["transform"],
            srcVoxelSpacing=self.header["spacing"],
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
            ),
            coord_space_world=True,
            mode=mode,
        )

        # ensure same datatype...
        patch = patch.astype(np.float32)

        # clip and scale...
        patch = dataloader.clip_and_scale(patch)
        return patch

    def _process_model(self, mode):

        if not self.suppress_logs:
            logging.info("Processing in " + mode)

        if mode == "2D":
            output_shape = [1, self.size_px, self.size_px]
        else:
            output_shape = [self.size_px, self.size_px, self.size_px]

        age_norm = dataloader_mask.normalize_age(self.age)
        gender_norm = dataloader_mask.normalize_gender(self.gender)

        nodules = []

        for _coord in self.coords:

            patch = self.extract_patch(_coord, output_shape, mode=mode)
            nodules.append(patch)

        nodules = np.array(nodules)
        nodules = torch.from_numpy(nodules).to(self.device)
        nodules = torch.nn.functional.interpolate(nodules, size=(64, int(64*self.config.UP_SCALE), int(64*self.config.UP_SCALE)))

        N = len(self.coords)

        ages = age_norm.unsqueeze(0).repeat(N, 1).to(self.device)
        genders = gender_norm.unsqueeze(0).repeat(N, 1).to(self.device)

        # print(ages.shape, ages[0], ages[0].shape)
        # print(genders.shape, genders[0], genders[0].shape)

        model = self._load_ckpt()        
        model.eval()

        features = model.extract_feature(nodules)
        logits, _ = model(
            features, 
            ages, 
            genders,
            self.config.AUX_VALIDATE,
        )

        logits = logits.data.cpu().numpy()
        logits = np.array(logits)

        return logits

    def predict(self):

        logits = self._process_model(self.mode)

        probability = torch.sigmoid(torch.from_numpy(logits)).numpy()
        return probability, logits
    

    # Model Inference (for valid, calibration, thresholding)

    def _run(self, model, input_images, input_ages, input_genders):
        features = model.extract_feature(input_images)
        cls_outputs, _ = model(
            features_main=features,
            age=input_ages,
            gender=input_genders,
        )

        return cls_outputs
    
    def _valid(self, model, valid_loader):
        model.eval()

        with torch.no_grad():

            y_pred = torch.tensor([], dtype=torch.float32, device=self.device)
            y = torch.tensor([], dtype=torch.float32, device=self.device)
            
            for val_data in tqdm(valid_loader, ncols=128):
                val_images, _, val_labels, val_ages, val_genders = (
                    val_data["image"].to(self.device),
                    val_data["mask"].to(self.device),
                    val_data["label"].float().to(self.device),
                    val_data["age"].to(self.device),
                    val_data["gender"].to(self.device),
                )
                
                val_images = torch.nn.functional.interpolate(val_images, size=(64, int(64*self.config.UP_SCALE), int(64*self.config.UP_SCALE)))
                
                cls_outputs = self._run(model, val_images, val_ages, val_genders)
                
                y_pred = torch.cat([y_pred, cls_outputs], dim=0)
                y = torch.cat([y, val_labels], dim=0)

        return y_pred, y

    
    # Validation

    def init_valid(self):
        # Load Metadata
        metadata = np.load(
            Path(self.model_dir) / self.model_name / "metadata.npy",
            allow_pickle=True
        ).item()
        # print(metadata)

        # Load Data
        valid_csv_path = metadata['valid_csv']
        # print(valid_csv_path)

        torch.manual_seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        random.seed(self.config.SEED)
        
        # Load original datasets
        if "mask_mclab" in str(valid_csv_path):
            valid_csv_path = str(valid_csv_path).replace("mask_mclab", "preload/preload_mclab")
        valid_df = pd.read_csv(valid_csv_path)

        print(f"Valid samples: {len(valid_df)}")

        valid_loader = get_data_loader(
            self.config.DATADIR,
            self.config.MASK_DATADIR,
            valid_df,  # Use new validation set (200 samples)
            mode=self.config.MODE,
            workers=self.config.NUM_WORKERS,
            batch_size=self.config.BATCH_SIZE,
            rotations=None,
            translations=None,
            size_mm=self.config.SIZE_MM,
            size_px=self.config.SIZE_PX,
        )

        return valid_loader

    def _compute_metrics(self, y_pred, y):
        # ===== ROC =====
        fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
        roc_auc = metrics.auc(fpr, tpr)

        # ===== PR-AUC =====
        pr_auc = metrics.average_precision_score(y, y_pred)

        # ===== Threshold (0.5 hoặc threshold đã tune) =====
        threshold = 0.5
        y_hat = (y_pred >= threshold).astype(int)

        # ===== Confusion matrix =====
        tn, fp, fn, tp = metrics.confusion_matrix(y, y_hat).ravel()

        # ===== Metrics =====
        sensitivity = tp / (tp + fn + 1e-8)      # Recall
        specificity = tn / (tn + fp + 1e-8)
        balanced_acc = 0.5 * (sensitivity + specificity)
        f1 = metrics.f1_score(y, y_hat, zero_division=0)

        return {
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "BalancedAcc": balanced_acc,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "F1": f1,
        }

    def validate(self, valid_loader): 
        mapping_metric = {
            "val_auc": "ROC-AUC",
            "val_pr_auc": "PR-AUC",
            "val_sensitivity": "Sensitivity",
            "val_specificity": "Specificity",
            "val_f1": "F1",
        }

        log_path = os.path.join(self.model_dir, self.model_name, "run_logging.json")
        logger = TrainVisualizer.from_json(log_path)

        sub = logger.summarize_best_epoch(
            ema_enable=False,
        )

        metrics_dict = {}
        for metric in sub["metrics"].keys():
            if sub["metrics"][metric] is not None:
                metrics_dict[mapping_metric[metric]] = sub["metrics"][metric]

        if len(metrics_dict.keys()) == 4:
            print("Validation already done in training phase!")
            metrics_dict['ROC-AUC'] = logger.best_records[sub["auc_key"]]["value"]
            metrics_dict["BalancedAcc"] = 0.5 * (metrics_dict["Sensitivity"] + metrics_dict["Specificity"])

            metric_order = ["ROC-AUC", "PR-AUC", "BalancedAcc", "Sensitivity", "Specificity", "F1"]
            metrics_dict = {metric: metrics_dict[metric] for metric in metric_order}

            return metrics_dict

        print("Validation is missing metric . . .")

        model = self._load_ckpt()
        
        y_pred, y = self._valid(model, valid_loader)

        y_pred = torch.sigmoid(y_pred.reshape(-1)).data.cpu().numpy().reshape(-1)
        y = y.data.cpu().numpy().reshape(-1)

        metrics_dict = self._compute_metrics(y_pred, y)

        # Saving sub records
        sub_records = {
            "epoch": logger.best_records['val_auc']['epoch'],
            "auc_key": 'val_auc',
            "metrics": {
                metric: metrics_dict[mapping_metric[metric]] if metric in mapping_metric else None for metric in logger.get_sub_candidates()
            },
        }
        # print(sub_records)

        logger.set_sub_records(sub_records)
        logger.export_json(log_path)

        return metrics_dict
    

    # Calibration
    
    def calibrate(self, valid_loader):
        # def calibration_loss():
        #     scaled_logits = logits_val / T
        #     loss = criterion(scaled_logits, labels_val)
        #     return loss

        calib_path = os.path.join(self.model_dir, self.model_name, "calibration.npy")

        if os.path.exists(calib_path):
            print(f"Reading T_opt from {calib_path} . . .", end=" ")
            self.temperature = np.load(calib_path)
            print(f"Done!")

            print(f"[Calibration] Optimal Temperature: {T_opt:.4f}")
            return
        
        print("Calibration is not saved . . .")

        model = self._load_ckpt()
        
        # 1. Collect validation logits & labels
        y_pred, y = self._valid(model, valid_loader)

        y_pred = y_pred.view(-1)
        y = y.view(-1)

        # 2. Temperature parameter
        T = torch.nn.Parameter(
            torch.ones(1, device=self.device)
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

        # 3. Optimization step
        def closure():
            optimizer.zero_grad()

            # Clamp T to avoid numerical issues
            T_clamped = torch.clamp(T, min=1e-6)

            scaled_logits = y_pred / T_clamped
            loss = criterion(scaled_logits, y)

            loss.backward()
            return loss
        
        optimizer.step(closure)

        T_opt = T.detach().item()

        print(f"[Calibration] Optimal Temperature: {T_opt:.4f}")

        # 4. Save temperature (recommend)
        self.temperature = T_opt

        np.save(
            calib_path,
            T_opt
        )

        print(f"T_opt saved to {calib_path}.")

        return T_opt
    
# ------------------- Detector ------------------- 

import torch
import numpy as np

from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
    DeleteItemsd,
)

from monai.apps.detection.transforms.dictionary import (
    ClipBoxToImaged,
    AffineBoxToWorldCoordinated,
    ConvertBoxModed,
)

from monai.data import Dataset, DataLoader
from monai.data.utils import no_collation


# define detector
class MalignancyDetector:
    """
    Lung nodule detector (MONAI RetinaNet).
    Detects candidate nodules before malignancy classification.
    """

    def __init__(self, model_path, device="cuda"):

        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path

        # Detector configs

        self.spatial_dims = 3
        self.num_classes = 1

        self.size_divisible = [16, 16, 8]
        self.infer_patch_size = [512, 512, 192]

        self.feature_map_scales = [1, 2, 4]

        self.base_anchor_shapes = [
            [6, 8, 4],
            [8, 6, 5],
            [10, 10, 6],
        ]

        self.box_key = "box"
        self.label_key = "label"

        self.score_thresh = 0.02
        self.topk_candidates_per_level = 1000
        self.nms_thresh = 0.22
        self.detections_per_img = 300

        self.overlap = 0.25
        self.sw_batch_size = 1
        self.mode = "constant"

        self.pixdim = [0.703125, 0.703125, 1.25]

        self.a_min = -1024.0
        self.a_max = 300.0
        self.b_min = 0.0
        self.b_max = 1.0
        self.clip = True

        self.score_keep = 0.3

        # Build components

        # preprocessing / postprocessing
        self.preprocess = self._build_preprocess()
        self.postprocess = self._build_postprocess()

        # build detector
        self.detector = self._build_detector()

    # Build preprocess
    def _build_preprocess(self, image_key="image"):

        keys = [image_key]

        transforms = [
            LoadImaged(keys=keys, reader="itkreader", affine_lps_to_ras=True),

            EnsureChannelFirstd(keys=keys),

            Orientationd(
                keys=keys,
                axcodes="RAS",
                labels=(("L", "R"), ("P", "A"), ("I", "S")),
            ),

            Spacingd(
                keys=keys,
                pixdim=self.pixdim,
                mode="bilinear",
                padding_mode="border",
            ),

            ScaleIntensityRanged(
                keys=keys,
                a_min=self.a_min,
                a_max=self.a_max,
                b_min=self.b_min,
                b_max=self.b_max,
                clip=self.clip,
            ),

            EnsureTyped(keys=keys),
        ]

        return Compose(transforms)
    
    # Build postprocess
    def _build_postprocess(self, image_key="image", affine_lps_to_ras=True):

        return Compose(
            [
                ClipBoxToImaged(
                    box_keys="box",
                    label_keys="label",
                    box_ref_image_keys=image_key,
                    remove_empty=True,
                ),

                AffineBoxToWorldCoordinated(
                    box_keys="box",
                    box_ref_image_keys=image_key,
                    affine_lps_to_ras=affine_lps_to_ras,
                ),

                ConvertBoxModed(
                    box_keys="box",
                    src_mode="xyzxyz",
                    dst_mode="cccwhd",
                ),

                DeleteItemsd(keys=[image_key]),
            ]
        )
    
    # Build RetinaNet detector
    def _build_detector(self):

        network = torch.jit.load(self.model_path, map_location=self.device)

        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=self.feature_map_scales,
            base_anchor_shapes=self.base_anchor_shapes,
        )

        detector = RetinaNetDetector(
            network=network,
            anchor_generator=anchor_generator,
            spatial_dims=self.spatial_dims,
            num_classes=self.num_classes,
            size_divisible=self.size_divisible,
        )

        detector.set_target_keys(
            box_key=self.box_key,
            label_key=self.label_key,
        )

        detector.set_box_selector_parameters(
            score_thresh=self.score_thresh,
            topk_candidates_per_level=self.topk_candidates_per_level,
            nms_thresh=self.nms_thresh,
            detections_per_img=self.detections_per_img,
        )

        detector.set_sliding_window_inferer(
            roi_size=self.infer_patch_size,
            overlap=self.overlap,
            sw_batch_size=self.sw_batch_size,
            mode=self.mode,
            device=self.device,
        )

        detector.eval()

        return detector

    # Detection API
    def detect(self, nifti_path):

        data = [{"image": nifti_path}]

        ds = Dataset(data=data, transform=self.preprocess)

        dl = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=no_collation,
        )

        results = []

        for item in dl:

            item = item[0]

            image_4d = item["image"].to(self.device)

            if image_4d.dim() == 4:
                image_for_detector = image_4d.unsqueeze(0)
            else:
                image_for_detector = image_4d

            with torch.no_grad(), torch.amp.autocast(
                device_type="cuda",
                enabled=(self.device == "cuda"),
                dtype=torch.float16,
            ):

                out = self.detector(image_for_detector, use_inferer=True)

            out0 = out[0]

            boxes = out0["box"] if "box" in out0 else out0.get("boxes")
            labels = out0["label"] if "label" in out0 else out0.get("labels")
            scores = out0["label_scores"] if "label_scores" in out0 else out0.get("scores")

            boxes = boxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()

            pred = {
                "box": boxes,
                "label": labels,
                "label_scores": scores,
            }

            post_in = {**pred, "image": image_4d}

            post_out = self.postprocess(post_in)

            if self.score_keep is not None and len(post_out["label_scores"]) > 0:

                keep = post_out["label_scores"] >= float(self.score_keep)

                filtered_boxes = np.asarray(post_out["box"])[keep].tolist()

                results.extend(filtered_boxes)

        return results


# -------------------- Helper functions --------------------
import os
import zipfile
import json
import SimpleITK as sitk
from pathlib import Path

def process_luna25_zip(zip_path, extract_to, output_dir):
    """
    Unzip and convert DICOM series to NIfTI with metadata.
    """
    # 1. Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    os.makedirs(output_dir, exist_ok=True)
    metadata_all = {}

    # 2. Search for DICOM files in the extracted folder (recursive)
    # Assume: extract_to/study_folder/series_folders/instances.dcm
    for root, dirs, files in os.walk(extract_to):
        if not any(f.lower().endswith('.dcm') for f in files):
            continue
            
        series_reader = sitk.ImageSeriesReader()
        series_ids = series_reader.GetGDCMSeriesIDs(root)
        
        if not series_ids:
            continue

        for s_id in series_ids:
            # Get file names for the series
            dicom_names = series_reader.GetGDCMSeriesFileNames(root, s_id)
            series_reader.SetFileNames(dicom_names)
            
            # Read 3D
            image = series_reader.Execute()
            
            # Read metadata from the first file in the series
            first_file_reader = sitk.ImageFileReader()
            first_file_reader.SetFileName(dicom_names[0])
            first_file_reader.LoadPrivateTagsOn()
            first_file_reader.ReadImageInformation()
            
            # Extract metadata (using DICOM Tags)
            # 0020|000e: Series Instance UID
            # 0010|0040: Patient Sex
            # 0010|1010: Patient Age
            s_uid = first_file_reader.GetMetaData("0020|000e").strip()
            p_sex = first_file_reader.GetMetaData("0010|0040").strip() if first_file_reader.HasMetaDataKey("0010|0040") else "U"
            p_age = first_file_reader.GetMetaData("0010|1010").strip() if first_file_reader.HasMetaDataKey("0010|1010") else "0"

            # 3. Convert to NIfTI and save
            file_name = f"{s_uid}.nii.gz"
            sitk.WriteImage(image, os.path.join(output_dir, file_name))
            
            # Save metadata to dictionary (keyed by SeriesInstanceUID)
            metadata_all[s_uid] = {
                "SeriesInstanceUID": s_uid,
                "PatientSex": p_sex,
                "PatientAge": p_age,
                "File": file_name
            }
            print(f" + Converted: {s_uid}")

    # 4. Save metadata to JSON
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata_all, f, indent=4)
    
    print(f" > Done! Metadata saved to {output_dir}/metadata.json ")
    
def process_raw_clinical(metadata, s_uid):
    # Sex
    sex = metadata[s_uid].get("PatientSex", "M")

    if sex in ["F", "Female"]:
        sex = "Female"
    else:
        sex = "Male"

    # Age
    age = metadata[s_uid].get("PatientAge", "0")

    if isinstance(age, str) and age.endswith("Y"):
        age = int(age[:-1])
    else:
        age = int(age)

    return sex, age

# sanity test
if __name__ == "__main__":
    # device= "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    
    """
    Simple sanity test for MalignancyProcessor
    This test checks whether the inference pipeline can run end-to-end.
    """

    logging.info("Running MalignancyProcessor sanity test")

    # -----------------------------
    # Init processor
    # -----------------------------
    processor = MalignancyProcessor(
        suppress_logs=False,
        model_name="LUNA25-aux-film-baseline-with-aux-seg-clinical-gate_mask_mclab_split-multitask-3D-20251217",
        model_dir="./results",
        device=device
    )

    # -----------------------------
    # Mock input data
    # -----------------------------

    # Fake CT volume (D, H, W)
    image = np.random.randn(128, 512, 512).astype(np.float32)

    # Fake header info (đúng key là đủ)
    header = {
        "origin": np.array([0.0, 0.0, 0.0]),
        "spacing": np.array([1.0, 1.0, 1.0]),
        "transform": np.eye(3),
    }

    # One or more nodule world coordinates (x, y, z)
    coords = [
        np.array([10.0, 20.0, 30.0]),
        np.array([40.0, 50.0, 60.0]),
    ]

    # Clinical info
    age = 65
    gender = "Male"  # hoặc "Female"

    # -----------------------------
    # Define inputs
    # -----------------------------
    processor.define_inputs(
        image=image,
        age=age,
        gender=gender,
        header=header,
        coords=coords
    )

    # -----------------------------
    # Run prediction
    # -----------------------------
    with torch.no_grad():
        probs, logits = processor.predict()

    # -----------------------------
    # Print results
    # -----------------------------
    print("Logits:", logits)
    print("Probabilities:", probs)

    logging.info("Sanity test finished successfully")

    """
    Simple sanity test for MalignancyDetector
    """
    logging.info("Running MalignancyDetector sanity test")

    import tempfile
    import SimpleITK as sitk

    # -----------------------------
    # Create a mock CT volume
    # -----------------------------
    mock_ct = np.random.randn(128, 256, 256).astype(np.float32)

    sitk_img = sitk.GetImageFromArray(mock_ct)
    sitk_img.SetSpacing((1.0, 1.0, 1.0))
    sitk_img.SetOrigin((0.0, 0.0, 0.0))

    # Save temporary NIfTI file
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, "mock_ct.nii.gz")
    sitk.WriteImage(sitk_img, tmp_path)

    logging.info(f"Mock CT saved to {tmp_path}")

    # -----------------------------
    # Init detector
    # -----------------------------
    detector = MalignancyDetector(
        model_path="./resources/dt_model.ts",  
        device=device
    )

    # -----------------------------
    # Run detection
    # -----------------------------
    try:
        boxes = detector.detect(tmp_path)

        print("Detected boxes (cccwhd):")
        print(boxes)

    except Exception as e:
        logging.error(f"Detector test failed: {e}")

    logging.info("MalignancyDetector sanity test finished")
