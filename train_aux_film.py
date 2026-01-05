from models.mclab.model_2d import ResNet18
from models.mclab.model_3d import I3D
from models.multi_task_film_model import *

from dataloader_mask import get_data_loader

from train_visualizer import TrainVisualizer

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import sklearn.metrics as metrics
import random
import pandas
from experiment_config import config
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

from preprocessing.mclab_utils import rot_flip_yz_2

from torch_ema import ExponentialMovingAverage

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

def compute_metrics(y_pred, y):
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
    f1 = metrics.f1_score(y, y_hat, zero_division=0)

    return roc_auc, pr_auc, sensitivity, specificity, f1


def make_weights_for_balanced_classes(labels):
    # Making sampling weights for the data samples
    n_samples = len(labels)
    unique, cnts = np.unique(labels, return_counts=True)
    cnt_dict = dict(zip(unique, cnts))

    weights = []
    for label in labels:
        weights.append(n_samples / float(cnt_dict[label]))
    return weights

class DiceLoss(nn.Module):
    # Dice Loss for segmentation tasks
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice

def train_multitask(experiment_name, train_csv_path, valid_csv_path, exp_save_root):

    # --------- Initialize ---------- 
    
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # # reproducible harder
    # torch.manual_seed(config.SEED)
    # torch.cuda.manual_seed_all(config.SEED)
    # np.random.seed(config.SEED)
    # random.seed(config.SEED)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    if not config.validate_config():
        raise RuntimeError("Config is incomplete. Please fix missing parameters.")

    if config.USE_AUX_MODEL:
        scaler = torch.cuda.amp.GradScaler()
    
    config.print()

    # --------- Load Data ---------

    # Load original datasets
    train_df = pandas.read_csv(train_csv_path)
    valid_df = pandas.read_csv(valid_csv_path)
    
    # Combine all data first
    all_df = pandas.concat([train_df, valid_df], ignore_index=True)
    
    # Get unique patients from validation set for patient-level sampling
    unique_patients = valid_df['PatientID'].unique()
    
    # Randomly sample patients to get approximately 200 samples for validation
    np.random.shuffle(unique_patients)
    
    # Find the number of patients needed to get close to 200 samples
    cumulative_samples = 0
    selected_patients = []
    valid_samples_num = config.VALID_SAMPLES_NUM
    
    for patient in unique_patients:
        patient_samples = len(valid_df[valid_df['PatientID'] == patient])
        if cumulative_samples + patient_samples <= valid_samples_num:
            selected_patients.append(patient)
            cumulative_samples += patient_samples
        else:
            # If adding this patient would exceed 200, decide based on how close we are
            if abs(valid_samples_num - cumulative_samples) > abs(valid_samples_num - (cumulative_samples + patient_samples)):
                selected_patients.append(patient)
                cumulative_samples += patient_samples
            break
    
    # If we still need more samples and there are remaining patients
    if cumulative_samples < valid_samples_num and len(selected_patients) < len(unique_patients):
        remaining_patients = [p for p in unique_patients if p not in selected_patients]
        if remaining_patients:
            selected_patients.append(remaining_patients[0])
    
    # Create new validation set with selected patients
    new_valid_df = valid_df[valid_df['PatientID'].isin(selected_patients)].reset_index(drop=True)
    
    # Create new training set: all data except the selected validation patients
    new_train_df = all_df[~all_df['PatientID'].isin(selected_patients)].reset_index(drop=True)
    
    logging.info(f"Original train samples: {len(train_df)}")
    logging.info(f"Original valid samples: {len(valid_df)}")

    logging.info(f"Total unique patients in validation set: {len(unique_patients)}")
    logging.info(f"Selected patients for validation: {len(selected_patients)}")
    logging.info(f"New train samples: {len(new_train_df)}")
    logging.info(f"New valid samples: {len(new_valid_df)} (target was {valid_samples_num})")
    
    logging.info(f"Number of malignant training samples: {new_train_df.label.sum()}")
    logging.info(f"Number of benign training samples: {len(new_train_df) - new_train_df.label.sum()}")
    logging.info(f"Number of malignant validation samples: {new_valid_df.label.sum()}")
    logging.info(f"Number of benign validation samples: {len(new_valid_df) - new_valid_df.label.sum()}")

    # Create data loaders with balanced sampling
    weights = make_weights_for_balanced_classes(new_train_df.label.values)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(new_train_df))
    
    train_loader = get_data_loader(
        config.DATADIR, 
        config.MASK_DATADIR,
        new_train_df,  # Use new combined training set
        mode=config.MODE,
        sampler=sampler,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=config.ROTATION,
        translations=config.TRANSLATION,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )

    valid_loader = get_data_loader(
        config.DATADIR,
        config.MASK_DATADIR,
        new_valid_df,  # Use new validation set (200 samples)
        mode=config.MODE,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=None,
        translations=None,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )

    # ---------- Multi-task Feature-wise Linear Modulation Model ----------

    device = torch.device("cuda")

    # Feature Extractor
    if config.MODE == "2D":
        feature_extractor = ResNet18().to(device)

    elif config.MODE == "3D":
        feature_extractor = I3D(
            num_classes=1,
            input_channels=3,
            pre_trained=True, 
            dropout_prob=config.DROP_RATE, 
            freeze_bn=True, 
            extract_feature=True, 
        ).to(device)

    # Auxiliary Model
    aux_model = SegmentationHead(
        in_channels=1024, 
        out_channels=1, 
        target_size=(64,64,64)
    ).to(device)

    # Build main Model
    model = MultiTaskFiLMModel(
        feature_extractor, 
        aux_model=aux_model, 
        aux_task=config.AUX_TASK,
        use_aux_model=config.USE_AUX_MODEL,
        use_seg_gate=config.USE_SEG_GATE,
        use_clinical_gate=config.USE_CLINICAL_GATE,
        clinical_dim=config.CLINICAL_DIM,
        cls_head_type=config.CLS_HEAD_TYPE,
        cls_tail_type=config.CLS_TAIL_TYPE
    ).to(device)

    # ---------- Training Setup ----------

    # EMA
    ema = ExponentialMovingAverage(
        model.parameters(), 
        decay=config.EMA_RATE
    )
    
    # Loss function
    loss_function = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    if config.SCHEDULER == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.STEP_LR_SIZE, gamma=0.3)
    elif config.SCHEDULER == "step_custom":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.DOWNSTEPS, gamma=0.3
        )

    # ---------- Train Visualizer ---------- 
    train_logger = TrainVisualizer(
        experiment_name=experiment_name,
        use_ema=True, 
        ema_decay=0.8
    )

    train_logger.add_criteria("cls_loss", higher_is_better=False)
    train_logger.add_criteria("dice_loss", higher_is_better=False)
    train_logger.add_criteria("auc", higher_is_better=True)
    train_logger.add_criteria("pr-auc", higher_is_better=True)
    train_logger.add_criteria("sensitivity", higher_is_better=True)
    train_logger.add_criteria("specificity", higher_is_better=True)
    train_logger.add_criteria("f1-score", higher_is_better=True)

    # ---------- Training Loop ----------

    best_metric = -1
    best_metric_epoch = -1
    epochs = config.EPOCHS
    patience = config.PATIENCE
    counter = 0

    for epoch in range(epochs):
        if counter > patience:
            logging.info(f"Model not improving for {patience} epochs")
            break

        logging.info("-" * 10)
        logging.info("epoch {}/{}".format(epoch + 1, epochs))

        # Train
        model.train()

        epoch_cls_loss = 0
        if config.USE_AUX_MODEL:
            epoch_seg_loss = 0
        step = 0

        for batch_data in tqdm(train_loader):
            step += 1

            # Input Prepare
            inputs, masks, labels, ages, genders = (
                batch_data["image"].to(device), 
                batch_data["mask"].to(device), 
                batch_data["label"].float().to(device), 
                batch_data["age"].to(device), 
                batch_data["gender"].to(device)
            )
            
            benigns = torch.where(labels.squeeze() == 0)
            malignants = torch.where(labels.squeeze() == 1)
            
            if random.random() < config.RF_RATIO[0]:
                inputs[benigns], masks[benigns] = rot_flip_yz_2(inputs[benigns], masks[benigns])
            if random.random() < config.RF_RATIO[1]:
                inputs[malignants], masks[malignants] = rot_flip_yz_2(inputs[malignants], masks[malignants])

            inputs = torch.nn.functional.interpolate(inputs, size=(64, int(64*config.UP_SCALE), int(64*config.UP_SCALE)))

            # Model training
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type):
                features = model.extract_feature(inputs)
                cls_outputs, seg_outputs = model(
                    features_main=features,
                    age=ages,
                    gender=genders
                )

                loss_cls = loss_function(cls_outputs.squeeze(), labels.squeeze())
                if config.USE_AUX_MODEL:
                    loss_seg = dice_loss(seg_outputs, masks)
                if config.USE_AUX_MODEL:
                    loss = loss_cls + config.AUX_LOSS_WEIGHT * loss_seg
                else:
                    loss = loss_cls

            # Backward
            if config.USE_AUX_MODEL:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            ema.update()

            # Calc loss step
            epoch_cls_loss += loss_cls.item()
            if config.USE_AUX_MODEL:
                epoch_seg_loss += loss_seg.item()
            
            if step % 100 == 0:
                epoch_len = len(new_train_df) // train_loader.batch_size

                if config.USE_AUX_MODEL:
                    logging.info(
                        "{}/{}, train_cls_loss: {:.4f}, train_seg_loss: {:.4f}".format(step, epoch_len, loss_cls.item(), loss_seg.item())
                    )
                else:
                    logging.info(
                        "{}/{}, train_cls_loss: {:.4f}".format(step, epoch_len, loss_cls.item())
                    )
            
        # Calc loss epoch
        epoch_cls_loss /= step
        if config.USE_AUX_MODEL:
            epoch_seg_loss /= step

        if config.USE_AUX_MODEL:
            logging.info(
                "epoch {} average train cls loss: {:.4f}, train seg loss: {:.4f}".format(epoch + 1, epoch_cls_loss, epoch_seg_loss)
            )
        else:
            logging.info(
                "epoch {} average train cls loss: {:.4f}".format(epoch + 1, epoch_cls_loss)
            )

        if config.USE_AUX_MODEL:
            train_logger.log(epoch, prefix="train", cls_loss = epoch_cls_loss, seg_loss = epoch_seg_loss)
        else:
            train_logger.log(epoch, prefix="train", cls_loss = epoch_cls_loss)

        with ema.average_parameters():

            # Validate
            model.eval()

            epoch_cls_loss = 0
            if config.USE_AUX_MODEL and config.AUX_VALIDATE:
                epoch_seg_loss = 0
            step = 0

            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.float32, device=device)
                
                for val_data in valid_loader:
                    step += 1
                    val_images, val_masks, val_labels, val_ages, val_genders = (
                        val_data["image"].to(device),
                        val_data["mask"].to(device),
                        val_data["label"].float().to(device),
                        val_data["age"].to(device),
                        val_data["gender"].to(device),
                    )
                    
                    val_images = torch.nn.functional.interpolate(val_images, size=(64, int(64*config.UP_SCALE), int(64*config.UP_SCALE)))
                    
                    with torch.amp.autocast(device_type=device.type):
                        features = model.extract_feature(val_images)
                        cls_outputs, seg_outputs = model(
                            features_main=features,
                            age=val_ages,
                            gender=val_genders,
                            validate=not config.AUX_VALIDATE
                        )
                        
                        loss_cls = loss_function(cls_outputs.squeeze(), val_labels.squeeze())
                        if config.USE_AUX_MODEL and config.AUX_VALIDATE:
                            loss_seg = dice_loss(seg_outputs, val_masks)
                    
                    epoch_cls_loss += loss_cls.item()
                    if config.USE_AUX_MODEL and config.AUX_VALIDATE:
                        epoch_seg_loss += loss_seg.item()
                    
                    y_pred = torch.cat([y_pred, cls_outputs], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                # Calc Loss Epoch

                epoch_cls_loss /= step
                if config.USE_AUX_MODEL and config.AUX_VALIDATE:
                    epoch_seg_loss /= step
                
                if config.USE_AUX_MODEL and config.AUX_VALIDATE:
                    logging.info(
                        "epoch {} average valid cls loss: {:.4f}, valid seg loss: {:.4f}".format(epoch + 1, epoch_cls_loss, epoch_seg_loss)
                    )
                else:
                    logging.info(
                        "epoch {} average valid cls loss: {:.4f}".format(epoch + 1, epoch_cls_loss)
                    )

                if config.USE_AUX_MODEL and config.AUX_VALIDATE:
                    train_logger.log(epoch, prefix="val", cls_loss = epoch_cls_loss, seg_loss = epoch_seg_loss)
                else:
                    train_logger.log(epoch, prefix="val", cls_loss = epoch_cls_loss)

                # Calc AUC
                y_pred = torch.sigmoid(y_pred.reshape(-1)).data.cpu().numpy().reshape(-1)
                y = y.data.cpu().numpy().reshape(-1)

                # fpr, tpr, _ = metrics.roc_curve(y, y_pred)
                # auc_metric = metrics.auc(fpr, tpr)

                auc_metric, pr_auc, sensitivity, specificity, f1 = compute_metrics(y_pred, y)

                if auc_metric > best_metric:
                    counter = 0
                    best_metric = auc_metric
                    best_metric_epoch = epoch + 1

                    # Save best model
                    torch.save(
                        model.state_dict(),
                        exp_save_root / "best_multitask_model.pth",
                    )
                    
                    # Save training metadata
                    metadata = {
                        "train_csv": train_csv_path,
                        "valid_csv": valid_csv_path,
                        "config": config,
                        "best_auc": best_metric,
                        "epoch": best_metric_epoch,
                        "data_split_info": {
                            "original_train_samples": len(train_df),
                            "original_valid_samples": len(valid_df),
                            "new_train_samples": len(new_train_df),
                            "new_valid_samples": len(new_valid_df),
                            "random_seed": config.SEED
                        }
                    }

                    np.save(
                        exp_save_root / "metadata.npy",
                        metadata,
                    )

                    # Save experiment config
                    config.save_json(
                        exp_save_root / "config.json",
                    )
                    
                    logging.info("saved new best multitask model")

                logging.info(
                    "current epoch: {} current AUC: {:.4f} best AUC: {:.4f} at epoch {}".format(
                        epoch + 1, auc_metric, best_metric, best_metric_epoch
                    )
                )

                train_logger.log(epoch, prefix="val", auc=auc_metric)
                train_logger.log(epoch, prefix="val", pr_auc=pr_auc)
                train_logger.log(epoch, prefix="val", sensitivity=sensitivity)
                train_logger.log(epoch, prefix="val", specificity=specificity)
                train_logger.log(epoch, prefix="val", f1=f1)

        # Saving and ploting training log after each epoch 
        train_logger.export_json(
            exp_save_root / "run_logging.json",
        )
        train_logger.save_plot(
            exp_save_root / "run_logging.png",
        )

        counter += 1
        scheduler.step()

    logging.info(
        "Multi-task training completed, best_metric: {:.4f} at epoch: {}".format(
            best_metric, best_metric_epoch
        )
    )

    # Show train visualize summary
    train_logger.summary()

    # Saving training logging
    train_logger.export_json(
        exp_save_root / "run_logging.json",
    )


def main():
    experiment_name = f"{config.EXPERIMENT_NAME}-multitask-{config.MODE}-{datetime.today().strftime('%Y%m%d')}"
    
    exp_save_root = config.EXPERIMENT_DIR / experiment_name
    exp_save_root.mkdir(parents=True, exist_ok=True)
    
    train_multitask(
        experiment_name,
        config.CSV_DIR_TRAIN,
        config.CSV_DIR_VALID,
        exp_save_root
    )
   
    logging.info(f"All results saved in: {exp_save_root}")

if __name__ == "__main__":
    main()