# ---------- Import ---------- #
import random
from datetime import datetime

import torch
from torch_ema import ExponentialMovingAverage
import sklearn.metrics as metrics

import numpy as np
import pandas as pd

from tqdm import tqdm

from experiment_config import config

from dataloader_mask import get_data_loader

from models.mclab.model_2d import ResNet18
from models.mclab.model_3d import I3D
from models.multi_task_film_model import *

from train_visualizer import TrainVisualizer

from preprocessing.mclab_utils import rot_flip_yz_2

torch.backends.cudnn.benchmark = True


# ---------- Logging ---------- #
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

# ---------- Helper Functions ---------- #
def init_seed(config):
    logging.info(f"Initialize SEED = {config.SEED} . . .")

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    logging.info("Done!\n")

def load_data(config):
    logging.info(f"Loading Data with split_preix = '{config.SPLIT_PREFIX}' . . .")

    # Get csv path
    train_csv_path = config.CSV_DIR_TRAIN
    valid_csv_path = config.CSV_DIR_VALID

    # Load original datasets
    train_df = pd.read_csv(train_csv_path)
    valid_df = pd.read_csv(valid_csv_path)
    
    # Combine all data first
    all_df = pd.concat([train_df, valid_df], ignore_index=True)
    
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

    # Logging
    logging.info("Done!\n")

    logging.info("Split Data information:")
    logging.info(f"Original train samples: {len(train_df)}")
    logging.info(f"Original valid samples: {len(valid_df)}\n")

    logging.info(f"Total unique patients in validation set: {len(unique_patients)}")
    logging.info(f"Selected patients for validation: {len(selected_patients)}")
    logging.info(f"New train samples: {len(new_train_df)}")
    logging.info(f"New valid samples: {len(new_valid_df)} (target was {valid_samples_num})\n")
    
    logging.info(f"Number of malignant training samples: {new_train_df.label.sum()}")
    logging.info(f"Number of benign training samples: {len(new_train_df) - new_train_df.label.sum()}")
    logging.info(f"Number of malignant validation samples: {new_valid_df.label.sum()}")
    logging.info(f"Number of benign validation samples: {len(new_valid_df) - new_valid_df.label.sum()}\n")

    return new_train_df, new_valid_df

def build_data_loader(config, train_df, valid_df):
    def make_weights_for_balanced_classes(labels):
        # Making sampling weights for the data samples
        n_samples = len(labels)
        unique, cnts = np.unique(labels, return_counts=True)
        cnt_dict = dict(zip(unique, cnts))

        weights = []
        for label in labels:
            weights.append(n_samples / float(cnt_dict[label]))
        return weights
    
    logging.info("Building Data Loader . . .")
    
    # Create data loaders with balanced sampling
    weights = make_weights_for_balanced_classes(train_df.label.values)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_df))
    
    train_loader = get_data_loader(
        config.DATADIR, 
        config.MASK_DATADIR,
        train_df,  # Use new combined training set
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
        valid_df,  # Use new validation set (200 samples)
        mode=config.MODE,
        workers=config.NUM_WORKERS,
        batch_size=config.BATCH_SIZE,
        rotations=None,
        translations=None,
        size_mm=config.SIZE_MM,
        size_px=config.SIZE_PX,
    )

    logging.info("Done!\n")

    return train_loader, valid_loader

def build_model(config, device):
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

    return model

def build_visualizer(experiment_name):
    train_logger = TrainVisualizer(
        experiment_name=experiment_name,
        use_ema=True, 
        ema_decay=0.8
    )

    # Loss
    train_logger.add_criteria("cls_loss", higher_is_better=False)
    train_logger.add_criteria("dice_loss", higher_is_better=False)

    # w_loss
    train_logger.add_criteria("w_loss", higher_is_better=False)

    # Metrics
    train_logger.add_criteria("auc", higher_is_better=True)
    train_logger.add_criteria("pr-auc", higher_is_better=True)
    train_logger.add_criteria("f1-score", higher_is_better=True)
    train_logger.add_criteria("sensitivity", higher_is_better=True)
    train_logger.add_criteria("specificity", higher_is_better=True)

    return train_logger

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
    

# ---------- Phase Aware EMA class ---------- #
class PhaseAwareEMA:
    def __init__(self, model, base_decay=0.999):
        self.model = model
        self.ema = ExponentialMovingAverage(
            model.parameters(),
            decay=base_decay
        )
        self.base_decay = base_decay
        self.current_decay = base_decay

    def set_decay(self, decay):
        self.current_decay = decay
        self.ema.decay = decay   # quan trọng

    def update(self):
        self.ema.update()

    def apply_shadow(self):
        self.ema.apply_shadow()

    def restore(self):
        self.ema.restore()

    def _phase_decay_schedule(
            self, 
            epoch, phase_epochs,
            min_decay=0.90,
            max_decay=0.999
        ):
        ratio = epoch / max(1, phase_epochs - 1)
        # cosine from 0 → 1
        cosine = 0.5 * (1 - math.cos(math.pi * ratio))
        return min_decay + cosine * (max_decay - min_decay)


# ---------- Phase Trainer class ---------- #
class PhaseTrainer:
    PHASE_CFG_REQUIRED_KEYS = [
        "name",
        "lr",
        "scheduler",
        "epochs",
        "patience",
        "train_parts",
        "freeze_parts",
        "loss"
    ]

    def __init__(
            self, 
            experiment_name,
            exp_save_root,
            config,
            train_df,
            train_loader, 
            valid_df,
            valid_loader, 
            device,
            model, 
            train_logger,
        ): 

        logging.info("Init Phase Trainer . . .")

        self.experiment_name = experiment_name
        self.exp_save_root = exp_save_root

        self.config = config

        self.train_df = train_df
        self.train_loader = train_loader
        self.valid_df = valid_df
        self.valid_loader = valid_loader

        self.model = model
        self.backbone_last = nn.ModuleList([
            self.model.backbone.mixed_5b,
            self.model.backbone.mixed_5c,
        ])

        self.device = device

        self.train_logger = train_logger

        logging.info("Done!\n")

    # Helper functions
    def _config_validate(self):
        phases = self.config.PHASE_CFG

        errors = []

        # -------- top-level --------
        if not isinstance(phases, list):
            raise TypeError("PHASE_CFG must be a list of phase configs")

        if len(phases) == 0:
            raise ValueError("PHASE_CFG must contain at least one phase")

        # -------- phase-level --------
        for i, phase in enumerate(phases):
            prefix = f"[Phase {i}]"

            if not isinstance(phase, dict):
                errors.append(f"{prefix} phase config must be a dict")
                continue

            # ---- required keys ----
            for key in self.PHASE_CFG_REQUIRED_KEYS:
                if key not in phase:
                    errors.append(f"{prefix} missing required key: '{key}'")

            # skip deeper checks if missing critical keys
            if any(k not in phase for k in ["lr", "loss", "scheduler"]):
                continue

            # ---- name ----
            if not isinstance(phase["name"], str):
                errors.append(f"{prefix} 'name' must be str")

            # ---- epochs / patience ----
            if not isinstance(phase["epochs"], int) or phase["epochs"] <= 0:
                errors.append(f"{prefix} 'epochs' must be positive int")

            if not isinstance(phase["patience"], int) or phase["patience"] < 0:
                errors.append(f"{prefix} 'patience' must be non-negative int")

            # ---- train_parts / freeze_parts ----
            available_parts = set(self.available_parts())

            for k in ["train_parts", "freeze_parts"]:
                if not isinstance(phase[k], list):
                    errors.append(f"{prefix} '{k}' must be a list")
                    continue

                for p in phase[k]:
                    if not isinstance(p, str):
                        errors.append(f"{prefix} '{k}' items must be str")
                        continue

                    if p not in available_parts:
                        errors.append(
                            f"{prefix} unknown part '{p}' in '{k}'. "
                            f"Available parts: {sorted(available_parts)}"
                        )

            # ---- overlap check ----
            train_set = set(phase["train_parts"])
            freeze_set = set(phase["freeze_parts"])

            overlap = train_set & freeze_set
            if overlap:
                errors.append(
                    f"{prefix} train_parts and freeze_parts overlap: {sorted(overlap)}"
                )

            # ---- lr ----
            if not isinstance(phase["lr"], dict):
                errors.append(f"{prefix} 'lr' must be a dict")
            else:
                for part, lr in phase["lr"].items():
                    if not isinstance(part, str):
                        errors.append(f"{prefix} lr key must be str (module name)")
                    if not isinstance(lr, (float, int)) or lr <= 0:
                        errors.append(
                            f"{prefix} lr for '{part}' must be positive number"
                        )

            # ---- loss ----
            if not isinstance(phase["loss"], dict):
                errors.append(f"{prefix} 'loss' must be a dict")
            else:
                for loss_name, cfg in phase["loss"].items():
                    if not isinstance(cfg, dict):
                        errors.append(
                            f"{prefix} loss '{loss_name}' must be a dict"
                        )
                        continue

                    if "start" not in cfg or "end" not in cfg:
                        errors.append(
                            f"{prefix} loss '{loss_name}' must have 'start' and 'end'"
                        )
                        continue

                    for k in ["start", "end"]:
                        if not isinstance(cfg[k], (float, int)):
                            errors.append(
                                f"{prefix} loss '{loss_name}.{k}' must be number"
                            )

            # ---- scheduler ----
            sched = phase["scheduler"]
            if not isinstance(sched, dict):
                errors.append(f"{prefix} 'scheduler' must be dict")
            else:
                if "type" not in sched:
                    errors.append(f"{prefix} scheduler missing 'type'")
                else:
                    stype = sched["type"]

                    if stype == "step":
                        if "step_lr_size" not in sched:
                            errors.append(
                                f"{prefix} step scheduler requires 'step_lr_size'"
                            )

                    elif stype == "cosine":
                        if "T_max" not in sched:
                            errors.append(
                                f"{prefix} cosine scheduler requires 'T_max'"
                            )

                    elif stype == "custom":
                        if "downsteps" not in sched:
                            errors.append(
                                f"{prefix} custom scheduler requires 'downsteps'"
                            )

                    else:
                        errors.append(
                            f"{prefix} unknown scheduler type '{stype}'"
                        )

        # -------- ema ---------
        for i, _ in enumerate(phases):
            if 'ema' not in phases[i]:
                phases[i]['ema'] = {
                    "min": 0.90,
                    "max": 0.999
                }

        # -------- report --------
        if errors:
            msg = "\n".join(errors)
            raise ValueError(
                "[PHASE_CFG VALIDATION ERROR]\n" + msg
            )

        return True

    
    # Freeze/Unfreeze Model module
    def _set_trainable(self, train_parts, freeze_parts=None):
        train_parts = set(train_parts)
        freeze_parts = set(freeze_parts or [])

        overlap = train_parts & freeze_parts
        if overlap:
            raise ValueError(
                f"train_parts and freeze_parts overlap: {overlap}"
            )

        # Freeze all first
        for p in self.model.parameters():
            p.requires_grad = False

        # Explicit unfreeze
        for part in train_parts:
            if not self._check_part(part):
                continue
            module = self._get_module(part)
            for p in module.parameters():
                p.requires_grad = True
    
    def _log_trainable_params(self):
        print("[Trainable parameters]")
        for name, module in self.model.named_children():
            trainable = any(p.requires_grad for p in module.parameters())
            print(f"  {name}: {'TRAIN' if trainable else 'FREEZE'}")

    def _get_module(self, part_name): 
        if part_name == "backbone":
            return self.model.backbone

        if part_name == "backbone_last":
            return self.backbone_last
        
        if part_name == "aux_model":
            return self.model.aux_model
        
        if part_name == "seg_gate":
            return self.model.seg_gate
        
        if part_name == "clinical_gate":
            return self.model.clinical_gate
        
        if part_name == "classifier_head":
            return self.model.classifier_head
        
        elif hasattr(self.model, part_name):
            return getattr(self.model, part_name)

        else:
            raise KeyError(
                f"Unknown part_name '{part_name}'. "
                f"Check PHASE_CFG train_parts / freeze_parts."
            )

    def _check_part(self, part_name):
        if part_name == "seg_gate" and not self.config.USE_SEG_GATE:
            return False
        if part_name == "clinical_gate" and not self.config.USE_CLINICAL_GATE:
            return False

        if part_name == "backbone_last":
            return True
        
        # existence check
        if not hasattr(self.model, part_name):
            print(f"[WARN] Model has no part '{part_name}', skip.")
            return False

        return True

    def available_parts(self):
        return [
            "backbone",
            "backbone_last",
            "aux_model",
            "seg_gate",
            "clinical_gate",
            "classifier_head",
        ]
    


    # Optimizer
    def _build_optimizer(self, lr_cfg):
        param_groups = []
        for part, lr in lr_cfg.items():
            if not self._check_part(part):
                continue
            module = self._get_module(part)
            param_groups.append({
                "params": module.parameters(),
                "lr": lr
            })
        return torch.optim.AdamW(param_groups, weight_decay=self.config.WEIGHT_DECAY)
    
    def _build_scheduler(self, optimizer, scheduler_cfg):
        if scheduler_cfg["type"] == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=scheduler_cfg["step_lr_size"], gamma=0.3
            )

        elif scheduler_cfg["type"] == "step_custom":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=scheduler_cfg["downsteps"], gamma=0.3
            )

        elif scheduler_cfg["type"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=scheduler_cfg["T_max"]
            )

        return scheduler

    
    # Loss Scheduler
    def _interp(self, start, end, t):
        return start + t * (end - start)

    def _get_loss_weights(self, phase_cfg, epoch):
        t = epoch / max(1, phase_cfg["epochs"] - 1)
        w_seg = self._interp(
            phase_cfg["loss"]["seg"]["start"],
            phase_cfg["loss"]["seg"]["end"],
            t
        )
        w_cls = self._interp(
            phase_cfg["loss"]["cls"]["start"],
            phase_cfg["loss"]["cls"]["end"],
            t
        )
        return w_seg, w_cls
    

    # Phase Training
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
        f1 = metrics.f1_score(y, y_hat, zero_division=0)

        return roc_auc, pr_auc, sensitivity, specificity, f1
    
    def train_phase(
            self, 
            phase_cfg, 
            ema_helper,
            best_metric, 
            best_metric_epoch, 
            epoch_offset = 0
        ):
        def _train_step(batch_data, cls_loss_function, seg_loss_function):
            # Input Prepare
            inputs, masks, labels, ages, genders = (
                batch_data["image"].to(self.device), 
                batch_data["mask"].to(self.device), 
                batch_data["label"].float().to(self.device), 
                batch_data["age"].to(self.device), 
                batch_data["gender"].to(self.device)
            )

            benigns = torch.where(labels.squeeze() == 0)
            malignants = torch.where(labels.squeeze() == 1)

            if random.random() < self.config.RF_RATIO[0]:
                inputs[benigns], masks[benigns] = rot_flip_yz_2(inputs[benigns], masks[benigns])
            if random.random() < self.config.RF_RATIO[1]:
                inputs[malignants], masks[malignants] = rot_flip_yz_2(inputs[malignants], masks[malignants])

            inputs = torch.nn.functional.interpolate(inputs, size=(64, int(64*self.config.UP_SCALE), int(64*self.config.UP_SCALE)))

            # Model training
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device.type):
                features = self.model.extract_feature(inputs)
                cls_outputs, seg_outputs = self.model(
                    features_main=features,
                    age=ages,
                    gender=genders
                )

                loss_cls = cls_loss_function(cls_outputs.squeeze(), labels.squeeze())
                loss_seg = seg_loss_function(seg_outputs, masks)

            return loss_cls, loss_seg

        def _valid_step(val_data, cls_loss_function, seg_loss_function):
            val_images, val_masks, val_labels, val_ages, val_genders = (
                val_data["image"].to(self.device),
                val_data["mask"].to(self.device),
                val_data["label"].float().to(self.device),
                val_data["age"].to(self.device),
                val_data["gender"].to(self.device),
            )
            
            val_images = torch.nn.functional.interpolate(val_images, size=(64, int(64*self.config.UP_SCALE), int(64*self.config.UP_SCALE)))
            
            with torch.amp.autocast(device_type=self.device.type):
                features = self.model.extract_feature(val_images)
                cls_outputs, seg_outputs = self.model(
                    features_main=features,
                    age=val_ages,
                    gender=val_genders,
                )
                
                loss_cls = cls_loss_function(cls_outputs.squeeze(), val_labels.squeeze())
                loss_seg = seg_loss_function(seg_outputs, val_masks)

            return cls_outputs, val_labels, loss_cls, loss_seg

        def _save_best():
            # Save best model
            torch.save(
                self.model.state_dict(),
                self.exp_save_root / "best_multitask_model.pth",
            )
            
            # Save training metadata
            metadata = {
                "train_csv": self.config.CSV_DIR_TRAIN,
                "valid_csv": self.config.CSV_DIR_VALID,
                "config_path": self.exp_save_root / "config.json",
                "best_auc": best_metric,
                "epoch": best_metric_epoch,
                "data_split_info": {
                    "train_samples": len(self.train_df),
                    "valid_samples": len(self.valid_df),
                    "random_seed": self.config.SEED
                }
            }

            np.save(
                self.exp_save_root / "metadata.npy",
                metadata,
            )

            # Save experiment config
            self.config.save_json(
                self.exp_save_root / "config.json",
            )
            
            logging.info("saved new best phase-training model")

        self._set_trainable(
            phase_cfg["train_parts"],
            phase_cfg["freeze_parts"]
        )
        self._log_trainable_params()

        # Loss Function
        loss_function = torch.nn.BCEWithLogitsLoss()
        dice_loss = DiceLoss()

        scaler = torch.cuda.amp.GradScaler()

        # Optimizer + Scheduler
        optimizer = self._build_optimizer(phase_cfg["lr"])
        scheduler = self._build_scheduler(optimizer, phase_cfg["scheduler"])

        epochs = phase_cfg["epochs"]
        epoch_start = epoch_offset
        epoch_end = epoch_offset

        patience = phase_cfg["patience"]
        counter = 0

        # Training loop
        for epoch_idx in range(epochs):
            if counter > patience:
                logging.info(f"Model not improving for {patience} epochs")
                break

            epoch_end = epoch_idx + epoch_offset
            epoch = epoch_end

            logging.info("-" * 10)
            logging.info("epoch {}/{}".format(epoch + 1, epochs + epoch_offset))

            # --- adjust EMA decay ---
            decay = ema_helper._phase_decay_schedule(
                epoch_idx, epochs,
                phase_cfg["ema"]['min'], phase_cfg["ema"]['max']
            )
            ema_helper.set_decay(decay)
            print(
                f"[EMA] epoch={epoch} decay={ema_helper.current_decay} with ema_min={phase_cfg['ema']['min']} and ema_max={phase_cfg['ema']['max']}"
            )

            # Train
            self.model.train()
            w_seg, w_cls = self._get_loss_weights(phase_cfg, epoch_idx)

            print(
                f"[LOSS] epoch={epoch} w_seg={w_seg} w_cls={w_cls}"
            )

            self.train_logger.log(epoch, prefix="seg", w_loss = w_seg)
            self.train_logger.log(epoch, prefix="cls", w_loss = w_cls)

            epoch_cls_loss = 0
            epoch_seg_loss = 0
            step = 0

            for batch_data in tqdm(self.train_loader, ncols=128, desc="Training"):
                step += 1

                loss_cls, loss_seg = _train_step(batch_data, loss_function, dice_loss)
                loss = w_cls * loss_cls + w_seg * loss_seg
                
                # Backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                ema_helper.update()

                # Calc loss step
                epoch_cls_loss += loss_cls.item()
                epoch_seg_loss += loss_seg.item()

            # Calc loss epoch
            epoch_cls_loss /= step
            epoch_seg_loss /= step

            logging.info(
                "epoch {} average train cls loss: {:.4f}, train seg loss: {:.4f}".format(epoch + 1, epoch_cls_loss, epoch_seg_loss)
            )

            self.train_logger.log(epoch, prefix="train", cls_loss = epoch_cls_loss, seg_loss = epoch_seg_loss)

            # ---------- ---------- ---------- #

            # Validate
            with ema_helper.ema.average_parameters():

                epoch_cls_loss = 0
                epoch_seg_loss = 0
                step = 0

                with torch.no_grad():
                    y_pred = torch.tensor([], dtype=torch.float32, device=self.device)
                    y = torch.tensor([], dtype=torch.float32, device=self.device)

                    for val_data in tqdm(self.valid_loader, ncols=128, desc="Validating"):
                        step += 1

                        cls_outputs, val_labels, loss_cls, loss_seg =_valid_step(val_data, loss_function, dice_loss)

                        epoch_cls_loss += loss_cls.item()
                        epoch_seg_loss += loss_seg.item()

                        y_pred = torch.cat([y_pred, cls_outputs], dim=0)
                        y = torch.cat([y, val_labels], dim=0)

                    # Calc loss epoch
                    epoch_cls_loss /= step
                    epoch_seg_loss /= step

                    logging.info(
                        "epoch {} average valid cls loss: {:.4f}, valid seg loss: {:.4f}".format(epoch + 1, epoch_cls_loss, epoch_seg_loss)
                    )

                    self.train_logger.log(epoch, prefix="val", cls_loss = epoch_cls_loss, seg_loss = epoch_seg_loss)

                    # Calc Metrics
                    y_pred = torch.sigmoid(y_pred.reshape(-1)).data.cpu().numpy().reshape(-1)
                    y = y.data.cpu().numpy().reshape(-1)

                    auc_metric, pr_auc, sensitivity, specificity, f1 = self._compute_metrics(y_pred, y)

                    if auc_metric > best_metric:
                        counter = 0
                        best_metric = auc_metric
                        best_metric_epoch = epoch + 1

                        _save_best()

                    logging.info(
                        "current epoch: {} current AUC: {:.4f} best AUC: {:.4f} at epoch {}".format(
                            epoch + 1, auc_metric, best_metric, best_metric_epoch
                        )
                    )

                    self.train_logger.log(epoch, prefix="val", auc=auc_metric)
                    self.train_logger.log(epoch, prefix="val", pr_auc=pr_auc)
                    self.train_logger.log(epoch, prefix="val", f1=f1)
                    self.train_logger.log(epoch, prefix="val", sensitivity=sensitivity)
                    self.train_logger.log(epoch, prefix="val", specificity=specificity)

            # Saving and ploting training log after each epoch 
            self.train_logger.export_json(
                self.exp_save_root / "run_logging.json",
            )
            self.train_logger.save_plot(
                self.exp_save_root / "run_logging.png",
            )

            counter += 1
            scheduler.step()

        logging.info(
            "Multi-task training completed, best_metric: {:.4f} at epoch: {}".format(
                best_metric, best_metric_epoch
            )
        )

        # Show train visualize summary
        self.train_logger.summary()

        # Saving training logging
        self.train_logger.export_json(
            self.exp_save_root / "run_logging.json",
        )

        return epoch_start, epoch_end, best_metric, best_metric_epoch

    def train(self):
        best_metric = -1
        best_metric_epoch = -1

        epoch_offset = 0
        phases = self.config.PHASE_CFG

        ema_helper = PhaseAwareEMA(self.model)

        for idx, phase_cfg in enumerate(phases):
            self._print_phase(phase_cfg, idx+1)
            print(f"==> Start phase: {phase_cfg['name']}")

            epoch_start, epoch_end, best_metric, best_metric_epoch = self.train_phase(
                phase_cfg, ema_helper, best_metric, best_metric_epoch, epoch_offset
            )
            self.train_logger.add_phase(phase_cfg['name'], epoch_start, epoch_end)
            epoch_offset = epoch_end + 1

            print("==> Done!\n")

        # Show train visualize summary
        self.train_logger.summary()

        # Export logging plot
        self.train_logger.save_plot(
            self.exp_save_root / "run_logging.png",
        )

        # Saving training logging
        self.train_logger.export_json(
            self.exp_save_root / "run_logging.json",
        )

    # Logging
    def _print_phase(self, phase_cfg, idx: int = None):
        prefix = f"[PHASE {idx}] " if idx is not None else "[PHASE] "
        name = phase_cfg.get("name", "unknown")

        print("\n" + "=" * 80)
        print(f"{prefix}{name}")
        print("=" * 80)

        # Epoch & patience
        print(f"- Epochs    : {phase_cfg.get('epochs')}")
        print(f"- Patience  : {phase_cfg.get('patience')}")

        # Scheduler
        sched = phase_cfg.get("scheduler", {})
        if sched:
            print(f"- Scheduler : {sched.get('type')} (T_max={sched.get('T_max')})")

        # Learning rates
        print("- Learning Rates:")
        for k, v in phase_cfg.get("lr", {}).items():
            print(f"    • {k:<16}: {v:.2e}")

        # Train / Freeze parts
        print("- Train parts :")
        for p in phase_cfg.get("train_parts", []):
            print(f"    ✓ {p}")

        if phase_cfg.get("freeze_parts"):
            print("- Freeze parts:")
            for p in phase_cfg.get("freeze_parts", []):
                print(f"    ✗ {p}")

        # Loss schedule
        print("- Loss weights:")
        for loss_name, cfg in phase_cfg.get("loss", {}).items():
            s, e = cfg.get("start"), cfg.get("end")
            arrow = "→" if s != e else "="
            print(f"    • {loss_name:<4}: {s:.3f} {arrow} {e:.3f}")

    def _print(self):
        print("\n" + "#" * 80)
        print("PHASE TRAINING CONFIGURATION SUMMARY")
        print("#" * 80)

        for i, phase_cfg in enumerate(self.phases):
            self._print_phase(phase_cfg, idx=i)

        print("#" * 80)


    # Dummy run
    def train_dummy(self):
        epoch_offset = 0
        phases = self.config.PHASE_CFG

        ema_helper = PhaseAwareEMA(self.model)

        ema_decays = []
        w_clses = []
        w_segs = []

        for idx, phase_cfg in enumerate(phases):
            print(f"==> Start phase: {phase_cfg['name']}")

            epochs = phase_cfg["epochs"]
            # epoch_start = epoch_offset
            epoch_end = epoch_offset

            # Training loop
            for epoch_idx in range(epochs):

                epoch_end = epoch_idx + epoch_offset
                epoch = epoch_end

                logging.info("-" * 10)
                logging.info("epoch {}/{}".format(epoch + 1, epochs + epoch_offset))

                # --- adjust EMA decay ---
                decay = ema_helper._phase_decay_schedule(
                    epoch, epochs,
                    phase_cfg["ema"]['min'], phase_cfg["ema"]['max']
                )       # Wrong
                # decay = ema_helper._phase_decay_schedule(
                #     epoch_idx, epochs,
                #     phase_cfg["ema"]['min'], phase_cfg["ema"]['max']
                # )       # Right

                ema_helper.set_decay(decay)

                print(
                    f"[EMA] epoch={epoch} decay={ema_helper.current_decay} with ema_min={phase_cfg['ema']['min']} and ema_max={phase_cfg['ema']['max']}"
                )

                ema_decays.append(decay)

                # Train
                w_seg, w_cls = self._get_loss_weights(phase_cfg, epoch)     # Wrong
                # w_seg, w_cls = self._get_loss_weights(phase_cfg, epoch_idx)     # Right

                print(
                    f"[LOSS] epoch={epoch} w_seg={w_seg} w_cls={w_cls}"
                )

                w_clses.append(w_cls)
                w_segs.append(w_seg)

            epoch_offset = epoch_end + 1
            print("==> Done!\n")

        print(f"EMA decays = {ema_decays[:40]}")
        print(f"W_cls = {w_clses[:40]}")
        print(f"W_seg = {w_segs[:40]}")

        return


# ---------- Main Function ---------- #
def main():
    # Config checking
    if not config.validate_config():
        raise RuntimeError("Config is incomplete. Please fix missing parameters.")
    
    config.print()
    
    # Experiment setup
    experiment_name = f"{config.EXPERIMENT_NAME}-multitask-{config.MODE}-{datetime.today().strftime('%Y%m%d')}"

    exp_save_root = config.EXPERIMENT_DIR / experiment_name
    exp_save_root.mkdir(parents=True, exist_ok=True)

    # Seed
    init_seed(config)

    # Data
    train_df, valid_df = load_data(config)
    train_loader, valid_loader = build_data_loader(config, train_df, valid_df)

    # Device
    device = torch.device("cuda")

    # Model
    model = build_model(config, device)

    # Visualizer
    train_logger = build_visualizer(experiment_name)


    # Phase Trainer
    phase_trainer = PhaseTrainer(
        experiment_name=experiment_name,
        exp_save_root=exp_save_root,
        config=config,
        train_df=train_df,
        train_loader=train_loader,
        valid_df=valid_df,
        valid_loader=valid_loader,
        device=device,
        model=model,
        train_logger=train_logger
    )

    if phase_trainer._config_validate():
        phase_trainer.train()

    logging.info(f"All results saved in: {exp_save_root}")


def dummy():
    experiment_name = "LUNA25-phase-training-cosine-avg-linear-with-aux_mask_mclab_split-multitask-3D-20251223-ema_error"
    exp_save_root = config.EXPERIMENT_DIR / experiment_name

    from experiment_config import Configuration
    dummy_config = Configuration(exp_save_root / "config.json")

    dummy_config.print()

    # Device
    device = torch.device("cuda")

    # Model
    model = build_model(config, device)

    dummy_phase_trainer = PhaseTrainer(
        experiment_name=experiment_name,
        exp_save_root=exp_save_root,
        config=dummy_config,
        train_df=None,
        train_loader=None,
        valid_df=None,
        valid_loader=None,
        device=device,
        model=model,
        train_logger=None
    )

    if dummy_phase_trainer._config_validate():
        dummy_phase_trainer.train_dummy()


if __name__ == "__main__":
    main()

    # dummy()