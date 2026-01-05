import json
import argparse
from pathlib import Path
from typing import Any, Dict

from typing import Optional


class Configuration(object):
    DEFAULT_CONFIG_FILE = "config.json"

    CONFIG_GROUPS = {
        "Working directory": [
            "WORKDIR",
            "RESOURCES",
            "MODEL_RGB_I3D",
        ],
        "Data parameters": [
            "DATADIR",
            "MASK_DATADIR",
            "SIZE_MM",
            "SIZE_PX",
            "PATCH_SIZE",
        ],
        "CSV split prefix and directory": [
            "CSV_DIR",
            "SPLIT_PREFIX",
            "CSV_DIR_TRAIN",
            "CSV_DIR_VALID",
        ],
        "Results (NAME + MODE)": [
            "EXPERIMENT_DIR",
            "EXPERIMENT_NAME",
            "MODE",
        ],
        "Training parameters": [
            "SEED",
            "NUM_WORKERS",
            "BATCH_SIZE",
            "ROTATION",
            "TRANSLATION",
            "ELASTIC",
            "EPOCHS",
            "PATIENCE",
            "LEARNING_RATE",
            "WEIGHT_DECAY",
        ],
        "MCLab configuration": [
            "VALID_SAMPLES_NUM",
            "SCHEDULER",
            "STEP_LR_SIZE",
            "DOWNSTEPS",
            "SUBMISSION",
            "SAVE_EPOCHS",
            "ROT_FLIP",
            "RF_RATIO",
            "DROP_RATE",
            "UP_SCALE",
            "EMA_RATE",
            "MIX_PROB",
            "LABEL_SMOOTHING_ALPHA"
        ],
        "MCLab Model configuration": [
            "AUX_TASK",
            "AUX_BATCH_SIZE",
            "DETECTOR_DROP_RATE",
            "AUX_LOSS_WEIGHT",
            "USE_AUX_MODEL",
            "USE_SEG_GATE",
            "USE_CLINICAL_GATE",
            "CLINICAL_DIM",
            "AUX_VALIDATE",
            "CLS_HEAD_TYPE",
            "CLS_TAIL_TYPE",
        ],
        "Phase Training configuration": [
            "PHASE_CFG"
        ]
    }

    SEMANTIC_KEYS = ["EXPERIMENT_DIR", "EXPERIMENT_NAME", "SPLIT_PREFIX", "MODE"]

    REQUIRED_CONFIG_GROUPS = {
        "Training parameters": [
            "BATCH_SIZE",
            "EPOCHS",
            "LEARNING_RATE",
        ],
        "Results (NAME + MODE)": [
            "EXPERIMENT_NAME",
            "MODE",
        ],
        "model_components": [
            "USE_AUX_MODEL",
            "USE_SEG_GATE",
            "USE_CLINICAL_GATE",
            "CLS_HEAD_TYPE",
            "CLS_TAIL_TYPE",
        ]
    }


    def __init__(self, config_file: Optional[str] = None, args: Optional[argparse.Namespace] = None) -> None:
        """
        Priority:
        1. Hardcoded defaults
        2. config.json (or provided config_file)
        3. CLI arguments
        """

        # ------------------
        # 1. Hardcoded defaults
        # ------------------
        self.WORKDIR = Path("/data/luna25/model/luna25-baseline-public")
        self.RESOURCES = self.WORKDIR / "resources"
        self.MODEL_RGB_I3D = self.RESOURCES / "model_rgb.pth"

        self.DATADIR = Path("/data/luna25/data/luna25_nodule_blocks")
        self.MASK_DATADIR = Path("/data/luna25/data/luna25_nodule_blocks_mask")

        self.CSV_DIR = Path("/data/luna25/data/dataset_csv")
        self.SPLIT_PREFIX = "normal"

        self.CSV_DIR_TRAIN = self.CSV_DIR / f"{self.SPLIT_PREFIX}_train.csv"
        self.CSV_DIR_VALID = self.CSV_DIR / f"{self.SPLIT_PREFIX}_valid.csv"

        self.EXPERIMENT_DIR = self.WORKDIR / "results"
        self.EXPERIMENT_NAME = f"LUNA25-baseline_{self.SPLIT_PREFIX}_split"
        self.MODE = "3D"

        self.SEED = 2025
        self.NUM_WORKERS = 8
        self.SIZE_MM = 50
        self.SIZE_PX = 64
        self.BATCH_SIZE = 32
        self.ROTATION = [[-20, 20], [-20, 20], [-20, 20]]
        self.TRANSLATION = True
        self.ELASTIC = False
        self.EPOCHS = 10
        self.PATIENCE = 20
        self.PATCH_SIZE = [64, 128, 128]        # default: [64, 128, 128]
        self.LEARNING_RATE = 1e-4               # default: 1e-4
        self.WEIGHT_DECAY = 5e-4

        # MCLAB CONFIG
        self.VALID_SAMPLES_NUM = 200

        self.SCHEDULER = "step_custom"          # "step" or "step_custom" or "cosine"
        # step
        self.STEP_LR_SIZE = 20
        # step_custom
        self.DOWNSTEPS = [60, 80]

        self.SUBMISSION = False
        self.SAVE_EPOCHS = []
        self.ROT_FLIP = 'yz' # 'xyz' or 'yz'
        self.RF_RATIO = [0.3, 0.3]
        self.DROP_RATE = 0.0
        self.UP_SCALE = 2.0
        self.EMA_RATE = 0.998
        self.MIX_PROB = 0.0
        self.LABEL_SMOOTHING_ALPHA = 0.0

        # MCLAB MODEL CONFIG
        self.AUX_TASK = "Segmentation"
        self.AUX_BATCH_SIZE = 16
        self.DETECTOR_DROP_RATE = 0.0
        self.AUX_LOSS_WEIGHT = 0.5

        self.USE_AUX_MODEL = True
        self.USE_SEG_GATE = False
        self.USE_CLINICAL_GATE = False
        self.CLINICAL_DIM = 2
        self.AUX_VALIDATE = True

        self.CLS_HEAD_TYPE = "avg_head"
        self.CLS_TAIL_TYPE = "linear"

        # PHASE CFG
        self.PHASE_CFG = {}

        # ------------------
        # 2. Load config.json
        # ------------------
        cfg_file = Path(config_file or self.DEFAULT_CONFIG_FILE)

        if not cfg_file.exists():
            cfg_file.parent.mkdir(parents=True, exist_ok=True)
            self.save_json(cfg_file)
            print(f"Default config created at: {cfg_file}")

        self.load_json(cfg_file)

        # ------------------
        # 3. Override by CLI arguments
        # ------------------
        if args is not None:
            self.apply_args(args)

        # ------------------
        # 4. Finalize derived paths
        # ------------------
        self._post_init()

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _post_init(self):
        """Recompute derived attributes & create dirs"""
        self.CSV_DIR_TRAIN = self.CSV_DIR / f"{self.SPLIT_PREFIX}_train.csv"
        self.CSV_DIR_VALID = self.CSV_DIR / f"{self.SPLIT_PREFIX}_valid.csv"
        self.EXPERIMENT_NAME = self.EXPERIMENT_NAME.split("_")[0] + f"_{self.SPLIT_PREFIX}_split"

        self.EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to JSON-serializable dict"""
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                out[k] = str(v)
            else:
                out[k] = v
        return out

    def load_json(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        for k, v in data.items():
            if k.startswith("__comment__"):
                continue
            setattr(self, k, Path(v) if self._is_path_key(k) else v)
        print(f"Loaded config from {path}")

    def save_json(self, path: str):
        """
        Save current config to JSON with grouped order and comments
        """
        cfg = self.to_dict()
        all_keys = set(cfg.keys())
        ordered = {}

        for group, keys in self.CONFIG_GROUPS.items():
            # Group comment
            ordered[f"__comment__{group}"] = group

            for k in keys:
                if k in cfg:
                    ordered[k] = cfg[k]
                    all_keys.remove(k)

        # Remaining keys
        if all_keys:
            ordered["__comment__Others"] = "Other configuration parameters"
            for k in sorted(all_keys):
                ordered[k] = cfg[k]

        with open(path, "w") as f:
            json.dump(ordered, f, indent=4)

        print(f"Config saved to {path}")


    def _parse_value(self, key: str, value: str):
        """
        Try to infer type from string.
        Raise ValueError with clear message if parsing fails.
        """
        try:
            # bool
            if value.lower() in ("true", "false"):
                return value.lower() == "true"

            # int
            try:
                return int(value)
            except ValueError:
                pass

            # float
            try:
                return float(value)
            except ValueError:
                pass

            # list / dict (JSON)
            try:
                return json.loads(value)
            except Exception:
                pass

            # string fallback
            return value

        except Exception as e:
            raise ValueError(f"Failed to parse --set {key}={value}: {e}")

    def apply_args(self, args: argparse.Namespace):
        """
        Apply CLI arguments to config.
        Ask for confirmation before saving changes to default config.
        """
        updates = {}
        errors = []
        force = getattr(args, "force", False)

        # 1. Explicit arguments
        for k in self.SEMANTIC_KEYS:
            v = getattr(args, k, None)
            if v is not None and hasattr(self, k):
                new_val = Path(v) if self._is_path_key(k) else v
                old_val = getattr(self, k)
                if old_val != new_val:
                    updates[k] = (old_val, new_val)

        # 2. Dynamic SET --set KEY=VALUE
        for item in args.set:
            if "=" not in item:
                errors.append(f"Invalid --set format: {item} (expected KEY=VALUE)")
                continue

            key, raw_value = item.split("=", 1)

            if not hasattr(self, key):
                errors.append(f"Unknown config key: {key}")
                continue

            try:
                new_val = self._parse_value(key, raw_value)
                if self._is_path_key(key):
                    new_val = Path(new_val)
            except Exception as e:
                errors.append(str(e))
                continue

            old_val = getattr(self, key)
            if old_val != new_val:
                updates[key] = (old_val, new_val)

        # 3. Dynamic UNSET --unset KEY
        for key in args.unset:
            if not hasattr(self, key):
                errors.append(f"Cannot unset unknown config key: {key}")
                continue

            old_val = getattr(self, key)
            updates[key] = (old_val, None)

        # 4. Breakpoint if errors
        if errors:
            print("\n[CONFIG PARSE ERRORS]")
            for e in errors:
                print(f"- {e}")

            if force:
                print("\n--force is enabled, terminating immediately due to errors.")
            else:
                input("\nPress ENTER to terminate and fix the errors...")

            raise SystemExit(1)
        
        # 5. EXPERIMENT_NAME special logic
        if "EXPERIMENT_NAME" in updates:
            reset_keys = self.REQUIRED_CONFIG_GROUPS["model_components"]
            for k in reset_keys:
                if hasattr(self, k):
                    updates[k] = (getattr(self, k), None)

        # 6. No changes → nothing to do
        if not updates:
            return

        # 7. Print diff
        print("\n[CONFIG CHANGES]")
        for k, (old, new) in updates.items():
            print(f"- {k}: {old}  →  {new}")

        # 8. Ask confirmation
        if not force:
            choice = input("\nApply these changes and save to default config? (Y/N): ").strip().lower()
            if choice not in ("y", "yes"):
                print("[INFO] Config changes discarded.")
                return
        else:
            print("[INFO] --force enabled, applying changes without confirmation.")


        # 9. Apply changes
        for k, (_, new) in updates.items():
            setattr(self, k, new)
        self._post_init()

        # 10. Save to default config
        self.save_json(self.DEFAULT_CONFIG_FILE)


    def validate_config(self) -> bool:
        missing = []

        for group, keys in self.REQUIRED_CONFIG_GROUPS.items():
            for k in keys:
                if not hasattr(self, k) or getattr(self, k) is None:
                    missing.append((group, k))

        if missing:
            print("\n[CONFIG VALIDATION FAILED]")
            for group, k in missing:
                print(f"- Missing [{group}] -> {k}")
            return False

        return True


    def print(self):
        cfg = self.to_dict()
        printed_keys = set()

        print("\n========== CONFIGURATION ==========")

        for group, keys in self.CONFIG_GROUPS.items():
            print(f"\n# {group}")
            for k in keys:
                if k in cfg:
                    print(f"{k:25s}: {cfg[k]}")
                    printed_keys.add(k)

        # In các key chưa được group
        remaining = sorted(set(cfg.keys()) - printed_keys)
        if remaining:
            print("\n# Others")
            for k in remaining:
                print(f"{k:25s}: {cfg[k]}")

        print("\n===================================\n")


    @classmethod
    def export_skeleton(cls, path: str):
        """
        Export JSON skeleton with ordered keys and group comments
        """
        cfg = cls()
        all_keys = set(cfg.to_dict().keys())
        skeleton = {}

        for group, keys in cls.CONFIG_GROUPS.items():
            # Add group comment
            skeleton[f"__comment__{group}"] = group

            for k in keys:
                if k in all_keys:
                    skeleton[k] = None
                    all_keys.remove(k)

        # Remaining keys
        if all_keys:
            skeleton["__comment__Others"] = "Other configuration parameters"
            for k in sorted(all_keys):
                skeleton[k] = None

        with open(path, "w") as f:
            json.dump(skeleton, f, indent=4)

        print(f"Skeleton exported to {path}")

    @staticmethod
    def _is_path_key(key: str) -> bool:
        return key.endswith("DIR") or key.endswith("FILE") or key.endswith("PATH") or key in {
            "WORKDIR", "RESOURCES", "MODEL_RGB_I3D", "DATADIR", "MASK_DATADIR", "CSV_DIR"
        }

def build_parser():
    parser = argparse.ArgumentParser("LUNA25 Training")

    # Config files
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config json"
    )

    # Main / semantic arguments (explicit)
    parser.add_argument("--EXPERIMENT_DIR", type=str)
    parser.add_argument("--EXPERIMENT_NAME", type=str)
    parser.add_argument("--SPLIT_PREFIX", type=str)
    parser.add_argument("--MODE", type=str)
    
    # Dynamic config override
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config value, e.g. --set BATCH_SIZE=16"
    )

    parser.add_argument(
        "--unset",
        action="append",
        default=[],
        metavar="KEY",
        help="Unset (reset) a config key"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Apply config changes without confirmation"
    )

    # Utilities
    parser.add_argument(
        "--export_config",
        type=str,
        help="Export final config to json"
    )

    parser.add_argument(
        "--export_skeleton",
        type=str,
        help="Export empty config skeleton"
    )

    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # Export skeleton rồi thoát
    if args.export_skeleton:
        Configuration.export_skeleton(args.export_skeleton)
        exit(0)

    # Load config
    config = Configuration(config_file=args.config, args=args)

    # Print config hiện tại
    config.print()

    # Export final config nếu cần
    if args.export_config:
        config.save_json(args.export_config)

# For importing
else:
    config = Configuration()

# Version 1
# from pathlib import Path


# class Configuration(object):
#     def __init__(self) -> None:

#         # Working directory
#         self.WORKDIR = Path("/data/luna25/model/luna25-baseline-public")
#         self.RESOURCES = self.WORKDIR / "resources"
#         # Starting weights for the I3D model
#         self.MODEL_RGB_I3D = (
#             self.RESOURCES / "model_rgb.pth"
#         )
        
#         # Data parameters
#         # Path to the nodule blocks folder provided for the LUNA25 training data. 
#         self.DATADIR = Path("/data/luna25/data/luna25_nodule_blocks")
#         self.MASK_DATADIR = Path("/data/luna25/data/luna25_nodule_blocks_mask")

#         # Path to the folder containing the CSVs for training and validation.
#         self.CSV_DIR = Path("/data/luna25/data/dataset_csv")
#         # We provide an NLST dataset CSV, but participants are responsible for splitting the data into training and validation sets.
#         self.SPLIT_PREFIX = "normal"    # normal split

#         self.CSV_DIR_TRAIN = self.CSV_DIR / f"{self.SPLIT_PREFIX}_train.csv" # Path to the training CSV
#         self.CSV_DIR_VALID = self.CSV_DIR / f"{self.SPLIT_PREFIX}_valid.csv" # Path to the validation CSV

#         # Results will be saved in the /results/ directory, inside a subfolder named according to the specified EXPERIMENT_NAME and MODE.
#         self.EXPERIMENT_DIR = self.WORKDIR / "results"
#         if not self.EXPERIMENT_DIR.exists():
#             self.EXPERIMENT_DIR.mkdir(parents=True)
            
#         self.EXPERIMENT_NAME = f"LUNA25-baseline_{self.SPLIT_PREFIX}_split"
#         self.MODE = "3D" # 2D or 3D

#         # Training parameters
#         self.SEED = 2025
#         self.NUM_WORKERS = 8
#         self.SIZE_MM = 50
#         self.SIZE_PX = 64
#         self.BATCH_SIZE = 32
#         self.ROTATION = ((-20, 20), (-20, 20), (-20, 20))
#         self.TRANSLATION = True
#         self.EPOCHS = 10
#         self.PATIENCE = 20
#         self.PATCH_SIZE = [64, 128, 128]
#         self.LEARNING_RATE = 1e-4
#         self.WEIGHT_DECAY = 5e-4


# config = Configuration()