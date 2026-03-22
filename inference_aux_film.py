import argparse
import logging
import shutil
import sys

import time

from typing import Dict, Tuple

from pathlib import Path
import json
from glob import glob
import SimpleITK
import numpy as np
from scipy.special import logit
import joblib

from processor_aux_film import MalignancyProcessor, MalignancyDetector, CancerPredictor, process_luna25_zip, process_raw_clinical, visualize_nodules


# suppress warnings from torch about inference mode and other deprecations
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger("processor_aux_film").setLevel(logging.INFO)


# Define paths
TEST_ROOT = Path("./test")
INPUT_PATH = TEST_ROOT / "input"
OUTPUT_PATH = TEST_ROOT / "output"
RESOURCE_PATH = Path("./results")


# Running Inference helper functions
def transform(input_image, point):
    """

    Parameters
    ----------
    input_image: SimpleITK Image
    point: array of points

    Returns
    -------
    tNumpyOrigin

    """
    return np.array(
        list(
            reversed(
                input_image.TransformContinuousIndexToPhysicalPoint(
                    list(reversed(point))
                )
            )
        )
    )

def itk_image_to_numpy_image(input_image):
    """

    Parameters
    ----------
    input_image: SimpleITK image

    Returns
    -------
    numpyImage: SimpleITK image to numpy image
    header: dict containing origin, spacing and transform in numpy format

    """

    numpyImage = SimpleITK.GetArrayFromImage(input_image)
    numpyOrigin = np.array(list(reversed(input_image.GetOrigin())))
    numpySpacing = np.array(list(reversed(input_image.GetSpacing())))

    # get numpyTransform
    tNumpyOrigin = transform(input_image, np.zeros((numpyImage.ndim,)))
    tNumpyMatrixComponents = [None] * numpyImage.ndim
    for i in range(numpyImage.ndim):
        v = [0] * numpyImage.ndim
        v[i] = 1
        tNumpyMatrixComponents[i] = transform(input_image, v) - tNumpyOrigin
    numpyTransform = np.vstack(tNumpyMatrixComponents).dot(np.diag(1 / numpySpacing))

    # define necessary image metadata in header
    header = {
        "origin": numpyOrigin,
        "spacing": numpySpacing,
        "transform": numpyTransform,
    }

    return numpyImage, header

class NoduleProcessor:
    def __init__(self, ct_image_file, nodule_locations, clinical_information, model_name="LUNA25-baseline-2D", device="cuda"):
        """
        Parameters
        ----------
        ct_image_file: Path to the CT image file
        nodule_locations: Dictionary containing nodule coordinates and annotationIDs
        clinical_information: Dictionary containing clinical information (Age and Gender)
        mode: 2D or 3D
        model_name: Name of the model to be used for prediction
        """
        self._image_file = ct_image_file
        self.nodule_locations = nodule_locations
        self.clinical_information = clinical_information
        self.model_name = model_name
        self.device = device

        self.processor = MalignancyProcessor(suppress_logs=True, model_name=model_name, model_dir=RESOURCE_PATH, device=self.device)


    def predict(self, input_image: SimpleITK.Image, coords: np.array) -> Dict:
        """

        Parameters
        ----------
        input_image: SimpleITK Image
        coords: numpy array with list of nodule coordinates in /input/nodule-locations.json

        Returns
        -------
        malignancy risk of the nodules provided in /input/nodule-locations.json
        """

        numpyImage, header = itk_image_to_numpy_image(input_image)

        malignancy_risks = []
        for i in range(len(coords)):
            self.processor.define_inputs(
                numpyImage, 
                self.clinical_information['age'],
                self.clinical_information['gender'],
                header, 
                [coords[i]]
            )
            malignancy_risk, logits = self.processor.predict()
            malignancy_risk = np.array(malignancy_risk).reshape(-1)[0]
            malignancy_risks.append(malignancy_risk)

        malignancy_risks = np.array(malignancy_risks)
        malignancy_risks = list(malignancy_risks)

        return malignancy_risks

    def load_inputs(self):
        # load image
        print(f"Reading {self._image_file}")
        image = SimpleITK.ReadImage(str(self._image_file))

        self.annotationIDs = [p["name"] for p in self.nodule_locations["points"]]
        self.coords = np.array([p["point"] for p in self.nodule_locations["points"]])
        self.coords = np.flip(self.coords, axis=1)  # reverse to [z, y, x] format

        return image, self.coords, self.annotationIDs

    def process(self):
        """
        Load CT scan(s) and nodule coordinates, predict malignancy risk and write the outputs
        Returns
        -------
        None
        """
        image, coords, annotationIDs = self.load_inputs()
        output = self.predict(image, coords)

        assert len(output) == len(annotationIDs), "Number of outputs should match number of inputs"
        results = {
            "name": "Points of interest",
            "type": "Multiple points",
            "points": [],
            "version": {
                "major": 1,
                "minor": 0
            }
        }

        # Populate the "points" section dynamically
        coords = np.flip(coords, axis=1)
        for i in range(len(annotationIDs)):
            results["points"].append(
                    {
                    "name": annotationIDs[i],
                    "point": coords[i].tolist(),
                    "probability": float(output[i])
                    }
                )
        return results


# Helper functions for file I/O and GPU info
def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())

def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))

def load_image_path(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )

    assert (
                len(input_files) == 1
            ), "Please upload only one .mha file per job for grand-challenge.org"
    
    result = input_files[0]

    return result

def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch version: {torch.version.cuda}")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


# Main functions for running inference, validation and calibration
def run(model_name="LUNA25-baseline-2D", device="cuda"):

    print(f"[INFERENCE]")

    # Read the inputs
    input_nodule_locations = load_json_file(
        location=INPUT_PATH / "nodule-locations.json",
    )
    input_clinical_information = load_json_file(
        location=INPUT_PATH / "clinical-information-lung-ct.json",
    )
    input_chest_ct = load_image_path(
        location=INPUT_PATH / "images/chest-ct",
    )
    
    # Validate access to GPU
    _show_torch_cuda_info()
    
    # Run your algorithm here
    processor = NoduleProcessor(ct_image_file=input_chest_ct,
                                nodule_locations=input_nodule_locations,
                                clinical_information=input_clinical_information,
                                model_name=model_name,
                                device=device)
    malignancy_risks = processor.process()

    # Save your output
    write_json_file(
        location=OUTPUT_PATH / "lung-nodule-malginancy-likelihoods.json",
        content=malignancy_risks,
    )
    print(f"Completed writing output to {OUTPUT_PATH}")
    print(f"Output: {malignancy_risks}") 
    return 0


def valid(model_name="LUNA25-baseline-2D", device="cuda"):

    print(f"[VALIDATION]")
    print(f"- model: {model_name}")

    # Validate access to GPU
    _show_torch_cuda_info()

    processor = MalignancyProcessor(suppress_logs=True, model_name=model_name, model_dir=RESOURCE_PATH, device=device)
    valid_loader = processor.init_valid()
    metrics_dict = processor.validate(valid_loader)
    print(metrics_dict)
    return 0


def calib(model_name="LUNA25-baseline-2D", device="cuda"):
    print(f"[CALIBRATION]")
    print(f"- model: {model_name}")

    # Validate access to GPU
    _show_torch_cuda_info()

    processor = MalignancyProcessor(suppress_logs=True, model_name=model_name, model_dir=RESOURCE_PATH, device=device)
    valid_loader = processor.init_valid()
    temperature = processor.calibrate(valid_loader)
    print(temperature)
    return 0


THRESHOLD = 0.5
def private_run(zip_root, model_name="LUNA25-baseline-2D", device="cuda"):
    # Initialize paths
    zip_root = Path(zip_root)
    temp_extract = zip_root / "temp_extract"
    input_dir = zip_root / "input"
    output_dir = zip_root / "output"

    # Initialize models
    detector = MalignancyDetector(
        model_path="./resources/dt_model.ts",  
        device=device
    )

    processor = MalignancyProcessor(
        suppress_logs=True, 
        model_name=model_name, 
        model_dir=RESOURCE_PATH, 
        device=device
    )

    predictor = CancerPredictor(
        threshold=THRESHOLD,
        method="max-rule"
    )

    cancer_predictions = []

    print("[PRIVATE RUN]")
    print(f"- model: {model_name}")

    # Validate access to GPU
    _show_torch_cuda_info()

    # For each zip file in the input directory
    for zip_filepath in zip_root.glob("*.zip"):
        output_results = []
        nodule_probs = []
        zip_name = zip_filepath.stem
        print(f"Processing zip file {zip_name}...")

        # 1 - Preprocess inputs
        if (input_dir / zip_name).exists():
            print(f" > Input for {zip_name} already exists, skipping extraction.")
        else:
            process_luna25_zip(
                zip_filepath, 
                extract_to=temp_extract,
                output_dir=input_dir / zip_name
            )

        print()

        # read metadata.json
        with open(input_dir / zip_name / "metadata.json", "r") as f:
            metadata = json.load(f)

        # 2 - Malignancy detection
        # using MalignancyDetector class
        for s_uid in metadata.keys():
            filename = metadata[s_uid]['File']

            if "NoduleLocations" in metadata[s_uid]:
                print(f" > Nodule locations already exist for series_UID {s_uid}, skipping detection.")
                continue

            print(f" + Detecting nodules in series_UID {s_uid}...")
            boxes = detector.detect(input_dir / zip_name / filename)

            print("  - Updating metadata with nodule locations...")
            metadata[s_uid]["NoduleLocations"] = boxes

            print("  - Updating metadata with transformed clinical information...")
            sex, age = process_raw_clinical(metadata, s_uid)
            metadata[s_uid]["PatientSex"] = sex
            metadata[s_uid]["PatientAge"] = age

        # save metadata.json
        with open(input_dir / zip_name / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        print()

        # Malignancy prediction
        # using MaligancyProcessor class
        total_runtime = 0
        for s_uid in metadata.keys():
            filename = metadata[s_uid]['File']

            # read image + clinical information
            image_file = metadata[s_uid]['File']
            input_image = SimpleITK.ReadImage(str(input_dir / zip_name / image_file))
            numpyImage, header = itk_image_to_numpy_image(input_image)

            patient_gender = metadata[s_uid]['PatientSex']        
            patient_age = metadata[s_uid]['PatientAge']

            print(f" + Predicting malignancy for nodules in series_UID {s_uid}...")

            for nodule_loc in metadata[s_uid]["NoduleLocations"]:
                # nodule_loc format = [cx cy cz w h d]
                # coord format = [z, y, x]
                x, y, z = nodule_loc[:3]
                coord = [z, y, x]

                # Timestamp for meansuring runtime
                start_time = time.time()

                # Predicing malignancy
                print(f"  - Processing nodule at coord {coord}...")
                processor.define_inputs(
                    numpyImage, 
                    patient_age,
                    patient_gender,
                    header,
                    [coord]
                )
                malignancy_risk, logits = processor.predict()
                malignancy_risk = np.array(malignancy_risk).reshape(-1)[0]
                malignancy_label = int(malignancy_risk >= THRESHOLD)

                # Calculate runtime
                run_time = time.time() - start_time

                # Append results to output list
                output_results.append({
                    "seriesInstanceUID": s_uid,
                    "probability": float(malignancy_risk),
                    "predictionLabel": malignancy_label,
                    "processingTimeMs": int(run_time * 1000),
                    "CoordX": round(coord[2], 2),
                    "CoordY": round(coord[1], 2),
                    "CoordZ": round(coord[0], 2)
                })

                # Append malignancy risk to nodule_probs for cancer prediction
                nodule_probs.append(malignancy_risk)

                total_runtime += run_time

        print()

        # Cancer prediction
        # using CancerPredictor class
        cancer_prediction = predictor.predict(nodule_probs)
        cancer_predictions.append({
            "seriesInstanceUID": s_uid,
            "cancerPrediction": int(cancer_prediction),
            "processingTimeMs": int(total_runtime * 1000)
        })
        print(f" > Overall cancer prediction for patient in {zip_name}: {cancer_prediction}")

        print()

        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filepath = output_dir / f"{zip_name}.json"
        with open(output_filepath, "w") as f:
            json.dump(output_results, f, indent=4)
        print(f" > Saved output to {output_filepath}\n")

        # Visualize nodules and save visualization
        visualize_nodules(
            metadata,
            output_results,
            zip_root,
            zip_name
        )

        # # debug break
        # break

    # Save overall cancer predictions for all patients
    cancer_output_filepath = output_dir / f"cancer_predictions.json"
    with open(cancer_output_filepath, "w") as f:
        json.dump(cancer_predictions, f, indent=4)
    print(f" > Saved overall cancer predictions to {cancer_output_filepath}\n")

    print()

    # Ceanup temporary files
    if temp_extract.exists():
        shutil.rmtree(temp_extract)

    # if input_dir.exists():
    #     shutil.rmtree(input_dir)

    # if output_dir.exists():
    #     shutil.rmtree(output_dir)

    return 0


# Argument parser and main execution logic
def build_parser():
    parser = argparse.ArgumentParser(
        description="LUNA25 inference / validation runner"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model checkpoint name",
    )

    parser.add_argument(
        "--device",
        type=str,
        help="Torch device cuda / cpu (default: cuda)",
        default="cuda"
    )

    parser.add_argument(
        "--zip_root",
        type=str,
        help="Path to the root directory containing zip files for private test (default: ./private-test/MTN/)",
        default="./private-test/MTN/"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run",
        action="store_true",
        help="Run inference",
    )
    group.add_argument(
        "--valid",
        action="store_true",
        help="Run validation",
    )
    group.add_argument(
        "--calib",
        action="store_true",
        help="Run calibration",
    )

    group.add_argument(
        "--private_run",
        action="store_true",
        help="Run private test with zip file input",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.run:
        sys.exit(
            run(
                model_name=args.model_name,
                device=args.device
            )
        )

    if args.valid:
        sys.exit(
            valid(
                model_name=args.model_name,
                device=args.device
            )
        )

    if args.calib:
        sys.exit(
            calib(
                model_name=args.model_name,
                device=args.device
            )
        )

    if args.private_run:
        sys.exit(
            private_run(
                zip_root=args.zip_root,
                model_name=args.model_name,
                device=args.device
            )
        )