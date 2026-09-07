"""Automated background orchestrator for full-dataset training of Module 2B.

Executes:
1. Re-builds target_domain_manifest.csv to incorporate all 3 datasets:
   - Svarah (real voice)
   - Orpheus TTS (synthetic voice)
   - indic-audio (synthetic voice)
2. Batch feature extraction across all splits using ProcessPoolExecutor
3. Multi-GBDT model training (LightGBM, XGBoost, CatBoost) & validation comparison
4. Platt probability calibration on held-out calibration split
5. Unbiased evaluation on untouched test split
6. SHAP explainability analysis & feature importance plotting
7. Unit test suite execution
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_FILE = ROOT / "Module 2B" / "output" / "full_pipeline.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("FullPipeline")
PYTHON = sys.executable


def run_step(cmd: list[str], description: str) -> None:
    logger.info(">>> Starting: %s", description)
    start = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    duration = time.time() - start
    if result.returncode != 0:
        logger.error("FAILED: %s (after %.1fs)", description, duration)
        logger.error("STDOUT:\n%s", result.stdout[-2000:])
        logger.error("STDERR:\n%s", result.stderr[-2000:])
        raise RuntimeError(f"Step '{description}' failed with return code {result.returncode}")
    logger.info("COMPLETED: %s (took %.1fs)", description, duration)
    if result.stdout.strip():
        lines = result.stdout.strip().splitlines()
        for line in lines[-5:]:
            logger.info("  | %s", line)


def main():
    logger.info("=================================================================")
    logger.info("Module 2B Full Pipeline Background Training Initiated")
    logger.info("Project Root: %s", ROOT)
    logger.info("Python Binary: %s", PYTHON)
    logger.info("=================================================================")

    # 1. Update manifest with all available datasets
    manifest_script = ROOT / "Module 2A" / "build_dataset_manifest.py"
    manifest_path = ROOT / "Module 2A" / "output" / "datasets" / "target_domain_manifest.csv"
    if manifest_script.exists():
        run_step(
            [PYTHON, str(manifest_script), "--output", str(manifest_path)],
            "Re-building target_domain_manifest.csv with all 3 datasets",
        )

    # 2. Batch feature extraction
    prepare_script = ROOT / "Module 2B" / "training" / "prepare_features.py"
    run_step(
        [
            PYTHON,
            str(prepare_script),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            "output/features",
            "--workers",
            "4",
        ],
        "Extracting 56 acoustic features on full dataset",
    )

    # 3. Model benchmarking and selection
    compare_script = ROOT / "Module 2B" / "training" / "compare_models.py"
    run_step(
        [PYTHON, str(compare_script)],
        "Training and benchmarking LightGBM, XGBoost, and CatBoost",
    )

    # 4. Platt calibration
    calibrate_script = ROOT / "Module 2B" / "calibrate.py"
    run_step(
        [PYTHON, str(calibrate_script)],
        "Fitting Platt scaling calibrator on calibration split",
    )

    # 5. Held-out test evaluation
    evaluate_script = ROOT / "Module 2B" / "training" / "evaluate.py"
    run_step(
        [PYTHON, str(evaluate_script)],
        "Evaluating winning model on untouched test split",
    )

    # 6. SHAP explainability
    explain_script = ROOT / "Module 2B" / "training" / "explainability.py"
    run_step(
        [PYTHON, str(explain_script)],
        "Generating SHAP beeswarm and feature importance plots",
    )

    # 7. Unit tests
    run_step(
        [PYTHON, "-m", "unittest", "discover", "-s", "Module 2B/tests", "-v"],
        "Running complete unit test suite",
    )

    logger.info("=================================================================")
    logger.info("Module 2B Full Pipeline Background Training Successfully Finished!")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
