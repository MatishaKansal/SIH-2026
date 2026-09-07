"""Background runner that monitors indic-audio download completion and then executes the full-pipeline training on all 3 datasets.

Logs all progress to Module 2B/output/full_training.log.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_FILE = ROOT / "Module 2B" / "output" / "full_training.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("BackgroundRunner")


def wait_for_indic_audio_download(target_count=5000, timeout_seconds=3600):
    logger.info("Monitoring indic-audio audio downloads in data/syntheticvoice/indic-audio/audio...")
    audio_dir = ROOT / "data" / "syntheticvoice" / "indic-audio" / "audio"
    start = time.time()
    last_count = 0

    while time.time() - start < timeout_seconds:
        if audio_dir.exists():
            count = sum(1 for _ in audio_dir.rglob("*.*") if _.is_file())
        else:
            count = 0

        if count != last_count:
            logger.info("Downloaded %d / ~%d files (%.1f%%)", count, target_count, (count / target_count) * 100)
            last_count = count

        if count >= target_count:
            logger.info("indic-audio download complete (%d files). Proceeding to training!", count)
            return True

        time.sleep(15)

    logger.warning("Download wait timed out after %ds (%d files downloaded). Proceeding with available files.", timeout_seconds, last_count)
    return False


def main():
    logger.info("==================================================================")
    logger.info("Step 1: Awaiting completion of indic-audio dataset download")
    logger.info("==================================================================")
    wait_for_indic_audio_download(target_count=5150, timeout_seconds=3600)

    logger.info("==================================================================")
    logger.info("Step 2: Launching full 3-dataset pipeline training")
    logger.info("==================================================================")
    orchestrator = ROOT / "Module 2B" / "training" / "run_full_pipeline.py"
    res = subprocess.run([sys.executable, str(orchestrator)], cwd=str(ROOT), capture_output=True, text=True)

    if res.returncode == 0:
        logger.info("Full 3-dataset training pipeline completed successfully!")
        logger.info(res.stdout)
    else:
        logger.error("Full 3-dataset pipeline training encountered an error!")
        logger.error("STDOUT:\n%s", res.stdout[-2000:])
        logger.error("STDERR:\n%s", res.stderr[-2000:])


if __name__ == "__main__":
    main()
