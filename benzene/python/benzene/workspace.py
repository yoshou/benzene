"""Where the models keep their data: work/ in the repository (not tracked).

work/<model>  one directory per model (models/<model>/paths.py): checkpoint, prepared data, evaluation outputs, exported
              model files, device input bundles, logs
work/datasets the raw datasets the models share (scripts/download-inputs.sh)
"""

from __future__ import annotations

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
WORK = REPOSITORY / "work"
DATASETS = WORK / "datasets"
TUM_RGBD = DATASETS / "tum_rgbd"  # TUM RGB-D sequences (rgbd_dataset_*)
MIPNERF360 = DATASETS / "mipnerf360"  # Mip-NeRF 360 scenes (images_8, COLMAP sparse, poses_bounds.npy)
CV_BENCH = DATASETS / "cv_bench"  # nyu-visionx/CV-Bench
REALWORLDQA = DATASETS / "realworldqa"  # lmms-lab/RealWorldQA


def board_directory(model: str):
    """The __getattr__ of a model's paths module: BOARD_DIR, the model's directory under the board user's home, read
    from the board settings (benzene.board) only when used."""

    def __getattr__(name: str) -> str:
        if name == "BOARD_DIR":
            from benzene import board

            return f"{board.HOME}/{model}"
        raise AttributeError(name)

    return __getattr__
