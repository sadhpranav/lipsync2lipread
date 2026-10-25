"""Speaker-independent GRID splits and processed-data paths."""
import os

TEST = ["s1", "s2", "s20", "s22"]   # LipNet unseen-speaker split
VAL = ["s13"]
ALL = [f"s{i}" for i in range(1, 35) if i != 21]  # s21 has no video
TRAIN_ALL = [s for s in ALL if s not in VAL + TEST]
VAL_CLIPS = 300


def work_dir():
    return os.environ.get("L2L_WORK", os.path.join(os.path.dirname(__file__), "..", "..", "work"))


def proc_dir(spk):
    return os.path.join(work_dir(), "proc", spk)


def ready(spk):
    """eface.npy is written last (teacher/precompute.py), so it marks a complete speaker."""
    return os.path.exists(os.path.join(proc_dir(spk), "eface.npy"))


def available_train():
    """Every fully processed speaker that is not val/test, discovered from WORK/proc."""
    root = os.path.join(work_dir(), "proc")
    spks = [s for s in os.listdir(root) if s not in VAL + TEST and ready(s)] if os.path.isdir(root) else []
    return sorted(spks, key=lambda s: int(s[1:]))
