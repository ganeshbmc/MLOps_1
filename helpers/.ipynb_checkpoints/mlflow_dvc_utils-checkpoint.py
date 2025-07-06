import subprocess
import os
import yaml

def get_git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"


def get_dvc_md5_hash_from_lock(target_path, lock_file="dvc.lock"):
    """
    Extracts the MD5 hash of a file tracked as an output in a DVC pipeline (via dvc.lock).
    Returns 'unknown' if not found.
    """
    if not os.path.exists(lock_file):
        return "unknown"

    try:
        with open(lock_file, "r") as f:
            lock_data = yaml.safe_load(f)

        for stage in lock_data.get("stages", {}).values():
            for out in stage.get("outs", []):
                if out.get("path") == target_path:
                    return out.get("md5", "unknown")
    except Exception:
        return "unknown"

    return "unknown"

