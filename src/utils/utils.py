import os
import random
import hashlib

import numpy as np


def init_seed():
    np.random.seed(0)
    random.seed(0)


def project_path():
    return os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "..",
        ".."
    )

def model_dir(model_name):
    return os.path.join(
        project_path(),
        "models",
        model_name
    )

def auto_increment_run_suffix(name: str, pad=3):
    suffix = name.split("-")[-1]
    next_suffix = str(int(suffix) + 1).zfill(pad)
    return name.replace(suffix, next_suffix)

def calculate_hash(filename):
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_hash(dst):
    hash_ = calculate_hash(dst)
    dst, _ = os.path.splitext(dst)
    with open(f"{dst}.sha256", "w") as f:
        f.write(hash_)


def read_hash(dst):
    dst, _ = os.path.splitext(dst)
    with open(f"{dst}.sha256", "r") as f:
        return f.read()