"""
Utility functions for managing eye‑tracking datasets.

This module provides helpers to:
- Build filtered file lists from a directory tree.
- Derive participant IDs from exported CSV filenames.
- Download raw data from the OSF project (https://osf.io/zx7hc/).
- Invoke the R-based cleaning pipeline to generate tidy CSVs.

All functions are side‑effect free unless explicitly noted (e.g., `download_osf_data`,
`clean_raw_data`). No executable logic is changed by these docstrings.
"""
from importlib.resources import files
import os
from osfclient import OSF
from pathlib import Path
import subprocess
from tqdm import tqdm
from typing import List

def create_file_list(path: str, indicators_pos: list = [], indicators_neg: list = [], sort: callable = None):
    """
    Get all files under a directory that match positive/negative string indicators.

    Args:
        path (str): Root directory to walk recursively.
        indicators_pos (list, optional): All strings that **must** appear in the filename
            for it to be included. Defaults to [].
        indicators_neg (list, optional): All strings that **must not** appear in the
            filename for it to be included. Defaults to [].
        sort (callable, optional): Reserved for custom sorting; currently unused.

    Returns:
        list: Absolute (or joined) file paths that satisfy the indicator filters.
    """
                
    # create condition based on all indicators
    condition_pos = lambda x: all([indicator in x for indicator in indicators_pos])
    condition_neg = lambda x: all([indicator not in x for indicator in indicators_neg])
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if condition_pos(file) and condition_neg(file):
                file_list.append(os.path.join(root, file))
    return file_list

def get_participant_ids(data_path: str = "data/raw") -> list:
    """
    Return unique participant IDs inferred from raw CSV filenames.

    The function searches `data_path` for `*.csv` files, then extracts the leading
    prefix before the first underscore (e.g., `06b8d2d3_targets.csv` → `06b8d2d3`).

    Args:
        data_path (str, optional): Directory containing raw CSV exports. Defaults to
            "data/raw".

    Returns:
        list: Unique participant IDs (order not guaranteed).
    """
    available_files = create_file_list(data_path, indicators_pos=[".csv"])
    participant_ids = list({
        Path(file).name.split("_")[0] for file in available_files
    })
    return participant_ids


def download_osf_data(out_dir: str = "data/raw",
                      train_test: str = "both",
                      participants: str | List[str] = "all",
                      overwrite: bool = False,
                      username: str = None,
                      password: str = None,
                      token: str = None):
    """
    Download raw eye‑movement data from the public OSF project.

    This function connects to the OSF project (ID: `zx7hc`) and streams files to
    `out_dir`, optionally filtering by split (`train`, `test`, or `both`) and by a
    subset of participants. Existing files are skipped unless `overwrite=True`.

    Authentication is optional; if `username`, `password`, and `token` are provided,
    `osfclient` will perform an authenticated session.

    Args:
        out_dir (str, optional): Local output root for the downloaded tree. Defaults to
            "data/raw".
        train_test (str, optional): Which dataset split to fetch: one of {"train",
            "test", "both"}. Defaults to "both".
        participants (str | List[str], optional): "all" to download every participant
            or a list of participant IDs (e.g., ["06b8d2d3", "7d248f8f"]). Defaults to "all".
        overwrite (bool, optional): If False, skip files that already exist locally.
            Defaults to False.
        username (str, optional): OSF username for authenticated access. Defaults to None.
        password (str, optional): OSF password for authenticated access. Defaults to None.
        token (str, optional): OSF personal access token. Defaults to None.

    Raises:
        AssertionError: If `train_test` or `participants` are invalid.

    Returns:
        None: Files are written to disk; progress is printed to stdout.
    """
    # Validate inputs
    assert train_test in ["train", "test", "both"], "train_test must be 'train', 'test' or 'both'"
    assert participants == "all" or isinstance(participants, list), "participants must be 'all' or a list of participant ids"

    # Get OSF project
    osf = OSF()
    if username and password and token:
        osf.login(
            username=username,
            password=password,
            token=token
        )
    project = osf.project("zx7hc") # hard coded project id https://osf.io/zx7hc/

    # Download all files
    print(f"Downloading data from OSF to {out_dir}.")
    for storage in project.storages:
        for file in tqdm(storage.files, total=len(list(storage.files))):

            # Get paths
            file_dir = "/".join(file.path.split('/')[:-1])
            out_path = f"{out_dir}{file_dir}"
            
            # Skip if file exists and not overwrite
            if not overwrite and os.path.exists(f"{out_path}/{file.name}"):
                continue

            # Skip train or test if specified
            if train_test == "train" and "test" in file_dir:
                continue
            if train_test == "test" and "train" in file_dir:
                continue

            # Skip if participant is not in list
            if participants != "all" and file.name.split(".")[0].split("_")[0] not in participants:
                continue

            # Create output directory
            os.makedirs(out_path, exist_ok=True)

            # Download file
            with open(f"{out_path}/{file.name}", "wb") as f:
                file.write_to(f)

def clean_raw_data(r_exe_path: str = "Rscript",):
    """
    Run the R cleaning pipeline (`clean_data.R`) to generate tidy CSVs.

    This calls the Rscript executable (`r_exe_path`) with the packaged `clean_data.R`
    script. The R script reads from `data/raw/` and writes cleaned outputs to
    `data/clean/` (see comments in `clean_data.R`).

    Args:
        r_exe_path (str, optional): Path or name of the Rscript binary. Defaults to
            "Rscript" (must be discoverable on PATH).

    Returns:
        None: Side effect is running the external R process and writing files.
    """
    r_script_path = files("eyemovement_data").joinpath("clean_data.R")
    print(f"Running {r_script_path} to clean raw data.")
    subprocess.run([r_exe_path, str(r_script_path)], check=True)