from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple

from eyemovement_data.preprocessor import Preprocessor, OriginalPreprocessor
from eyemovement_data.data_quality_preprocessor import DataQualityPreprocessor
from eyemovement_data.classifier import Classifier, OriginalClassifier
from eyemovement_data.utils import create_file_list

class Participant:
    """
    Represents a single participant and orchestration for preprocessing, classification,
    quality checks, and plotting.

    This class wraps the full pipeline for a participant identified by an ID like
    `06b8d2d3`. It provides helpers to locate data files, load raw/clean/preprocessed
    CSV tables, run preprocessing and classification, and compute common quality
    metrics and visualizations.
    """
    def __init__(self, 
                 id, 
                 preprocessor: Preprocessor = OriginalPreprocessor(), 
                 classifier: Classifier = OriginalClassifier()):
        """
        Initialize a `Participant` object.

        Args:
            id (str): Participant identifier (e.g., `06b8d2d3`).
            preprocessor (Preprocessor, optional): Preprocessor instance used by
                `preprocess_clean_data`. Defaults to `OriginalPreprocessor()`.
            classifier (Classifier, optional): Classifier instance used by
                `classify_preprocessed_data`. Defaults to `OriginalClassifier()`.

        Attributes:
            subset (str): Inferred dataset split ("train"|"test"|"unknown_subset").
            raw_data (dict): Loaded raw CSVs keyed by type (e.g., "gaze", "trials").
            clean_data (dict): Loaded cleaned CSVs.
            preprocessed_data (dict): Outputs of preprocessing.
            classified_data (dict): Outputs of classification.
        """
        self.id = id
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.subset = "unknown_subset"
        self.raw_data = dict()
        self.clean_data = dict()
        self.preprocessed_data = dict()
        self.classified_data = dict()

        self.identify_subset()
    
    def identify_subset(self, data_path: str = "data") -> str:
        """
        Infer whether this participant belongs to the train or test split.

        Args:
            data_path (str, optional): Root path that contains split folders (e.g.,
                `data/train`, `data/test`). Defaults to "data".

        Returns:
            str: The inferred subset name assigned to `self.subset` ("train",
                "test", or "unknown_subset").

        Notes:
            Uses a filename search containing the participant ID and `.csv`. If neither
            `train` nor `test` are present in the first match, a warning is printed and
            the subset remains `"unknown_subset"`.
        """
        available_files = create_file_list(data_path, indicators_pos=[self.id, ".csv"])
        if "train" in available_files[0]:
            self.subset = "train"
        elif "test" in available_files[0]:
            self.subset = "test"
        else:
            print(f"WARNING! Could not identify subset for participant {self.id}.")
            self.subset = "unknown_subset"

    def load_data(self, data_path: str) -> dict():
        """
        Load all CSVs for this participant from a directory tree.

        Args:
            data_path (str): Directory that contains the participant's CSV exports for
                a split, e.g., `data/raw`, `data/clean`, or similar.

        Returns:
            dict[str, pd.DataFrame]: Mapping from data type ("gaze", "targets",
                "trials", "blinks", etc.) to loaded DataFrames.

        Warnings:
            Prints a warning if no files were found for this participant.
        """
        available_files = create_file_list(data_path, indicators_pos=[self.id, ".csv"])

        if len(available_files) == 0:
            print(f"WARNING! No data found for participant {self.id} under {data_path}.")

        data = dict()
        for file in available_files:
            data_type = file.split("/")[-1].split("_")[1].split(".")[0]
            data[data_type] = pd.read_csv(file)
        
        return data
    
    def set_raw_data(self, data_path: str = "data/raw") -> None:
        """
        Populate `self.raw_data` by reading this participant's raw CSVs.

        Args:
            data_path (str, optional): Path to the raw data root (default: "data/raw").

        Returns:
            None
        """
        self.raw_data = self.load_data(data_path)
    
    def set_clean_data(self, data_path: str = "data/clean") -> None:
        """
        Populate `self.clean_data` by reading this participant's cleaned CSVs.

        Args:
            data_path (str, optional): Path to the cleaned data root (default: "data/clean").

        Returns:
            None
        """
        self.clean_data = self.load_data(data_path)
    
    def set_preprocessed_data(self, data_path: str = "data/preprocessed") -> None:
        """
        Populate `self.preprocessed_data` by reading this participant's preprocessed CSVs.

        Args:
            data_path (str, optional): Path to the preprocessed data root (default:
                "data/preprocessed").

        Returns:
            None
        """
        self.preprocessed_data = self.load_data(data_path)
    
    def set_classified_data(self, data_path: str = "data/classified") -> None:
        """
        Populate `self.classified_data` by reading this participant's classified CSVs.

        Args:
            data_path (str, optional): Path to the classified data root (default:
                "data/classified").

        Returns:
            None
        """
        self.classified_data = self.load_data(data_path)
    
    def save_data(self, out_path: str = "data/preprocessed", what: str = "preprocessed") -> None:
        """
        Write one of the participant's data dictionaries to disk.

        Args:
            out_path (str, optional): Output root directory (e.g., `data/preprocessed`).
            what (str, optional): Which dataset to save: one of {"raw", "clean",
                "preprocessed", "classified"}. Defaults to "preprocessed".

        Returns:
            None

        Raises:
            AssertionError: If `what` is not a supported value or if the corresponding
                data dictionary is empty.

        Side Effects:
            Creates directories at `{out_path}/{self.subset}/{self.id}` and writes CSVs
            named `{self.id}_{key}.csv` for each entry in the chosen dictionary.
        """
        assert what in ["raw", "clean", "preprocessed", "classified"], "what must be 'raw', 'clean', or 'preprocessed'"

        # Get the data
        if what == "raw":
            assert len(self.raw_data) > 0, "No raw data available. Load data with set_raw_data() first."
            data = self.raw_data.copy()
        elif what == "clean":
            assert len(self.clean_data) > 0, "No clean data available. Load data with set_clean_data() first."
            data = self.clean_data.copy()
        elif what == "preprocessed":
            assert len(self.preprocessed_data) > 0, "No preprocessed data available. Load data with set_preprocessed_data() first or use preprocess_data() function."
            data = self.preprocessed_data.copy()
        elif what == "classified":
            assert len(self.classified_data) > 0, "No classified data available. Load data with set_classified_data() first or  use classify_preprocessed_data() function."
            data = self.classified_data.copy()

        # Save the data
        for key, data in data.items():
            os.makedirs(f"{out_path}/{self.subset}/{self.id}", exist_ok=True)
            data.to_csv(f"{out_path}/{self.subset}/{self.id}/{self.id}_{key}.csv", index=False)

    def preprocess_clean_data(self, **kwargs) -> None:
        """
        Run the preprocessor on the cleaned gaze and target tables.

        Steps:
            1) `preprocess_gaze` on `clean_data["gaze"]` & blinks.
            2) `preprocess_targets` on `clean_data["targets"]` & trials.
            3) Copy `trials` and `blinks` into `preprocessed_data`.

        Args:
            **kwargs allows to pass additional arguments such as `blink_offset` for blink preprocessing.

        Returns:
            None
        """
        # Preprocess gaze data
        self.preprocessed_data["gaze"] = self.preprocessor.preprocess_gaze(self.clean_data["gaze"].copy(), self.clean_data["blinks"].copy(), **kwargs)

        # Preprocess target data
        self.preprocessed_data["targets"] = self.preprocessor.preprocess_targets(self.clean_data["targets"].copy(), self.clean_data["trials"].copy(), self.id, **kwargs)

        # Copy the remaining data
        self.preprocessed_data["trials"] = self.clean_data["trials"].copy()
        self.preprocessed_data["blinks"] = self.clean_data["blinks"].copy()
    
    def classify_preprocessed_data(self, trial_numbers: list = None, **kwargs) -> None:
        """
        Classify preprocessed gaze data and store results in `self.classified_data`.

        Args:
            trial_numbers (list, optional): Subset of trial numbers to classify. If
                None, all available trials are classified.

        Returns:
            None

        Raises:
            AssertionError: If `self.preprocessed_data` has not been set.
        """
        assert self.preprocessed_data, "No preprocessed data available. Load data with preprocessed=True first."
        
        self.classified_data["classified"] = self.classifier.classify(self.preprocessed_data.copy(), trial_numbers, **kwargs)

    def plot_trial(self, trial_number: int, filename: str = None, save_args: dict = None, show_legend: bool = True, colors = None) -> None:
        """
        Create a time‑series plot of target and gaze for a given trial, plus labels for
        classified events.

        Args:
            trial_number (int): Trial number to plot.
            filename (str, optional): If provided, save the figure to this path; when
                omitted, the plot is shown interactively.
            save_args (dict, optional): Keyword args forwarded to `plt.savefig`. Supports
                additional convenience keys `font_size` (default 10) and `figsize`
                (default (6, 4)). Defaults to LaTeX‑friendly values when not provided.
            show_legend (bool, optional): Whether to show legends for traces and ground
                truth. Defaults to True.
            colors (dict, optional): Mapping of label → color for all required keys
                {"Target X", "Target Y", "Gaze X", "Gaze Y", "Blink", "Saccade",
                "Fixation", "Smooth Pursuit", "none"}. If None, a color‑blind‑friendly
                palette is used.

        Returns:
            None

        Notes:
            Expects `self.classified_data["classified"]` to contain columns
            `trial_time`, `target_x`, `target_y`, `gaze_x`, `gaze_y`, and
            `ground_truth`.
        """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        import numpy as np

        # Use LaTeX-friendly plot defaults
        if save_args is None:
            save_args = {"dpi": 300, "bbox_inches": "tight", "format": "pdf"}

        font_size = save_args.pop("font_size", 10)  # default to 10 if not set
        figsize = save_args.pop("figsize", (6, 4)) # default figure size

        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "legend.fontsize": font_size * 0.9,
            "xtick.labelsize": font_size * 0.9,
            "ytick.labelsize": font_size * 0.9,
            "figure.dpi": save_args.get("dpi", 300),
        })

        plt.figure(figsize=figsize)

        # Get the trial data
        trial = self.classified_data["classified"].loc[
            self.classified_data["classified"]["trial_number"] == trial_number
        ].copy()

        # Extract metadata
        trial_name = trial['trial_name'].values[0]
        target_type = trial['target_type'].values[0]
        target_speed = trial['actual_speed'].values[0]
        target_trajectory = trial['target_trajectory'].values[0]

        # Define colors
        if colors is None:
            # Use color blind friendly colors
            colors = {
                "Target X": "#D55E00",
                "Target Y": "#0072B2",
                "Gaze X": "#56B4E9",
                "Gaze Y": "#E69F00",
                "Blink": "#000000",
                "Saccade": "#8742F0",
                "Fixation": "#009E73",
                "Smooth Pursuit": "#CC79A7",
                "none": "#000000"
            }
        else:
            required_keys = ["Target X", "Target Y", "Gaze X", "Gaze Y", "Blink", "Saccade", "Fixation", "Smooth Pursuit", "none"]
            assert isinstance(colors, dict), "Colors must be a dictionary."
            assert all(key in colors for key in required_keys), f"Colors must contain the following keys: {required_keys}"

        # Plot gaze and target
        plt.plot(trial["trial_time"], trial["target_x"], label="Target X", linestyle='solid', linewidth=0.5, color=colors["Target X"])
        plt.plot(trial["trial_time"], trial["target_y"], label="Target Y", linestyle='solid', linewidth=0.5, color=colors["Target Y"])
        plt.plot(trial["trial_time"], trial["gaze_x"], label="Gaze X", color=colors["Gaze X"])
        plt.plot(trial["trial_time"], trial["gaze_y"], label="Gaze Y", color=colors["Gaze Y"])

        # Legend for gaze and target
        if show_legend:
            legend_handles = [
                mlines.Line2D([], [], color=colors["Target X"], linestyle='solid', label="Target X"),
                mlines.Line2D([], [], color=colors["Target Y"], linestyle='solid', label="Target Y"),
                mlines.Line2D([], [], color=colors["Gaze X"], label="Gaze X"),
                mlines.Line2D([], [], color=colors["Gaze Y"], label="Gaze Y")
            ]
            plt.figlegend(
                handles=legend_handles,
                loc='upper left',
                bbox_to_anchor=(1.02, 0.9),
                borderaxespad=0.,
                title="Gaze & Target",
                frameon=False
            )
        # Ground truth
        y_base = min(trial["gaze_y"].min(), trial["gaze_x"].min())
        y = np.full(len(trial), np.nan)
        y_blinks = y.copy()
        y_blinks[trial["ground_truth"] == "Blink"] = y_base - 1
        y_sac = y.copy()
        y_sac[trial["ground_truth"] == "Saccade"] = y_base - 2
        y_pursuit = y.copy()
        y_pursuit[trial["ground_truth"] == "Smooth Pursuit"] = y_base - 3
        y_fix = y.copy()
        y_fix[trial["ground_truth"] == "Fixation"] = y_base - 3

        plt.plot(trial["trial_time"], y_blinks, color=colors["Blink"])
        plt.plot(trial["trial_time"], y_sac, color=colors["Saccade"])
        plt.plot(trial["trial_time"], y_pursuit, color=colors["Smooth Pursuit"])
        plt.plot(trial["trial_time"], y_fix, color=colors["Fixation"])

        # Ground truth legend
        if show_legend:
            gt_legend = [
                mlines.Line2D([], [], color=colors["Blink"], label="Blink"),
                mlines.Line2D([], [], color=colors["Saccade"], label="Saccade")
            ]
            if target_type == "jumping_circle":
                gt_legend.append(mlines.Line2D([], [], color=colors["Fixation"], label="Fixation"))
            else:
                gt_legend.append(mlines.Line2D([], [], color=colors["Smooth Pursuit"], label="Pursuit"))
            
            plt.figlegend(
                handles=gt_legend,
                loc='lower left',
                bbox_to_anchor=(1.02, 0.08),
                borderaxespad=0.,
                title="Ground Truth",
                frameon=False
            )

        # Layout and labels
        plt.subplots_adjust(right=0.98)
        plt.title(
            f"Participant: {self.id}\n{trial_name}: {target_type}, {target_speed} dva/s, {target_trajectory}",
            loc="left"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Position (deg)")

        # Save or show
        if filename:
            plt.savefig(filename, **save_args)
            plt.close()
        else:
            plt.show()
    
    def validation_check(self, raw_data_path: str = "data/raw") -> pd.DataFrame:
        """
        Parse the participant's raw `.asc` file and extract Eyelink validation sections.

        Args:
            raw_data_path (str, optional): Root path where raw `.asc` files are stored.
                Defaults to "data/raw".

        Returns:
            pd.DataFrame: One row per validation event with columns:
                ["participant_id", "validation_nr", "type", "eye", "result",
                 "error_avg", "error_max", "offset", "pix", "first_trial",
                 "last_trial"].
        """
        # Compile regex
        target_onset_re = re.compile(r".* (\d+): TARGET_ONSET.*")
        validation_re = re.compile(
            r".*VALIDATION (\w+) \w+ (\w+) (\w+) \w+ (-?\d+(?:\.\d+)?) avg\. (-?\d+(?:\.\d+)?) max  OFFSET (-?\d+(?:\.\d+)?) deg\. (-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?) pix\."
        )

        # Get the raw asc file
        asc_file = create_file_list(raw_data_path, [self.id, ".asc"])[0]
        with open(asc_file) as f:
            lines = f.readlines()

        # Initialize data frame
        results = pd.DataFrame(columns=["participant_id",
                                            "validation_nr", 
                                            "type", 
                                            "eye", 
                                            "result", 
                                            "error_avg", 
                                            "error_max", 
                                            "offset", 
                                            "pix", 
                                            "first_trial", 
                                            "last_trial"])
        
        # Analyze file to add rows to the dataframe
        i = 0
        trials = []
        new_row = []
        for line in lines:
            target_onset_match = target_onset_re.match(line)
            validation_match = validation_re.match(line)

            if target_onset_match:
                trials.append(target_onset_match.group(1))

            if validation_match:
                i += 1
                if len(trials) > 0:
                    new_row += [trials[0], trials[-1]]
                    results = pd.concat([results, pd.DataFrame([new_row], columns=results.columns)], ignore_index=True)
                    trials = []
                elif i > 1: 
                    new_row += [None, None]
                    results = pd.concat([results, pd.DataFrame([new_row], columns=results.columns)], ignore_index=True)
                new_row = [self.id] + [i] + list(validation_match.groups())

        if len(trials) > 0:
            new_row += [trials[0], trials[-1]]
            results = pd.concat([results, pd.DataFrame([new_row], columns=results.columns)], ignore_index=True)

        # Return validation results
        return results


    def missing_data_check(self, trials: list | str | int = "all", override_clean_data_path: str = None) -> float:
        """
        Compute the percentage of missing gaze samples after basic quality trimming.

        Args:
            trials (list | str | int, optional): Trial selector; "all" (default), a
                single int, or a list of ints.
            override_clean_data_path (str, optional): If provided, load clean data from
                this path instead of the default.

        Returns:
            float: Percentage of missing samples (0–100) based on NaNs in `gaze_x/y`.
        """
        # Make sure clean data is loaded
        if len(self.clean_data) == 0:
            if override_clean_data_path:
                self.set_clean_data(override_clean_data_path)
            else:
                self.set_clean_data()

        # Get relevant data
        gaze_data = self.clean_data["gaze"].copy()
        blink_data = self.clean_data["blinks"].copy()

        # Run data quality preprocessor
        gaze_data = DataQualityPreprocessor.preprocess_for_missing_data_check(gaze_data, blink_data)
        
        # Filter trials if specified
        trials_message = "for all trials"
        if trials != "all":
            if isinstance(trials, int):
                trials = [trials]
            trials_message = f"for trials {trials}"
            gaze_data = gaze_data[gaze_data["trial_number"].isin(trials)]

       # Calculate percentage of missing data
        n_missing = max(gaze_data["gaze_x"].isna().sum(), gaze_data["gaze_y"].isna().sum())
        n = len(gaze_data)
        p_missing = n_missing / n * 100

        # Return percentage of missing data
        print(f"Participant {self.id} has {p_missing:.2f}% missing data ({n_missing}/{n} total samples) {trials_message}.")
        return p_missing
    
    def fixation_precision_check(self, trials: list | str | int = "all", override_classified_data_path: str = None) -> float:
        """
        Estimate fixation precision (RMSE in degrees) during jumping‑circle trials.

        Args:
            trials (list | str | int, optional): Trial selector; "all" (default), a
                single int, or a list of ints.
            override_classified_data_path (str, optional): If provided, load classified
                data from this path instead of the default.

        Returns:
            float: Mean RMSE over detected fixation groups (degrees of visual angle).
        """
        # Make sure classified data is loaded
        if len(self.classified_data) == 0:
            if override_classified_data_path:
                self.set_classified_data(override_classified_data_path)
            else:
                self.set_classified_data()
        
        # Get relevant data
        classified_data = self.classified_data["classified"].copy()

        # Filter trials if specified
        trials_message = "for all trials"
        if trials != "all":
            if isinstance(trials, int):
                trials = [trials]
            trials_message = f"for trials {trials}"
            classified_data = classified_data[classified_data["trial_number"].isin(trials)]

        # Filter only jumping circle trials
        classified_data = classified_data[classified_data["target_type"] == "jumping_circle"]

        # Identify fixation groups
        classified_data["is_fixation"] = classified_data["ground_truth"] == "Fixation"
        classified_data["is_new_group"] = classified_data.groupby("trial_number")["is_fixation"].transform(lambda x: x & ~x.shift(fill_value=False))
        classified_data["fixation_group"] = classified_data.groupby("trial_number")["is_new_group"].cumsum()

        # Calculate RMSE for each group
        rmse = classified_data[classified_data["ground_truth"] == "Fixation"].groupby(["trial_number", "fixation_group"]).apply(
            lambda x: np.sqrt(np.mean(np.sqrt((x["gaze_x"] - x["gaze_x"].mean())**2 + (x["gaze_y"] - x["gaze_y"].mean())**2)**2))
            )
        
        # Return mean RMSE
        print(f"Participant {self.id} has a fixation precision of {rmse.mean():.2f} dva {trials_message}.")
        return rmse.mean()
    
    def target_alignment_check(self, trials: list | str | int = "all", override_clean_data_path: str = None) -> float:
        """
        Compute cosine similarity between gaze and reconstructed target trajectories.

        Args:
            trials (list | str | int, optional): Trial selector; "all" (default), a
                single int, or a list of ints.
            override_clean_data_path (str, optional): If provided, load clean data from
                this path instead of the default.

        Returns:
            float: Cosine similarity in [−1, 1] comparing concatenated x/y series.
        """
        # Make sure clean data is loaded
        if len(self.clean_data) == 0:
            if override_clean_data_path:
                self.set_clean_data(override_clean_data_path)
            else:
                self.set_clean_data()
        
        # Get relevant data
        gaze_data = self.clean_data["gaze"].copy()
        blink_data = self.clean_data["blinks"].copy()
        target_data = self.clean_data["targets"].copy()
        trial_info = self.clean_data["trials"].copy()

        # Run data quality preprocessor
        combined_data = DataQualityPreprocessor.process_for_alignment_check(gaze_data, blink_data, target_data, trial_info, self.id)

        # Filter trials if specified
        trials_message = "for all trials"
        if trials != "all":
            if isinstance(trials, int):
                trials = [trials]
            trials_message = f"for trials {trials}"
            combined_data = combined_data[combined_data["trial_number"].isin(trials)]

        # Select only x and y data
        gaze_trajectory = combined_data[["gaze_x", "gaze_y"]].values.astype(np.float64)
        target_trajectory = combined_data[["target_x", "target_y"]].values.astype(np.float64)

        # Get na mask
        na_mask = ~np.isnan(gaze_trajectory).any(axis=1) & ~np.isnan(target_trajectory).any(axis=1)
        
        # Get masked trajectories
        gaze_trajectory = gaze_trajectory[na_mask]
        target_trajectory = target_trajectory[na_mask]

        # Calculate cosine similarity
        cos_sim = cosine_similarity([gaze_trajectory.flatten()], [target_trajectory.flatten()])[0][0]
        
        # Return cosine similarity
        print(f"Participant {self.id} has a cosine similarity of {cos_sim:.2f} between gaze and target positions {trials_message}.")
        return cos_sim