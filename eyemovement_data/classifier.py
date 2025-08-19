import numpy as np
import pandas as pd
from typing import Tuple


class Classifier:
    """
    Base class for eye movement classification algorithms.

    This class defines the interface and common utilities for classifiers that
    process preprocessed eye movement data to classify eye movement events.
    """

    @staticmethod
    def classify(preprocessed_data: dict[str, pd.DataFrame],
                 trial_numbers: list = None, 
                 **kwargs) -> pd.DataFrame:
        """
        Abstract method to classify eye movement data.

        Args:
            preprocessed_data (dict[str, pd.DataFrame]): Dictionary containing preprocessed dataframes.
            trial_numbers (list, optional): List of trial numbers to classify. If None, classify all.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.

        Returns:
            pd.DataFrame: Classified eye movement data.
        """
        raise NotImplementedError("Subclass must implement abstract method")


    @staticmethod
    def validate_trial_numbers(preprocessed_data: dict[str, pd.DataFrame],
                               trial_numbers: list = None) -> list:
        """
        Ensures that the trial numbers provided are valid and exist in both the trial and gaze data.

        Args:
            preprocessed_data (dict[str, pd.DataFrame]): Dictionary with keys "trials" and "gaze".
            trial_numbers (list, optional): List of trial numbers to validate. If None, uses all.

        Returns:
            list: Validated list of trial numbers.
        """
        trial_data = preprocessed_data["trials"]
        gaze_data = preprocessed_data["gaze"]

        if trial_numbers is not None:
            for trial_number in trial_numbers:
                if trial_number not in trial_data.trial_number.values:
                    raise ValueError(f"Invalid trial number: {trial_number} missing from trial data")
                if trial_number not in gaze_data.trial_number.values:
                    raise ValueError(f"Invalid trial number: {trial_number} missing from gaze data")
        else:
            trial_numbers = trial_data.trial_number.values
            trial_number = [trial_number for trial_number in trial_numbers if trial_number in gaze_data.trial_number.values]
        
        return trial_numbers

    

class OriginalClassifier(Classifier):
    """
    Implements a velocity-threshold-based classification of eye movements.

    This classifier uses velocity thresholds to differentiate between saccades,
    fixations, smooth pursuits, and blinks based on preprocessed gaze data.
    """
    
    @staticmethod
    def classify(preprocessed_data: dict[str, pd.DataFrame],
                 trial_numbers: list = None,
                 **kwargs) -> pd.DataFrame:
        """
        Perform classification of eye movement events on the provided data.

        Args:
            preprocessed_data (dict[str, pd.DataFrame]): Dictionary containing preprocessed dataframes.
            trial_numbers (list, optional): List of trial numbers to classify. If None, classify all.
            **kwargs: Additional keyword arguments for classification steps.

        Returns:
            pd.DataFrame: DataFrame containing classified eye movement events.
        """
        
        # Validate trial_numbers
        trial_numbers = OriginalClassifier.validate_trial_numbers(preprocessed_data, trial_numbers)

        # Get relevant data
        gaze_data = preprocessed_data["gaze"].copy()
        target_data = preprocessed_data["targets"].copy()
        trial_data = preprocessed_data["trials"].copy()
        blink_data = preprocessed_data["blinks"].copy()

        # Select relevant trials
        gaze_data = gaze_data.loc[gaze_data["trial_number"].isin(trial_numbers)]
        target_data = target_data.loc[target_data["trial_number"].isin(trial_numbers)]
        trial_data = trial_data.loc[trial_data["trial_number"].isin(trial_numbers)]
        blink_data = blink_data.loc[blink_data["trial_number"].isin(trial_numbers)]

        # Initialize classified data
        classified_data = trial_data.merge(gaze_data, 
                                            on=["participant_id", "trial_number"], 
                                            how="right").merge(target_data,
                                                                on=["participant_id", "trial_number", "trial_time"],
                                                                how="left")
        classified_data["ground_truth"] = "none"

        # Classify blinks
        classified_data = OriginalClassifier.classify_blinks(classified_data, **kwargs)

        # Determine Velocity Threshold
        classified_data = OriginalClassifier.determine_velocity_threshold(classified_data, **kwargs)

        # Classify based on velocity threshold
        classified_data = OriginalClassifier.velocity_threshold_based_classification(classified_data, **kwargs)

        # Relabel short events
        classified_data = OriginalClassifier.relabel_short_events(classified_data, **kwargs)

        return classified_data
    
    
    @staticmethod
    def classify_blinks(gaze_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Classify blinks according to the blinks identified during preprocessing.

        Args:
            gaze_data (pd.DataFrame): DataFrame containing gaze data with blink information.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: Updated gaze data with blinks classified.
        """
        gaze_data.loc[gaze_data["blink"], "ground_truth"] = "Blink"
        return gaze_data
    
    @staticmethod
    def determine_velocity_threshold(gaze_data: pd.DataFrame,
                                     threshold_time_window: Tuple[float, float] = (0, 30),
                                     velocity_threshold_scaling_constant: float = 1.5,
                                     **kwargs) -> pd.DataFrame:
        """
        Determine the velocity threshold for this trial.

        Args:
            gaze_data (pd.DataFrame): DataFrame containing gaze data.
            threshold_time_window (Tuple[float, float], optional): Time window to calculate threshold in seconds. Defaults to (0, 30) covering the entire trial.
            velocity_threshold_scaling_constant (float, optional): Scaling constant for threshold. Defaults to 1.5.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: Gaze data with velocity threshold values added.
        """
        threshold_window = gaze_data.loc[(gaze_data["trial_time"] <= threshold_time_window[1]) & 
                                         (gaze_data["trial_time"] >= threshold_time_window[0])]        

        threshold_window["median_vel"] = threshold_window.groupby(["trial_number"])["velocity"].transform("median")
        threshold_window["q3_vel"] = threshold_window.groupby(["trial_number"])["velocity"].transform(lambda x: x.quantile(0.75))
        threshold_window["sac_vel_threshold"] = threshold_window["median_vel"] + velocity_threshold_scaling_constant * threshold_window["q3_vel"]
        gaze_data = gaze_data.merge(threshold_window[["trial_number", "sac_vel_threshold"]].drop_duplicates(), on="trial_number")
        
        return gaze_data
    
    @staticmethod
    def velocity_threshold_based_classification(gaze_data: pd.DataFrame,
                                                **kwargs) -> pd.DataFrame:
        """
        Classify saccades, fixations, and smooth pursuits based on the velocity threshold.

        Args:
            gaze_data (pd.DataFrame): DataFrame containing gaze data with velocity thresholds.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: Gaze data with eye movement events classified.
        """
        # Classify fast events as saccades
        gaze_data.loc[gaze_data["velocity"] > gaze_data["sac_vel_threshold"], "ground_truth"] = "Saccade"
        
        # Classify slow events in jumping circle trials as fixations
        gaze_data.loc[(gaze_data["velocity"] <= gaze_data["sac_vel_threshold"]) & (gaze_data["target_type"] == "jumping_circle"), "ground_truth"] = "Fixation"
        
        # Classify slow events in non-jumping circle trials as smooth pursuit
        gaze_data.loc[(gaze_data["velocity"] <= gaze_data["sac_vel_threshold"]) & (gaze_data["target_type"] != "jumping_circle"), "ground_truth"] = "Smooth Pursuit"
        
        return gaze_data
    
    @staticmethod
    def relabel_short_events(gaze_data: pd.DataFrame,
                             min_sac_duration: float = 0.01,  # According to Lappi (2016)
                             min_fix_duration: float = 0.01,  # Conservative estimate to allow for short events
                             min_sp_duration: float = 0.01,  # Conservative estimate to allow for short events
                             **kwargs) -> pd.DataFrame:
        """
        Relabel short events to the previous event type, grouped by trial_number.

        Args:
            gaze_data (pd.DataFrame): DataFrame containing classified gaze data.
            min_sac_duration (float, optional): Minimum saccade duration to keep. Defaults to 0.01.
            min_fix_duration (float, optional): Minimum fixation duration to keep. Defaults to 0.01.
            min_sp_duration (float, optional): Minimum smooth pursuit duration to keep. Defaults to 0.01.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: Gaze data with short events relabeled.
        """
        
        def calculate_durations(trial_df):
            """Calculate event durations within each trial."""
            trial_df["event_group"] = (trial_df["ground_truth"] != trial_df["ground_truth"].shift()).cumsum()
            duration_map = (
                trial_df.groupby("event_group")["trial_time"].transform("last") -
                trial_df.groupby("event_group")["trial_time"].transform("first") + 0.001
            )
            trial_df["duration"] = duration_map
            return trial_df
        
        def relabel_events(trial_df):
            """Apply relabeling rules within each trial."""
            trial_df = calculate_durations(trial_df)
            
            # Relabel moving circle trials
            mask_moving_circle = trial_df["target_type"].isin(["moving_circle", "back_and_forth_array"])
            trial_df.loc[
                mask_moving_circle & (trial_df["ground_truth"] == "Smooth Pursuit") & (trial_df["duration"] < min_sp_duration),
                "ground_truth"
            ] = "Saccade"
                        
            trial_df = calculate_durations(trial_df)  # Recalculate after relabeling
            trial_df.loc[
                mask_moving_circle & (trial_df["ground_truth"] == "Saccade") & (trial_df["duration"] < min_sac_duration),
                "ground_truth"
            ] = "Smooth Pursuit"
            
            # Relabel jumping circle trials
            mask_jumping_circle = trial_df["target_type"].isin(["jumping_circle"])
            trial_df.loc[
                mask_jumping_circle & (trial_df["ground_truth"] == "Fixation") & (trial_df["duration"] < min_fix_duration),
                "ground_truth"
            ] = "Saccade"

            trial_df = calculate_durations(trial_df)  # Recalculate after relabeling
            trial_df.loc[
                mask_jumping_circle & (trial_df["ground_truth"] == "Saccade") & (trial_df["duration"] < min_sac_duration),
                "ground_truth"
            ] = "Fixation"
            
            return trial_df
        
        # Apply relabeling per trial
        gaze_data = gaze_data.groupby("trial_number", group_keys=False).apply(relabel_events)
        
        return gaze_data
