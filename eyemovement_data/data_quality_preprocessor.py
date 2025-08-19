import pandas as pd
from typing import Tuple

from eyemovement_data.preprocessor import Preprocessor, OriginalPreprocessor


class DataQualityPreprocessor(Preprocessor):
    """
    Preprocessor for data quality checks.

    This subclass of `Preprocessor` provides two pipelines:
    * `preprocess_for_missing_data_check`: prepares gaze data for missing-data
      quality checks (e.g., removing outside-target intervals and blink samples).
    * `process_for_alignment_check`: prepares gaze and target data for alignment
      analyses by rescaling, converting to degrees, aligning trial times, and merging
      gaze with reconstructed targets.
    """

    @staticmethod
    def preprocess_for_missing_data_check(gaze_data: pd.DataFrame,
                                          blink_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess gaze data for a missing-data quality check.

        Steps:
            1) Get and validate monitor info.
            2) Validate gaze data columns.
            3) Trim to between TARGET_ONSET and TARGET_OFFSET.
            4) Remove blink samples (no offset padding).

        Args:
            gaze_data (pd.DataFrame): Raw gaze samples.
            blink_data (pd.DataFrame): Blink annotations.

        Returns:
            pd.DataFrame: Preprocessed gaze samples ready for missing-data checks.
        """
        # Get monitor info
        monitor_info = Preprocessor.get_monitor_info()

        # Validate inputs
        Preprocessor._validate_monitor_dict(monitor_info)
        Preprocessor._validate_gaze_data(gaze_data)

        # Remove before and after target
        gaze_data = gaze_data.groupby("trial_number").apply(Preprocessor._drop_rows_before_after_target).reset_index(drop=True)

        # Remove blinks
        gaze_data = Preprocessor._remove_blinks(gaze_data, blink_data, blink_offset=(0,0))

        return gaze_data
        
    def process_for_alignment_check(gaze_data: pd.DataFrame,
                                    blink_data: pd.DataFrame,
                                    target_data: pd.DataFrame,
                                    trial_info: pd.DataFrame,
                                    participant_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess gaze and target data for alignment quality check.

        Steps:
            1) Validate monitor and gaze data.
            2) Trim gaze to target interval, remove blinks.
            3) Rescale from corner to center, convert px→deg, align trial times.
            4) Preprocess targets with `OriginalPreprocessor`.
            5) Merge gaze and targets on participant/trial/time.
            6) Sort, forward/backfill target positions to match gaze sampling.

        Args:
            gaze_data (pd.DataFrame): Raw gaze samples.
            blink_data (pd.DataFrame): Blink annotations.
            target_data (pd.DataFrame): Raw target samples.
            trial_info (pd.DataFrame): Trial-level metadata.
            participant_id (str): Identifier of the participant.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Combined gaze-target data sorted and aligned.
        """
        # Get monitor info
        monitor_info = Preprocessor.get_monitor_info()

        # Validate inputs
        Preprocessor._validate_monitor_dict(monitor_info)
        Preprocessor._validate_gaze_data(gaze_data)

        # Preprocess gaze data
            # Remove before and after target
        gaze_data = gaze_data.groupby("trial_number").apply(Preprocessor._drop_rows_before_after_target).reset_index(drop=True)

            # Remove blinks
        gaze_data = Preprocessor._remove_blinks(gaze_data, blink_data, blink_offset=(0,0))

            # Rescale corner to center
        gaze_data = Preprocessor._rescale_corner_to_center(monitor_info, gaze_data, x_columns=["gaze_x"], y_columns=["gaze_y"])

            # Convert pixels to degrees
        gaze_data["gaze_x"] = round(gaze_data["gaze_x"].apply(lambda x: Preprocessor._pix2deg(x, monitor_info["distance"], monitor_info["width"], monitor_info["resolution"][0])), 3)
        gaze_data["gaze_y"] = round(gaze_data["gaze_y"].apply(lambda y: Preprocessor._pix2deg(y, monitor_info["distance"], monitor_info["height"], monitor_info["resolution"][1])), 3)

            # Set trial start times to 0 and convert to seconds
        gaze_data = Preprocessor._set_relative_trial_time(gaze_data)
        gaze_data["trial_time"] = round(gaze_data["trial_time"] / 1000, 3)

        # Preprocess target data
        target_data = OriginalPreprocessor.preprocess_targets(target_data, trial_info, participant_id)

        # Merge gaze and target data
        combined_data = gaze_data.merge(target_data,
                                        on=["participant_id", "trial_number", "trial_time"],
                                        how="left")
        
        # Sort by trial number and time
        combined_data = combined_data.sort_values(by=["trial_number", "trial_time"]).reset_index(drop=True)
        
        # Backfill and forward fill target positions to account for lower sampling rate
        combined_data["target_x"] = combined_data["target_x"].bfill().ffill()
        combined_data["target_y"] = combined_data["target_y"].bfill().ffill()
        
        return combined_data