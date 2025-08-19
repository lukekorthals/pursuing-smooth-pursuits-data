import math
import numpy as np
import pandas as pd
from typing import Tuple, Callable

class Preprocessor:
    """
    Base class for preprocessing eye-tracking data.

    This class provides validation utilities and shared helpers for transforming
    raw gaze/target data into a tidy, analysis-ready format (e.g., rescaling
    coordinates, converting pixels↔degrees, removing blinks, and aligning time
    within trials). Subclasses should implement the concrete `preprocess_gaze` and
    `preprocess_targets` methods.
    """
    @staticmethod
    def _validate_monitor_dict(monitor: dict):
        """
        Validate that the monitor info dictionary contains required keys and types.

        Args:
            monitor (dict): Dictionary expected to include keys `resolution` (tuple of
                two ints), `height` (int, mm), and `distance` (int, mm).

        Raises:
            AssertionError: If any key is missing or has an unexpected type/shape.
        """
        # Validate resolution
        assert "resolution" in monitor, "Monitor info does not contain a resolution key."
        assert isinstance(monitor["resolution"], tuple), "Monitor resolution must be a tuple."
        assert len(monitor["resolution"]) == 2, "Monitor resolution must be a tuple with two elements."
        assert all(isinstance(i, int) for i in monitor["resolution"]), "Monitor resolution elements must be integers."

        # Validate height
        assert "height" in monitor, "Monitor info does not contain a height key."
        assert isinstance(monitor["height"], int), "Monitor height must be a float."
        
        # Validate distance
        assert "distance" in monitor, "Monitor info does not contain a distance key."
        assert isinstance(monitor["distance"], int), "Monitor distance must be a float."
    
    @staticmethod
    def _validate_gaze_data(gaze_data: pd.DataFrame):
        """
        Validate that the gaze DataFrame contains required columns.

        Args:
            gaze_data (pd.DataFrame): Gaze samples with at least `gaze_x`, `gaze_y`,
                and `trial_number`.

        Raises:
            AssertionError: If required columns are missing.
        """
        assert "gaze_x" in gaze_data.columns, "gaze_data does not contain a gaze_x column."
        assert "gaze_y" in gaze_data.columns, "gaze_data does not contain a gaze_y column."
        assert "trial_number" in gaze_data.columns, "gaze_data does not contain a timestamp column."
    
    def _validate_trajectory(trajectory: str):
        """
        Validate that a trajectory label is one of the supported compass directions.

        Args:
            trajectory (str): One of {"north","south","west","east","northeast",
                "northwest","southeast","southwest"}.

        Raises:
            AssertionError: If the label is unsupported.
        """
        assert trajectory in ["north", "south", "west", "east", "northeast", "northwest", "southeast", "southwest"], "Trajectory must be one of 'north', 'south', 'west', 'east', 'northeast', 'northwest', 'southeast', 'southwest'."
    
    # Utils
    @staticmethod
    def _pix2deg(px, distance, size, resolution):
        """
        Convert a linear pixel displacement to degrees of visual angle (DVA).

        This uses the exact geometric relation `deg = 2*atan((px*px_size)/(2*distance))`.

        Args:
            px (float): Pixel displacement.
            distance (float): Eye-to-screen distance in the same linear units as `size`.
            size (float): Physical size of the full screen dimension corresponding to
                `resolution` (e.g., width in mm for x, height in mm for y).
            resolution (int): Number of pixels along that screen dimension.

        Returns:
            float: Displacement in degrees of visual angle, rounded to 3 decimals.
        """
        px_size = size / resolution
        return round(2*math.degrees(math.atan((px*px_size)/(2*distance))), 3)
    
    @staticmethod
    def _deg2pix(deg, distance, size, resolution):
        """
        Convert degrees of visual angle (DVA) to pixels.

        Args:
            deg (float): Displacement in degrees of visual angle.
            distance (float): Eye-to-screen distance in the same linear units as `size`.
            size (float): Physical size of the corresponding screen dimension.
            resolution (int): Number of pixels along that dimension.

        Returns:
            float: Pixel displacement (rounded to 3 decimals).
        """
        px_size = size / resolution
        return round((2*distance*math.tan(math.radians(deg/2)))/px_size, 3)
    
    @staticmethod
    def _pix2deg_faulty(monitor: dict, px: float) -> float:
        """
        Convert pixels to degrees using a simplified screen-height–based approximation.

        Note:
            This formula assumes a linear mapping based on total screen height and may
            be inaccurate for large eccentricities. Prefer `_pix2deg` when precise
            geometry is needed.

        Args:
            monitor (dict): Monitor info as returned by `get_monitor_info()`.
            px (float): Pixel displacement along the vertical dimension.

        Returns:
            float: Approximate degrees of visual angle.
        """
        Preprocessor._validate_monitor_dict(monitor)

        return math.degrees(math.atan2(.5 * monitor["height"], monitor["distance"])) / (.5 * monitor["resolution"][1]) * px
    
    @staticmethod
    def _deg2pix_faulty(monitor: dict, deg: float) -> float:
        """
        Convert degrees to pixels using a simplified screen-height–based approximation.

        Args:
            monitor (dict): Monitor info as returned by `get_monitor_info()`.
            deg (float): Degrees of visual angle along the vertical dimension.

        Returns:
            float: Approximate pixel displacement.
        """
        Preprocessor._validate_monitor_dict(monitor)

        return (deg / math.degrees(math.atan2(.5 * monitor["height"], monitor["distance"])) * (.5 * monitor["resolution"][1]))
    
    @staticmethod
    def _rescale_corner_to_center(monitor_info: dict, data: pd.DataFrame, x_columns: list = ["gaze_x"], y_columns: list = ["gaze_y"]) -> pd.DataFrame:
        """
        Rescale coordinates from screen-origin-at-corner to screen-origin-at-center.

        The function subtracts half the screen width from x-columns and half the screen
        height from y-columns, and flips the y-axis so positive values are up.

        Args:
            monitor_info (dict): Monitor dictionary including `resolution` (w, h).
            data (pd.DataFrame): DataFrame containing the coordinate columns to modify.
            x_columns (list[str], optional): Names of x-columns to re-center. Defaults to ["gaze_x"].
            y_columns (list[str], optional): Names of y-columns to re-center/flip. Defaults to ["gaze_y"].

        Returns:
            pd.DataFrame: The same DataFrame with adjusted coordinates.
        """
        Preprocessor._validate_monitor_dict(monitor_info)

        assert all(col in data.columns for col in x_columns), "Data does not contain all defined x_columns."
        assert all(col in data.columns for col in y_columns), "Data does not contain all defined y_columns."

        screen_x_center = monitor_info["resolution"][0] / 2
        screen_y_center = monitor_info["resolution"][1] / 2

        for col in x_columns:
            data[col] = data[col] - screen_x_center 
        for col in y_columns:
            data[col] = screen_y_center - data[col]
        
        return data
    
    @staticmethod
    def get_monitor_info() -> dict:
        """
        Return canonical monitor parameters used throughout preprocessing.

        Returns:
            dict: Dictionary with keys `height` (mm), `width` (mm), `distance` (mm),
                `resolution` (tuple[int, int]), and `refresh_rate` (Hz).
        """
        monitor = {
            "height": 365, # mm
            "width": 614, # mm
            "distance": 700, # mm
            "resolution": (2560, 1440), # px
            "refresh_rate": 144 # Hz
            }
        return monitor
    
    @staticmethod
    def _rolling_mean_smoothing(series: pd.Series, rolling_mean_window: int = 1, **kwargs) -> pd.Series:
        """
        Apply a simple rolling-mean smoother.

        Args:
            series (pd.Series): Input time series to smooth.
            rolling_mean_window (int, optional): Window size in samples. Defaults to 1
                (no smoothing).
            **kwargs: Accepted for API compatibility; unused.

        Returns:
            pd.Series: Smoothed series (same index as input).
        """
        return series.rolling(window=rolling_mean_window).mean()
    
    @staticmethod
    def _drop_rows_before_after_target(gaze_data: pd.DataFrame) -> pd.DataFrame:
        """
        Trim gaze data to the interval between TARGET_ONSET and TARGET_OFFSET (inclusive).

        Args:
            gaze_data (pd.DataFrame): Gaze samples including a `message` column that may
                contain TARGET_ONSET/TARGET_OFFSET markers.

        Returns:
            pd.DataFrame: Subset of rows within the target interval. If markers are
                missing, returns the original DataFrame and prints a warning.
        """
        assert "message" in gaze_data.columns, "Data does not have a message column."

        gaze_data = gaze_data.reset_index(drop=True)
        gaze_data["message"] = gaze_data["message"].fillna("")
        start_index = gaze_data[gaze_data["message"].str.contains("TARGET_ONSET")].index
        end_index = gaze_data[gaze_data["message"].str.contains("TARGET_OFFSET")].index
        if not start_index.empty:
            return gaze_data.loc[start_index[0]:end_index[0]]
        else:
            print("WARNING! TARGET_ONSET and/or TARGET_OFFSET not found in message column.")
            return gaze_data
        
    @staticmethod
    def _remove_blinks(gaze_data: pd.DataFrame, blink_data: pd.DataFrame, blink_offset: Tuple[int, int] = (50, 50), **kwargs) -> pd.DataFrame:
        """
        Mask gaze samples during blinks and optional padding by setting gaze to NaN.

        Args:
            gaze_data (pd.DataFrame): Gaze samples with `trial_number` and `trial_time`.
            blink_data (pd.DataFrame): Blink epochs with columns `trial_number`,
                `blink_start`, and `blink_end` in the same time units as `trial_time`.
            blink_offset (Tuple[int, int], optional): Padding before/after each blink
                (e.g., in ms) to mark as blink. Defaults to (50, 50).
            **kwargs: Accepted for compatibility; unused.

        Returns:
            pd.DataFrame: Copy with a boolean `blink` column and NaN in `gaze_x/y` where
                blinks occur.
        """
        gaze_data["blink"] = False
        for i, row in blink_data.iterrows():
            gaze_data.loc[(gaze_data["trial_number"] == row["trial_number"]) &
                        (gaze_data["trial_time"] >= row["blink_start"] - blink_offset[0]) &
                        (gaze_data["trial_time"] <= row["blink_end"] + blink_offset[1]), "blink"] = True
        gaze_data.loc[gaze_data["blink"], ["gaze_x", "gaze_y"]] = np.nan
        return gaze_data
        
    # Preprocessing methods
    @staticmethod
    def preprocess_gaze(gaze_data: pd.DataFrame, 
                        blink_data: pd.DataFrame, 
                        **kwargs) -> pd.DataFrame:
        """
        Abstract hook to preprocess gaze samples.

        Args:
            gaze_data (pd.DataFrame): Raw gaze samples.
            blink_data (pd.DataFrame): Blink annotations.
            **kwargs: Implementation-specific options.

        Returns:
            pd.DataFrame: Preprocessed gaze samples ready for classification.
        """
        raise NotImplementedError("Implement this method in a subclass.")
    
    @staticmethod
    def preprocess_targets(target_data: pd.DataFrame, 
                           trial_info: pd.DataFrame,
                           target_distance_in_deg: float, 
                           target_radius_in_pix: float, 
                           participant_id: str, 
                           **kwargs) -> pd.DataFrame:
        """
        Abstract hook to preprocess target/trajectory samples.

        Args:
            target_data (pd.DataFrame): Raw target positions over time.
            trial_info (pd.DataFrame): Trial-level metadata for the same participant.
            target_distance_in_deg (float): Target eccentricity in degrees.
            target_radius_in_pix (float): Radius of the moving element in pixels.
            participant_id (str): Participant identifier.
            **kwargs: Implementation-specific options.

        Returns:
            pd.DataFrame: Preprocessed target samples aligned to trials/time base.
        """
        raise NotImplementedError("Implement this method in a subclass.")

    @staticmethod
    def _set_relative_trial_time(data: pd.DataFrame) -> pd.DataFrame:
        """
        Re-reference time so that each trial starts at 0.

        Args:
            data (pd.DataFrame): Samples containing `trial_time` and `trial_number`.

        Returns:
            pd.DataFrame: DataFrame with `trial_time` shifted to start at 0 per trial.
        """
        assert "trial_time" in data.columns, "Data does not have a trial_time column."
        assert "trial_number" in data.columns, "Data does not have a trial_number column."

        trial_start_times = data.groupby("trial_number")["trial_time"].transform("min")
        data["trial_time"] = data["trial_time"] - trial_start_times

        return data
    

class OriginalPreprocessor(Preprocessor):
    """
    Concrete preprocessor implementing a typical pipeline for this project.

    Operations include trimming to target epochs, blink masking, smoothing, recentre
    from corner to screen center, pixel↔degree conversions, and trial-time alignment.
    For targets, it also reconstructs positions for back-and-forth arrays and
    upsamples to 1000 Hz where required.
    """

    @staticmethod
    def preprocess_gaze(gaze_data: pd.DataFrame, 
                        blink_data: pd.DataFrame, 
                        **kwargs) -> pd.DataFrame:
        """
        Preprocess gaze samples for a single participant/session.

        Steps:
            1) Validate monitor and input columns.
            2) Keep only data between TARGET_ONSET and TARGET_OFFSET.
            3) Drop unused columns, remove blinks, compute samplewise velocity,
               smooth gaze, re-center coordinates, convert px→deg, set trial-relative
               time, and convert ms→s.

        Args:
            gaze_data (pd.DataFrame): Raw gaze samples with `gaze_x`, `gaze_y`,
                `trial_time`, `trial_number`, and `message`.
            blink_data (pd.DataFrame): Blink annotations for the same trials.
            **kwargs: Passed to internal helpers (e.g., smoothing window).

        Returns:
            pd.DataFrame: Cleaned and augmented gaze samples, including a `velocity`
                column (deg/s) and trial-relative `trial_time` (seconds).
        """
        # Get monitor info
        monitor_info = Preprocessor.get_monitor_info()

        # Validate inputs
        Preprocessor._validate_monitor_dict(monitor_info)
        Preprocessor._validate_gaze_data(gaze_data)

        # Remove before and after target
        gaze_data = gaze_data.groupby("trial_number").apply(Preprocessor._drop_rows_before_after_target).reset_index(drop=True)

        # Drop unnecessary columns
        gaze_data = gaze_data.drop(columns=["message", "cornea_info"])

        # Remove blinks
        gaze_data = Preprocessor._remove_blinks(gaze_data, blink_data, **kwargs)

        # Calc velocities
        gaze_data["velocity"] = np.sqrt(np.square(gaze_data["gaze_x"].diff()) + np.square(gaze_data["gaze_y"].diff())) / gaze_data["trial_time"].diff()
        
        # Rolling mean smoothing
        gaze_data[["gaze_x", "gaze_y"]] = gaze_data[["gaze_x", "gaze_y"]].apply(lambda x: OriginalPreprocessor._rolling_mean_smoothing(x, **kwargs))

        # Rescale corner to center
        gaze_data = Preprocessor._rescale_corner_to_center(monitor_info, gaze_data, x_columns=["gaze_x"], y_columns=["gaze_y"])

        # Convert pixels to degrees
        gaze_data["gaze_x"] = round(gaze_data["gaze_x"].apply(lambda x: Preprocessor._pix2deg(x, monitor_info["distance"], monitor_info["width"], monitor_info["resolution"][0])), 3)
        gaze_data["gaze_y"] = round(gaze_data["gaze_y"].apply(lambda y: Preprocessor._pix2deg(y, monitor_info["distance"], monitor_info["height"], monitor_info["resolution"][1])), 3)

        # Set trial start times to 0 and convert to seconds
        gaze_data = Preprocessor._set_relative_trial_time(gaze_data)
        gaze_data["trial_time"] = round(gaze_data["trial_time"] / 1000, 3)

        
        return gaze_data
    
    @staticmethod
    def preprocess_targets(target_data: pd.DataFrame, 
                           trial_info: pd.DataFrame,
                           participant_id: str,
                           target_distance_in_deg: float = 20, 
                           target_radius_in_pix: float= 10, 
                           **kwargs) -> pd.DataFrame:
        """
        Preprocess target positions and reconstruct back-and-forth array trajectories.

        Args:
            target_data (pd.DataFrame): Raw target positions with `trial_number`,
                `trial_time`, `target_x`, `target_y`.
            trial_info (pd.DataFrame): Trial metadata including `target_type`,
                `target_speed`, and `target_trajectory`.
            participant_id (str): Participant identifier used for start-direction flags.
            target_distance_in_deg (float, optional): Eccentricity in degrees. Defaults to 20.
            target_radius_in_pix (float, optional): Element radius in pixels. Defaults to 10.
            **kwargs: Additional options (unused).

        Returns:
            pd.DataFrame: Target samples in degrees, upsampled to 1000 Hz, with
                trial-relative time in seconds.
        """
        # Get monitor info
        monitor_info = Preprocessor.get_monitor_info()

        # Drop unnecessary columns
        target_data = target_data.drop(columns="experiment_time")

        # Recalculate target positions for back and forth arrays
        target_distance_in_pix = Preprocessor._deg2pix_faulty(monitor_info, target_distance_in_deg)
        start_go_back = False
        if participant_id in ["06b8d2d3", "6cde27b5", "21db28aa"]:
            start_go_back = True
        
        bf_trial_numbers = trial_info.query("target_type == 'back_and_forth_array'")["trial_number"].reset_index(drop=True)
        bf_target_speeds = trial_info.query("target_type == 'back_and_forth_array'")["target_speed"]
        bf_target_trajectories = trial_info.query("target_type == 'back_and_forth_array'")["target_trajectory"]

        for trial_number, target_speed, target_trajectory in zip(bf_trial_numbers, bf_target_speeds, bf_target_trajectories):
            trial_times = target_data.query(f"trial_number == {trial_number}")["trial_time"]
            target_speed_in_pix = Preprocessor._deg2pix_faulty(monitor_info, target_speed)
            fixed_target_positions = OriginalPreprocessor._target_fix_calculate_target_positions(target_trajectory,
                                                                                         target_distance_in_pix,
                                                                                         target_speed_in_pix,
                                                                                         trial_times,
                                                                                         target_radius_in_pix,
                                                                                         start_go_back,
                                                                                         monitor_info["refresh_rate"]
                                                                                         )
            target_data.loc[target_data["trial_number"] == trial_number, "target_x"] = fixed_target_positions[0]
            target_data.loc[target_data["trial_number"] == trial_number, "target_y"] = fixed_target_positions[1]

        # Convert to degrees
        target_data["target_x"] = round(target_data["target_x"].apply(lambda x: OriginalPreprocessor._pix2deg(x, monitor_info["distance"], monitor_info["width"], monitor_info["resolution"][0])), 3)
        target_data["target_y"] = round(target_data["target_y"].apply(lambda y: OriginalPreprocessor._pix2deg(y, monitor_info["distance"], monitor_info["height"], monitor_info["resolution"][1])), 3)

        # round trial time to 3 decimals
        target_data["trial_time"] = round(target_data["trial_time"], 3)

        # Upsample to 1000 Hz
        def resample_group(group):
            # convert to datetime and set index
            group["trial_time"] = pd.to_datetime(group["trial_time"], unit='s')
            group.set_index("trial_time", inplace=True)

            # Upsample to 1ms
            group = group.resample('1ms').ffill()

            # reset to before
            group.reset_index(inplace=True)
            group["trial_time"] = group["trial_time"].astype('int64') / 1e9
            
            # move trial_time to 3rd column
            cols = group.columns.tolist()
            cols = cols[1:3] + [cols[0]] + cols[3:]
            group = group[cols]

            return group

        target_data = target_data.groupby("trial_number").apply(resample_group).reset_index(drop=True)


        return target_data
    
    # For preprocessing target data
    @staticmethod
    def _target_fix_calculate_target_positions(target_trajectory: str, 
                                               target_distance_in_pix: float, 
                                               target_speed_in_pix: float, 
                                               trial_time: np.array, 
                                               target_radius_in_pix=10, 
                                               start_go_back: bool = True, 
                                               monitor_refresh_rate: float = 144):
        """
        Compute target (x, y) positions by combining field motion and element motion.

        Args:
            target_trajectory (str): Compass direction of the field movement.
            target_distance_in_pix (float): Total movement distance of the field (pixels).
            target_speed_in_pix (float): Field speed (pixels per time unit of `trial_time`).
            trial_time (np.array): Time vector (matching sampling of target display).
            target_radius_in_pix (int, optional): Element radius inside the field. Defaults to 10.
            start_go_back (bool, optional): Initial back/forth state for the element. Defaults to True.
            monitor_refresh_rate (float, optional): Display refresh rate in Hz. Defaults to 144.

        Returns:
            Tuple[np.array, np.array]: Arrays of x and y positions in pixels.
        """
        ## Get the positions of the moving field
        start_pos = OriginalPreprocessor._target_fix_get_field_start_position(target_trajectory, target_distance_in_pix)
        movement = OriginalPreprocessor._target_fix_get_field_movement_vector(target_trajectory, target_speed_in_pix, trial_time)
        field_positions = (start_pos[0] + movement[0], start_pos[1] + movement[1])

        ## Get the positions of the jumping element within the moving field
        element_positions = OriginalPreprocessor._target_fix_calcualate_element_positions(target_trajectory, trial_time, target_radius_in_pix, start_go_back, monitor_refresh_rate)
        
        ## Combine for target positions
        target_positions = (field_positions[0] + element_positions[0], field_positions[1] + element_positions[1])

        return target_positions

    @staticmethod
    def _target_fix_get_field_start_position(target_trajectory: str, movement_distance_in_pix: float):
        """
        Return the starting (x, y) position of the moving field for a trajectory.

        Args:
            target_trajectory (str): Compass direction.
            movement_distance_in_pix (float): Total movement distance (pixels).

        Returns:
            Tuple[float, float]: Starting coordinates relative to screen center.
        """
        # Offsets from the center of the screen
        center = 0
        top = right = movement_distance_in_pix / 2
        bot = left = -movement_distance_in_pix / 2

        # Starting positions for each target trajectory
        starting_positions = {
            "north": (center, bot),
            "northeast": (left, bot),
            "east": (left, center),
            "southeast": (left, top),
            "south": (center, top),
            "southwest": (right, top),
            "west": (right, center),
            "northwest": (right, bot)
        }

        return starting_positions[target_trajectory]
    
    def _target_fix_get_field_movement_vector(target_trajectory: str, target_speed: float, trial_time: np.array) -> Tuple[np.array, np.array]:
        """
        Compute the field movement vector over time for a given trajectory and speed.

        Args:
            target_trajectory (str): Compass direction.
            target_speed (float): Speed in pixels per unit time.
            trial_time (np.array): Time vector.

        Returns:
            Tuple[np.array, np.array]: Delta x and delta y over time (pixels).
        """
        # Movement 
        travelled = target_speed * trial_time
        center = 0
        up = right = travelled
        down = left = -travelled


        # Next positions for each target trajectory
        next_positions = {
            "north": (center, up),
            "northeast": (up, right),
            "east": (right, center),
            "southeast": (right, down),
            "south": (center, down),
            "southwest": (left, down),
            "west": (left, center),
            "northwest": (left, up)
        }

        return next_positions[target_trajectory]
    
    def _target_fix_calcualate_element_positions(target_trajectory: str, trial_time: np.array, target_radius_in_pix: int, go_back: float = True, monitor_refresh_rate: float = 144) -> np.array:
        """
        Compute the element's back-and-forth motion inside the moving field.

        The element alternates between `back` and `forth` offsets every display frame,
        with initial direction controlled by `go_back` and toggled at `monitor_refresh_rate`.

        Args:
            target_trajectory (str): Compass direction of field movement.
            trial_time (np.array): Time vector (one element per sample).
            target_radius_in_pix (int): Radius of the element (pixels).
            go_back (bool, optional): Initial state for back-and-forth toggling. Defaults to True.
            monitor_refresh_rate (float, optional): Display refresh rate in Hz. Defaults to 144.

        Returns:
            Tuple[np.array, np.array]: Element x and y offsets relative to the field (pixels).
        """

        # Determine back and forth element positions inside the moving field
        negative = -target_radius_in_pix * 2   
        positive = target_radius_in_pix * 2   
        center = 0


        back_and_forth_offset = {
            # In the experimental code all vertical trajectories used the same back and forth offset
            "north": {"back": (center, negative), "forth": (center, positive)},
            "south": {"back": (center, negative), "forth": (center, positive)},

            # In the experimental code all horizontal trajectories used the same back and forth offset
            "west": {"back": (negative, center), "forth": (positive, center)},
            "east": {"back": (negative, center), "forth": (positive, center)},

            # In the experimental code the diagonal trajectories used specific back and forth offsets
            "northeast": {"back": (negative, negative), "forth": (positive, positive)},
            "northwest": {"back": (positive, negative), "forth": (negative, positive)},

            "southeast": {"back": (negative, positive), "forth": (positive, negative)},
            "southwest": {"back": (positive, positive), "forth": (negative, negative)}
        }

        # Get frames where the element position was updated
        frames = []
        for f, _ in enumerate(trial_time):
            if f % monitor_refresh_rate == 0:
                go_back = not go_back
            frames.append(go_back)
        frames[0] = True
        frames = np.array(frames)

        # Get vector of element positions    
        element_pos_x = np.full(frames.shape, back_and_forth_offset[target_trajectory]["forth"][0])
        element_pos_x[frames] = back_and_forth_offset[target_trajectory]["back"][0]

        element_pos_y = np.full(frames.shape, back_and_forth_offset[target_trajectory]["forth"][1])
        element_pos_y[frames] = back_and_forth_offset[target_trajectory]["back"][1]

        return (element_pos_x, element_pos_y)
    
    


    
    
    

    