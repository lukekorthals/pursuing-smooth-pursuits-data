# ----------------------------------------------------------------------------
# Data Cleaning Pipeline for Eye-Tracking Exports
#
# This script converts Eyelink .asc files to a set of tidy CSVs and performs
# minimal renaming/selection to standardize columns for downstream analysis.
# It assumes the following directory layout:
#   data/
#     raw/   -> contains per-participant folders with an .asc and associated CSVs
#     clean/ -> will be created/updated with cleaned CSVs
#
# Main steps per participant:
#   1) Convert .asc to gaze/eyetracker/blinks CSVs (via `asc2csv`).
#   2) Clean auxiliary CSVs exported by the experiment code (participant, targets, trials).
#   3) Write cleaned files into data/clean/{train|test}/{participant}/
#
# ----------------------------------------------------------------------------

# Ensure required packages are installed and loaded
if(!require("eyelinker")) install.packages("eyelinker")
if(!require("tidyverse")) install.packages("tidyverse")

# Load libraries
library(eyelinker)
library(tidyverse)


#' Convert an Eyelink .asc file to standardized CSVs
#'
#' Reads an `.asc` file with `eyelinker::read.asc()` and writes three CSVs:
#' `{prefix}_gaze.csv`, `{prefix}_eyetracker.csv`, and `{prefix}_blinks.csv`.
#'
#' @param asc_path Character. Full path to the `.asc` file. Must end with `.asc`.
#' @param output_path Character. Directory where CSVs will be written. Created if missing.
#'
#' @details
#' * Gaze CSV joins raw samples with messages, renames common columns, and adds
#'   `participant_id` and zero-based `trial_number`.
#' * Eyetracker CSV copies device/session metadata and adds `participant_id`.
#' * Blinks CSV standardizes blink start/end/duration names and zero-bases trials.
#'
#' @return Invisibly writes CSV files to `output_path`; returns nothing.
#' @examples
#' # asc2csv("data/raw/train/P001/P001.asc", "data/clean/train/P001")
asc2csv = function(asc_path, output_path){
  # Validate input: only accept explicit `.asc` paths
  stopifnot(endsWith(asc_path, ".asc"))
  # Normalize output path to end with a slash
  output_path = ifelse(endsWith(output_path, "/"), output_path, paste0(output_path, "/"))
  # Derive file prefix from asc filename (without extension)
  file_prefix = str_split(tail(str_split(asc_path, "/")[[1]], 1), ".asc")[[1]][1]
  
  # Parse the .asc file into lists: raw samples, messages, blinks, info
  dat = read.asc(asc_path)
  
  # Create output directory if needed
  if (!dir.exists(output_path)){
    dir.create(output_path)
  }
  
  # ---- Gaze samples ----
  filename = paste0(output_path, file_prefix, "_gaze.csv")
  if (!file.exists(filename)){
    
    # Join samples with messages, standardize names, add identifiers, and write CSV
    dat$raw %>%
      left_join(dat$msg) %>%
      rename(
        trial_number = block,
        trial_time = time,
        gaze_x = xp,
        gaze_y = yp,
        pupil_size = ps,
        cornea_info = cr.info,
        message = text
        ) %>%
      add_column(participant_id = participant, .before = "trial_number") %>%
      mutate(trial_number = trial_number - 1) %>%
      write.csv(filename, row.names = FALSE)
  
  } else {
    print(paste(filename, "already exists!"))
  }

  # ---- Eyetracker/session metadata ----
  filename = paste0(output_path, file_prefix, "_eyetracker.csv")
  if (!file.exists(filename)){
    
    # Add participant id and write CSV
    dat$info %>%
      add_column(participant_id = participant, .before = "date") %>%
      write.csv(filename, row.names = FALSE)
    
  } else {
    print(paste(filename, "already exists!"))
  }
  
  # ---- Blinks ----
  filename = paste0(output_path, file_prefix, "_blinks.csv")
  if (!file.exists(filename)){
    
    # Standardize blink columns, zero-base trials, and write CSV
    dat$blinks %>%
      rename(
        trial_number = block,
        blink_start = stime,
        blink_end = etime,
        blink_duration = dur
      ) %>%
      add_column(participant_id = participant, .before = "trial_number") %>%
      mutate(trial_number = trial_number - 1) %>%
      write.csv(filename, row.names = FALSE)
    
    } else {
    print(paste(filename, "already exists!"))
  }
}


#' Clean the trials CSV exported by the experiment
#'
#' Removes unused columns, renames `section` to `trial_name`, and maps
#' compact trajectory labels (e.g., `ver_up`) to compass directions
#' (`north`, `south`, `east`, `west`, `northeast`, etc.).
#'
#' @param raw_data_path Character. Path to the raw `*_trials.csv` file.
#' @param output_path Character. Directory where the cleaned CSV is written.
#'
#' @return Writes a cleaned `trials.csv` to `output_path`; returns nothing.
cleanup_trials_csv = function(raw_data_path, output_path){
  # Sanity check: ensure correct file type
  stopifnot(endsWith(raw_data_path, "trials.csv"))
  # Normalize output path
  output_path = ifelse(endsWith(output_path, "/"), output_path, paste0(output_path, "/"))
  # Preserve original filename for output
  file_prefix = tail(str_split(raw_data_path, "/")[[1]], 1)
  
  # Load raw trials
  dat = read.csv(raw_data_path)
  filename = paste0(output_path, file_prefix)
  # Drop unused columns, standardize names, and map trajectories to compass terms
  dat %>% 
    select(!experiment_name) %>%
    rename(trial_name = section) %>%
    mutate(target_trajectory = case_when(
      target_trajectory == "ver_up" ~ "north",
      target_trajectory == "ver_down" ~ "south",
      target_trajectory == "hor_left" ~ "west",
      target_trajectory == "hor_right" ~ "east",
      target_trajectory == "diag_up_left" ~ "northwest",
      target_trajectory == "diag_up_right" ~ "northeast",
      target_trajectory == "diag_down_left" ~ "southwest",
      target_trajectory == "diag_down_right" ~ "southeast"
    )) %>%
    write.csv(filename, row.names = FALSE)
}


#' Clean the targets CSV exported by the experiment
#'
#' Selects and orders commonly used columns for target positions over time.
#'
#' @param raw_data_path Character. Path to the raw `*_targets.csv` file.
#' @param output_path Character. Directory where the cleaned CSV is written.
#'
#' @return Writes a cleaned `targets.csv` to `output_path`; returns nothing.
cleanup_targets_csv = function(raw_data_path, output_path){
  # Sanity check: ensure correct file type
  stopifnot(endsWith(raw_data_path, "targets.csv"))
  # Normalize output path
  output_path = ifelse(endsWith(output_path, "/"), output_path, paste0(output_path, "/"))
  # Preserve original filename for output
  file_prefix = tail(str_split(raw_data_path, "/")[[1]], 1)
  
  # Load raw targets
  dat = read.csv(raw_data_path)
  filename = paste0(output_path, file_prefix)
  # Keep only identifiers, time, and target coordinates
  dat %>% 
    select(!experiment_name) %>%
    select(participant_id,
           experiment_time,
           trial_number,
           trial_time,
           target_x,
           target_y) %>%
    write.csv(filename, row.names = FALSE)
}


#' Clean the participant CSV exported by the experiment
#'
#' Drops unused columns and preserves participant/session information.
#'
#' @param raw_data_path Character. Path to the raw `*_participant.csv` file.
#' @param output_path Character. Directory where the cleaned CSV is written.
#'
#' @return Writes a cleaned `participant.csv` to `output_path`; returns nothing.
cleanup_participant_csv = function(raw_data_path, output_path){
  # Sanity check: ensure correct file type
  stopifnot(endsWith(raw_data_path, "participant.csv"))
  # Normalize output path
  output_path = ifelse(endsWith(output_path, "/"), output_path, paste0(output_path, "/"))
  # Preserve original filename for output
  file_prefix = tail(str_split(raw_data_path, "/")[[1]], 1)
  
  # Load raw participant metadata
  dat = read.csv(raw_data_path)
  filename = paste0(output_path, file_prefix)
  # Remove experiment name and write cleaned file
  dat %>% 
    select(!experiment_name) %>%
    write.csv(filename, row.names=FALSE)
}

# ----------------------------------------------------------------------------
# Batch processing over train/test splits and participants
# ----------------------------------------------------------------------------
raw_data_path = "data/raw/"
output_data_path = "data/clean/"
for (train_test in c("train", "test")){
  # Iterate over dataset splits
  # Discover participant folders (non-recursive)
  super_folder = paste0(raw_data_path, train_test)
  participants = list.dirs(super_folder, full.names = FALSE, recursive = FALSE)
  for (participant in participants){
    # Prepare per-participant output path
    output_path = paste0(output_data_path, train_test, "/", participant)
    
    # Ensure split and participant directories exist
    if (!dir.exists(paste0(output_data_path, train_test))){
      dir.create(paste0(output_data_path, train_test), recursive = TRUE)
    }
    
    # Create directory if it doesnt exist
    if (!dir.exists(output_path)){
      dir.create(output_path, recursive = TRUE)
    }
    
    # If all expected outputs already exist, skip work for this participant
    files = list.files(output_path)
    if (all(c(paste0(participant, "_gaze.csv"), 
              paste0(participant, "_eyetracker.csv"), 
              paste0(participant, "_blinks.csv"),
              paste0(participant, "_targets.csv"),
              paste0(participant, "_trials.csv"),
              paste0(participant, "_participant.csv")) %in% files)) {
      print(paste("Skipping", participant, "because all files exist!"))
    
    } else {
      # Construct paths to source files
      asc_path = paste0(raw_data_path, train_test, "/", participant, "/", participant, ".asc")
      participant_csv_path = str_replace(asc_path, ".asc", "_participant.csv")
      targets_csv_path = str_replace(asc_path, ".asc", "_targets.csv")
      trials_csv_path = str_replace(asc_path, ".asc", "_trials.csv")
      
      # Convert EDF to ASC if the ASC is missing (requires Eyelink edf2asc in PATH)
      if (!file.exists(asc_path)){
        system_call = paste("edf2asc", str_replace(asc_path, ".asc", ".EDF"))
        system(system_call)
      }
      # Run conversion and cleaning steps
      asc2csv(asc_path, output_path)
      cleanup_participant_csv(participant_csv_path, output_path)
      cleanup_targets_csv(targets_csv_path, output_path)
      cleanup_trials_csv(trials_csv_path, output_path)
      }
  }
}
