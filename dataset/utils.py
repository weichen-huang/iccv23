import pandas as pd
import numpy as np
import argparse
import time
import os

from sklearn.model_selection import train_test_split
from collections import Counter
from logger import log
import nibabel as nib
import cv2

from tqdm import tqdm
from datetime import datetime


def get_volume(filepath):
    """
    Extract a NIfTI volume from the file at the given filepath and return it as a 3D numpy array.

    Parameters:
    filepath (str): The path to the NIfTI file.

    Returns:
    np.ndarray: A 3D numpy array containing the volume.
    """
    # Load the NIfTI file using nibabel
    nifti_img = nib.load(filepath)

    # Get the data from the NIfTI file as a numpy array
    volume_data = nifti_img.get_fdata()

    # Ensure the volume data is 3D
    if len(volume_data.shape) == 4:
        # If the data has a 4th dimension (like time or multiple volumes),
        # you might want to select one of the volumes. Here, we assume the first.
        volume_data = volume_data[:, :, :, 0]

    return volume_data


def get_subpaths(path):
    """
    Get all subpaths of a folder, including subdirectories and files.

    Parameters:
    path (str): The path to the directory whose subpaths are to be found.

    Returns:
    list: A list of all subpaths.
    """
    subpaths = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            subpaths.append(os.path.join(root, name))
        for name in files:
            subpaths.append(os.path.join(root, name))

    new_subpaths = []

    for subpath in subpaths:
        if subpath.endswith('.nii'):
            tmp = subpath.split('/')
            img_id = tmp[-2].replace("I", "")
            visit_time = tmp[-3][:10]
            patient_id = tmp[-5]
            new_subpaths.append([subpath, img_id, visit_time, patient_id])

    return new_subpaths


def get_row_by_imageuid(df, imageuid):
    """
    Get the row from a CSV file where the column 'IMAGEUID_UCSFFSX_11_02_15_UCSFFSX51_08_01_16' matches a given value.

    Parameters:
    filepath (str): The file path to the CSV file containing the data.
    imageuid (str or int): The value to match in the 'IMAGEUID_UCSFFSX_11_02_15_UCSFFSX51_08_01_16' column.

    Returns:
    pd.Series or None: The corresponding row as a pandas Series, or None if no match is found.
    """

    # Find the row where the column value matches the given imageuid
    row = df[df['IMAGEUID_UCSFFSX_11_02_15_UCSFFSX51_08_01_16'].str.contains(imageuid, na=False)]

    # If a row is found, return it as a pandas Series. If not, return None.
    if not row.empty:
        return row.iloc[0].to_dict()
    else:
        return None


def get_row_by_examdate_and_ptid(df, examdate_substring, ptid_substring):
    """
    Get the row from a CSV file where the 'EXAMDATE' column contains a given examdate substring
    and the 'PTID' column contains a given ptid substring.

    Parameters:
    filepath (str): The file path to the CSV file containing the data.
    examdate_substring (str): The substring to search for in the 'EXAMDATE' column.
    ptid_substring (str): The substring to search for in the 'PTID' column.

    Returns:
    pd.Series or None: The corresponding row as a pandas Series, or None if no match is found.
    """
    # Create a boolean mask where both conditions are met
    mask = (
            df['EXAMDATE'].str.contains(examdate_substring, na=False) &
            df['PTID'].str.contains(ptid_substring, na=False)
    )

    # Use the mask to filter the DataFrame
    filtered_df = df[mask]

    # If at least one row is found, return the first one as a pandas Series. If not, return None.
    if not filtered_df.empty:
        return filtered_df.iloc[0].to_dict()
    else:
        return None


def get_viscode(df, imageid):
    """
    Get the corresponding "Visit" for a given "Image Data ID" from a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The pandas DataFrame containing the data.
    imageid (str or int): The "Image Data ID" to match.

    Returns:
    str or None: The corresponding "Visit" as a string, or None if no match is found.
    """
    # Find the row where the "Image Data ID" column matches the given imageid
    matched_row = df[df['Image Data ID'] == imageid]

    # If a row is found, return the "Visit" value. If not, return None.
    if not matched_row.empty:
        return matched_row['Visit'].iloc[0]
    else:
        return None


def get_row_by_ptid_and_viscode(df, ptid, viscode):
    """
    Get the row from a pandas DataFrame where the 'PTID' matches a given patient ID
    and the 'VISCODE' matches a given visit code.

    Parameters:
    df (pd.DataFrame): The pandas DataFrame containing the data.
    ptid (str): The patient ID to match in the 'PTID' column.
    viscode (str): The visit code to match in the 'VISCODE' column.

    Returns:
    pd.Series or None: The corresponding row as a pandas Series, or None if no match is found.
    """
    # Filter the DataFrame for the given PTID and VISCODE
    matched_row = df[(df['PTID'] == ptid) & (df['VISCODE'] == viscode)]

    # If a row is found, return it as a pandas Series. If not, return None.
    if not matched_row.empty:
        return matched_row.iloc[0].to_dict()
    else:
        return None


def process_row(dict):
    """
    Process a dictionary and returns the cleaned data.

    Parameters:
    dict (dict): The dictionary

    Returns:
    dict: The cleaned dictionary
    """

    new_dict = {}
    dx_mapping = {"NL": 0.0, "MCI": 1.0, "Dementia": 2.0}

    for k, v in dict.items():
        if k == "path":
            new_dict[k] = v
            continue
        if k == "EXAMDATE":
            date_object = datetime.strptime(v, '%Y-%m-%d')
            current_date = datetime.now()
            days_difference = float((current_date - date_object).days)
            new_dict[k] = days_difference
        if k == 'DX':
            new_dict[k] = dx_mapping[v]
        else:
            try:
                new_dict[k] = float(v)
            except:
                new_dict[k] = 0.0

            if np.isnan(new_dict[k]):
                new_dict[k] = 0.0
    return new_dict


def get_tabular_data(verbose=False):
    if verbose:
        log('Testing get_tabular_data()')
        # log current dir
        log('Current dir: {}'.format(os.getcwd()))
    # chdir to ../
    os.chdir('../')

    table = pd.read_csv("data/tadpole_challenge/TADPOLE_D1_D2.csv")
    metadata = pd.read_csv("data/adni/metadata.csv")

    if verbose:
        log('Table shape: {}'.format(table.shape))
        log('Table columns: {}'.format(str(list(table.columns)[:30])))

        log('Table VISCODE column: {}'.format(table['VISCODE'].head()))

    # get all subpaths

    subpaths = get_subpaths("data/adni/ADNI")

    if verbose:
        log('Subpaths: {}'.format(len(subpaths)))
        log('Subpaths[0]: {}'.format(subpaths[0]))

    patient_dict = {}

    relevant_keys = ["EXAMDATE",
                     "DX", "ADAS13", "Ventricles",
                     "CDRSB", "ADAS11", "MMSE", "RAVLT_immediate",
                     "Hippocampus", "WholeBrain", "Entorhinal", "MidTemp",
                     "FDG", "AV45",
                     "ABETA_UPENNBIOMK9_04_19_17", "TAU_UPENNBIOMK9_04_19_17", "PTAU_UPENNBIOMK9_04_19_17",
                     "APOE4", "AGE"]

    ok_cnt = 0

    dim0 = dim1 = dim2 = 1e9

    for subpath in tqdm(subpaths):
        patient_id = subpath[3]
        exam_date = subpath[2]
        image_id = subpath[1]
        image_path = subpath[0]
        if verbose:
            log("Trying get_volume for image_path: {}".format(image_path))
        try:
            volume = get_volume(image_path)
            if verbose:
                log(volume.shape)
            shape = sorted(list(volume.shape))
            dim0 = min(dim0, shape[0])
            dim1 = min(dim1, shape[1])
            dim2 = min(dim2, shape[2])
        except:
            if verbose:
                log('Failed to get volume for image_path: {}... skipping'.format(image_path), mode="ERROR")
            continue

        patient = get_row_by_imageuid(table, image_id)
        if patient is None:
            if verbose:
                log('Patient not found for image_id: {}... trying patient ID + exam date'.format(image_id), mode="WARN")
            patient = get_row_by_examdate_and_ptid(table, exam_date, patient_id)
            if patient is None:
                if verbose:
                    log('Patient not found for patient ID + exam date: {} + {}... trying patient ID + viscode'.format(
                        patient_id, exam_date), mode="ERROR")
                viscode = get_viscode(metadata, "I" + image_id)
                if viscode is None:
                    if verbose:
                        log('Viscode not found for image_id: {}... skipping'.format(image_id), mode="ERROR")
                    continue
                patient = get_row_by_ptid_and_viscode(table, patient_id, viscode)
                if patient is None:
                    if verbose:
                        log('Patient not found for patient ID + viscode: {} + {}... skipping'.format(patient_id,
                                                                                                     viscode),
                            mode="ERROR")
                    continue
        patient_id = patient['PTID']

        if patient_id not in patient_dict:
            patient_dict[patient_id] = []

        cur = {"path": image_path, "image_id": image_id}

        for key in relevant_keys:
            cur[key] = patient[key]

        cur = process_row(cur)

        patient_dict[patient_id].append(cur)
        ok_cnt += 1

    for patient_id in patient_dict:
        patient_dict[patient_id] = sorted(patient_dict[patient_id], key=lambda x: x['EXAMDATE'])
        min_examdate = patient_dict[patient_id][0]['EXAMDATE']
        for i in range(len(patient_dict[patient_id])):
            patient_dict[patient_id][i]['EXAMDATE'] = patient_dict[patient_id][i]['EXAMDATE'] - min_examdate + i


    log('Succeeded/total/patient dict: {}/{}/{}'.format(ok_cnt, len(subpaths), len(patient_dict)))
    # log(patient_dict["082_S_5014"][0]["path"])
    if verbose:
        for i in range(10):
            log('Patient dict[]: {}'.format(patient_dict[list(patient_dict.keys())[i]]))
        log("Image dimensions: {}, {}, {}".format(dim0, dim1, dim2))
    return patient_dict, (dim0, dim1, dim2)


def get_dynamic_image(frames, normalized=True):
    """ Adapted from https://github.com/tcvrick/Python-Dynamic-Images-for-Action-Recognition"""
    """ Takes a list of frames and returns either a raw or normalized dynamic image."""

    def _get_channel_frames(iter_frames, num_channels):
        """ Takes a list of frames and returns a list of frame lists split by channel. """
        frames = [[] for channel in range(num_channels)]

        for frame in iter_frames:
            for channel_frames, channel in zip(frames, cv2.split(frame)):
                channel_frames.append(channel.reshape((*channel.shape[0:2], 1)))
        for i in range(len(frames)):
            frames[i] = np.array(frames[i])
        return frames

    def _compute_dynamic_image(frames):
        """ Adapted from https://github.com/hbilen/dynamic-image-nets """
        num_frames, h, w, depth = frames.shape

        # Compute the coefficients for the frames.
        coefficients = np.zeros(num_frames)
        for n in range(num_frames):
            cumulative_indices = np.array(range(n, num_frames)) + 1
            coefficients[n] = np.sum(((2 * cumulative_indices) - num_frames) / cumulative_indices)

        # Multiply by the frames by the coefficients and sum the result.
        x1 = np.expand_dims(frames, axis=0)
        x2 = np.reshape(coefficients, (num_frames, 1, 1, 1))
        result = x1 * x2
        return np.sum(result[0], axis=0).squeeze()

    num_channels = frames[0].shape[2]
    # print(num_channels)
    channel_frames = _get_channel_frames(frames, num_channels)
    channel_dynamic_images = [_compute_dynamic_image(channel) for channel in channel_frames]

    dynamic_image = cv2.merge(tuple(channel_dynamic_images))
    if normalized:
        dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        dynamic_image = dynamic_image.astype('uint8')

    return dynamic_image


def get_video_frames(video_path):
    # Initialize the frame number and create empty frame list
    video = cv2.VideoCapture(video_path)
    frame_list = []

    # Loop until there are no frames left.
    try:
        while True:
            more_frames, frame = video.read()

            if not more_frames:
                break
            else:
                frame_list.append(frame)

    finally:
        video.release()

    return frame_list

def transpose_volume(tensor):
    sorted_indices = np.argsort(tensor.shape)
    new_tensor = tensor.transpose(sorted_indices)
    # expand first dimension by 1 to [1, x, y, z]
    new_tensor = np.expand_dims(new_tensor, axis=0)
    return new_tensor

def get_frames(array):
    array = get_dynamic_image(array)
    array = np.expand_dims(array, 0)
    array = np.concatenate([array, array, array], 0)
    return array

def slice_volume(input_volume, x, y, z):
    """
    Slice both sides of a volume to obtain a new volume with dimensions [x, y, z].

    Parameters:
    - input_volume: The input 3D volume.
    - x, y, z: The desired dimensions for the sliced volume.

    Returns:
    - sliced_volume: The sliced 3D volume.
    """
    # Determine the starting indices for slicing
    start_x = (input_volume.shape[1] - x) // 2
    start_y = (input_volume.shape[2] - y) // 2
    start_z = (input_volume.shape[3] - z) // 2

    # Create slices for each dimension
    x_slice = slice(start_x, start_x + x)
    y_slice = slice(start_y, start_y + y)
    z_slice = slice(start_z, start_z + z)

    # Slice the volume using the defined slices
    sliced_volume = input_volume[:, x_slice, y_slice, z_slice]

    return sliced_volume

def one_hot_encode(value, num_classes):
    """
    Perform one-hot encoding on a given value.

    Parameters:
    - value: The categorical value to be one-hot encoded.
    - num_classes: The total number of classes/categories.

    Returns:
    - one_hot_vector: The one-hot encoded vector.
    """
    # Initialize an array of zeros with the length equal to the number of classes
    one_hot_vector = np.zeros(num_classes, dtype=int)

    # Set the corresponding index to 1 for the given value
    one_hot_vector[value] = 1

    return one_hot_vector
