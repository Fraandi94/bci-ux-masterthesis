import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mne


# plot each channel of interest for checking the eeg-quality
def plot_raw_data(raw_data, channel_names):

    eeg_data, eeg_times = raw_data[:, :]

    channels_of_interest = {"Fp1": 0, "Fp2": 1, "F3": 2, "F4": 3, "F7": 6, "F8": 7}

    for n in range(len(channels_of_interest)):
        fig = plt.figure(figsize=(15, 7), dpi=100)
        plt.plot(eeg_times, eeg_data[channels_of_interest[channel_names[n]]].T)
        plt.xlabel('Zeit (Sekunden)')
        plt.ylabel('Potential (mikroVolt)')
        plt.title(channel_names[n])
        plt.show()


# read bdf-File and return the raw data
def read_bdf_file(filename):

    data = mne.io.read_raw_edf(DATA_PATH + filename, preload=True)

    return data


# only the channels Fp1, Fp2, F3, F4, F7, F8 are needed for this experiment
def pick_necessary_channels(raw_data_copy, channels):

    selected_channels = raw_data_copy.pick_channels(channels)

    return selected_channels


# all null-/ NaN-values have to be replaced with interpolated values
def clean_smarting_bdf_data(data):

    data_items, times = data[:, :]

    # create a pandas DataFrame for the recorded eeg data
    df_data = create_pandas_df(data_items.T, times, data.info["ch_names"])

    # check for null-/NaN-values and clean them (using interpolate-method)
    cleaned_data = clean_missing_values(df_data)

    # when true, df_data had null-/Nan-values and was cleaned
    if cleaned_data is not None:
        df_data = cleaned_data

    # create new RawArray
    new_raw_data = create_new_raw_array(df_data.T.values, data.info)

    return new_raw_data


def clean_missing_values(dataframe):

    # return if no NaN is in the data
    if not dataframe.isnull().any().any():
        return

    else:
        print("Number of NaN:", dataframe.isnull().sum())
        print(dataframe.select_dtypes(exclude=[np.number]))

        # interpolate missing values
        cleaned_data_frame = dataframe.interpolate(method="linear")

        return cleaned_data_frame


# create pandas data-frame for the mne raw data
def create_pandas_df(data, times, columns):

    return pd.DataFrame(data, index=times, columns=columns)


# create new RawArray
def create_new_raw_array(data, info):

    return mne.io.RawArray(data, info)



#   calculate a DC correction of the eeg data by substracting the mean of each value for each channel
def eeg_data_dc_correction(eeg_data):

    for n in range(len(eeg_data)):
        eeg_data[n] = eeg_data[n] - np.mean(eeg_data[n])

    return eeg_data


# use MNE's build in notch-filter to remove the frequencies of 50 Hz and 125 Hz from the data
def apply_notch_filter(eeg_data):

    filtered_data = mne.filter.notch_filter(eeg_data, Fs=500, freqs=[50, 125], filter_length="10s", copy=True)

    return filtered_data


# use MNE's build in bandpass-filter to remove artifacts, to clean the signal and to get the data in the desired frequency-range
def apply_bandpass_filter(eeg_data):

    filtered_data = mne.filter.band_pass_filter(eeg_data, Fs=500, Fp1=1, Fp2=30.0, filter_length="10s", method="fft", copy=True)

    return filtered_data


# use MNE's build in power spectral density function according to Welch's method to calculate the powers of alpha- and beta-band
# the eeg data gets transformed into the frequency domain
def apply_psd_welch_transformation(data, n_fft):

    alpha_psd_data, alpha_freqs = mne.time_frequency.psd_welch(data, fmin=8.0, fmax=12.0, n_fft=n_fft)

    beta_psd_data, beta_freqs = mne.time_frequency.psd_welch(data, fmin=12.0, fmax=29.3, n_fft=n_fft)

    return alpha_psd_data, alpha_freqs, beta_psd_data, beta_freqs



# get psd data for each specific channel
def get_psd_channel_data(psd_data):

    fp1 = psd_data[0]
    fp2 = psd_data[1]
    f3 = psd_data[2]
    f4 = psd_data[3]
    f7 = psd_data[4]
    f8 = psd_data[5]

    return {"Fp1": fp1, "Fp2": fp2, "F3": f3, "F4": f4, "F7": f7, "F8": f8}



# calculate the frontal alpha asymmetry index (faa) for the psd data of the six channels
def calculate_faa_index(psd_channel_data):

    right_hemispheric_power = psd_channel_data["Fp2"] + psd_channel_data["F4"] + psd_channel_data["F8"]

    left_hemispheric_power = psd_channel_data["Fp1"] + psd_channel_data["F3"] + psd_channel_data["F7"]

    faa_index = (np.log(right_hemispheric_power) - np.log(left_hemispheric_power))

    return faa_index



# calculate the valence index for the eeg data
def calculate_valence(alpha_data, beta_data):

    valence_index = (np.mean(alpha_data["F4"]) / np.mean(beta_data["F4"])) - (np.mean(alpha_data["F3"]) / np.mean(beta_data["F3"]))

    return valence_index



# calculate the arousal index for the eeg data
def calculate_arousal(alpha_data, beta_data):

    arousal_index = (np.mean(beta_data["F3"]) + np.mean(beta_data["F4"])) / (np.mean(alpha_data["F3"]) + np.mean(alpha_data["F4"]))

    return arousal_index




if __name__ == "__main__":

    DATA_PATH = "../data/MT_data/"

    filename = "tp8-3_C1.bdf"

    # eeg-channels of interest in this study
    eeg_channels_of_interest = ["Fp1", "Fp2", "F3", "F4", "F7", "F8"]


    # load bdf-file
    original_raw_data = read_bdf_file(filename)


    # for TP3 and task2 there are two files containing the task-data
    # so this to files have to be concatenated
    # filename_second_file = tp3-3_B1_2.bdf"
    # original_raw_data_second_file = read_bdf_file(filename_second_file)
    # original_raw_data.append(original_raw_data_second_file)


    # plot channels of interest for quality checking
    plot_raw_data(original_raw_data, eeg_channels_of_interest)


    # calculate and save a duration of 5 seconds depending on eeg-record frequency
    five_seconds = int(original_raw_data.info["sfreq"] * 5)


    # drop unnecessary gyro- und accelerometer channels
    original_raw_data.drop_channels(["GyroX", "GyroY", "GyroZ", "AccelerX", "AccelerY", "AccelerZ"])


    # get raw data of desired channels only
    # .copy() - function is used to not overwrite original raw data
    eeg_channels_of_interest_raw_data = pick_necessary_channels(original_raw_data.copy(), eeg_channels_of_interest)



    #**********************
    #
    # Preprocessing
    #
    #**********************

    # clean eeg data in case of NaN-values or missing values
    cleaned_eeg_channels_of_interest_raw_data = clean_smarting_bdf_data(eeg_channels_of_interest_raw_data.copy())


    # calculate the last five seconds of the recorded eeg-file
    last_five_seconds = int(cleaned_eeg_channels_of_interest_raw_data.n_times - five_seconds)


    # get eeg data and timestamps from raw object for the intervall starting at sec 5 until five seconds before the end of the recording
    eeg_data, eeg_times = cleaned_eeg_channels_of_interest_raw_data[:, five_seconds:last_five_seconds]


    # calculate DC correction of the eeg data
    dc_corrected_data = eeg_data_dc_correction(eeg_data)


    # notch-filter the data
    notch_filtered_data = apply_notch_filter(dc_corrected_data)


    # bandpass-filter the data for removing artifacts and cleaning the signal
    bandpass_filtered_data = apply_bandpass_filter(notch_filtered_data)


    # create pandas dataframe of the bandpass-filtered data in order to create a raw array of the filtered data
    df_bandpass_eeg_data = create_pandas_df(bandpass_filtered_data.T, eeg_times, cleaned_eeg_channels_of_interest_raw_data.info["ch_names"])


    # create new raw array of the preprocessed data in order to be able to use the data for feature extraction
    preprocessed_data_raw = create_new_raw_array(df_bandpass_eeg_data.T, cleaned_eeg_channels_of_interest_raw_data.info)



    # **********************
    #
    # Feature extraction
    #
    # **********************

    #*****
    # continous i.e. raw data


    # calculate the power spectral density values for alpha and beta band
    alpha_psd_data, alpha_freqs, beta_psd_data, beta_freqs = apply_psd_welch_transformation(preprocessed_data_raw, 2048)


    # get alpha data for each channel
    alpha_psd_channel_data = get_psd_channel_data(alpha_psd_data)

    # get beta data for each channel
    beta_psd_channel_data = get_psd_channel_data(beta_psd_data)


    # get the faa-index for the alpha psd values
    raw_faa_index = calculate_faa_index(alpha_psd_channel_data)


    # calculate valence-index for raw data
    raw_valence = calculate_valence(alpha_psd_channel_data, beta_psd_channel_data)


    # calculate arousal-index for raw data
    raw_arousal = calculate_arousal(alpha_psd_channel_data, beta_psd_channel_data)


    # print necessary values
    print("\n---------------")
    print("Raw faa index: \t", np.mean(raw_faa_index), "\tSD: ", np.std(raw_faa_index))
    print("Valence raw: \t", np.mean(raw_valence))
    print("Arousal raw: \t", np.mean(raw_arousal))
