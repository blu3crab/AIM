####################################################################
from types import SimpleNamespace

tuner = SimpleNamespace(

    # input files paths
    FILE_PATH_BASE="/kaggle/input/hms-harmful-brain-activity-classification/",
    TRAIN_CSV_FILENAME="train.csv",
    TEST_CSV_FILENAME="test.csv",
    TRAIN_EEG_PATH="train_eegs/",
    TEST_EEG_PATH="test_eegs/",
    TRAIN_SPECTROGRAM_PATH="train_spectrograms/",
    TEST_SPECTROGRAM_PATH="test_spectrograms/",

    PARQUET_FILE_EXT=".parquet",

    # train csv column labels
    EEG_ID_COLUMN = 'eeg_id',
    EEG_SUB_ID_COLUMN = 'eeg_sub_id',
    # eeg_label_offset_seconds
    SPECTROGRAM_ID_COLUMN = "spectrogram_id",
    SPECTROGRAM_SUB_ID_COLUMN = "spectrogram_sub_id",
    # spectrogram_label_offset_seconds
    LABEL_ID_COLUMN = "label_id",
    PATIENT_ID_COLUMN = "patient_id",
    EXPERT_CONSENSUS_COLUMN = "expert_consensus",
    SEIZURE_VOTE_COLUMN = "seizure_vote",
    LPD_VOTE_COLUMN = "lpd_vote",
    GPD_VOTE_COLUMN = "gpd_vote",
    LRDA_VOTE_COLUMN = "lrda_vote",
    GRDA_VOTE_COLUMN = "grda_vote",
    OTHER_VOTE_COLUMN = "other_vote"
)
####################################################################