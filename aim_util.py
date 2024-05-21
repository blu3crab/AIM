#
# AIM Utilities
import os
from google.colab import drive
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

################################################################
def write_preds_fileset(predictions, ordered_preds, word_preds, base_name):
    # base_name = 'citizen_1864_xform0_carlv'
    # aim_util.write_text(str(predictions), gdrive_path + base_name + "_bbox.txt")
    aim_util.write_text(str(ordered_preds), gdrive_path + base_name + "_word_list.txt")
    aim_util.write_text(word_preds, gdrive_path + base_name + "_word_string.txt")

    preds_df = preds_to_pd(predictions)

    pickle_path = gdrive_path + base_name + "_bbox.pkl"
    write_to_pickle(preds_df, pickle_path)


################################################################
import pickle
import numpy as np
import pandas as pd


def preds_to_pd(predictions):
    data_list = []

    for word, coords in predictions:
        upper_left_X, upper_left_Y, lower_left_X, lower_left_Y, lower_right_X, lower_right_Y, upper_right_X, upper_right_Y = coords.flatten()
        data_list.append(
            [word, upper_left_X, upper_left_Y, lower_left_X, lower_left_Y, lower_right_X, lower_right_Y, upper_right_X,
             upper_right_Y])

    df = pd.DataFrame(data_list,
                      columns=['word', 'upper_left_X', 'upper_left_Y', 'lower_left_X', 'lower_left_Y', 'lower_right_X',
                               'lower_right_Y', 'upper_right_X', 'upper_right_Y'])
    df = df.astype({col: np.float32 for col in df.columns[1:]})

    return df
################################################################

def write_to_pickle(df, path):
    """Write the DataFrame to a Pickle file (*.pkl) at the specified path"""
    with open(path, 'wb') as f:
        pickle.dump(df, f)
        print(f"Dataframe successfully written to: {path}")

def read_from_pickle(path):
    """Read the Pickle file into a Pandas dataframe"""
    with open(path, 'rb') as f:
        return pickle.load(f)

################################################################
# import json
import numpy as np
from google.colab import files

def to_json(basename, preds):
    # Convert numpy arrays to lists as JSON can't handle numpy arrays
    json_friendly_preds = []
    for key, val in preds:
      json_friendly_preds.append({
          'key': key,
          'val': val.tolist()
      })

    # Write to JSON file
    filename = basename + ".json"
    with open(filename, 'w') as f:
      json.dump(json_friendly_preds, f, indent=4)


    files.download(filename)
################################################################
# from google.colab import files
# files.download('example.txt')
################################################################
import ast
# groundtruth_path = '/content/gdrive/MyDrive/AIM/citizen_1864_groundtruth_word_list_lower.txt'
def file_to_list(path):
    with open(path, 'r') as f:
        target_list = ast.literal_eval(f.read())
    return target_list
# groundtruth_list = file_to_list(groundtruth_path)
##############################################

def show_image(image_path):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    return img


def mount_gdrive(gdrive_path="'content/gdrive"):
    # mount Google Drive
    drive.mount(gdrive_path)


def mount_gdrive(gdrive_path="'content/gdrive", image_path="MyDrive/ha-image/original_image_full.jpg"):
    # mount Google Drive
    drive.mount(gdrive_path)

    # image file path on the shared drive
    #image_path = '/content/gdrive/MyDrive/ha-image/original_image_full.jpg'

    show_image(gdrive_path + image_path)

# from google.colab import drive
# import sys
# drive.mount('/content/gdrive')
# sys.path.append('/content/gdrive/MyDrive/AIM')
# !ls -l '/content/gdrive/MyDrive/AIM'
# import aim_util

def write_text(text_string, file_path):
    # Check if the directory exists, if not, create it
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Open the file in write mode and write the text string
    with open(file_path, "w") as file:
        file.write(text_string)

    print(f"Text string of {len(text_string)} bytes successfully written to {file_path}.")
################################################################
