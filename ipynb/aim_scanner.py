###############################################################################
# AIM scanner
import pickle
import numpy as np
import pandas as pd

import keras_ocr
import math

import aim_util

###############################################################################
# keras OCR
# !pip install keras_ocr

def get_pipeline():
    pipeline_OCR = keras_ocr.pipeline.Pipeline()
    return pipeline_OCR

def Detect(image_path,pipeline):
    """OCR for text detection"""


    # Read in image path
    read_image = keras_ocr.tools.read(image_path)
    # prediction_groups is a list of (word, box) tuples
    prediction_groups = pipeline.recognize([read_image])
    return prediction_groups[0]

def Distance(predictions):
    """
    Returns dictionary with (key,value):

    """

    # Point of origin
    x0, y0 = 0, 0
    # Generate dictionary
    detections = []
    for group in predictions:

        # Get center point of bounding box
      top_left_x, top_left_y = group[1][0]
      bottom_right_x, bottom_right_y = group[1][1]
      center_x = (top_left_x + bottom_right_x) / 2
      center_y = (top_left_y + bottom_right_y) / 2
      # Use the Pythagorean Theorem to solve for distance from origin
      distance_from_origin = math.dist([x0,y0], [center_x, center_y])
      # Calculate difference between y and origin to get unique rows
      distance_y = center_y - y0
      # Append all results
      detections.append({
                          'text':group[0],
                          'center_x':center_x,
                          'center_y':center_y,
                          'distance_from_origin':distance_from_origin,
                          'distance_y':distance_y
                      })
    return detections

def distinguish_rows(lst, thresh):
    """Function to help distinguish unique rows"""

    sublists = []
    for i in range(0, len(lst)-1):
        if lst[i+1]['distance_y'] - lst[i]['distance_y'] <= thresh:
            if lst[i] not in sublists:
                sublists.append(lst[i])
            sublists.append(lst[i+1])
        else:
            yield sublists
            sublists = [lst[i+1]]
    yield sublists


def OCR(image_path, pipeline, order='yes',thresh=6):
    """
    Function returns predictions in human readable order
    from left to right & top to bottom
    """
    ordered_preds = []
    predictions = Detect(image_path, pipeline)
    #print(f"predictions (raw)->{predictions}")
    predictions_2 = Distance(predictions)
    #print(f"predictions_2->{predictions_2}")
    longitud=len(predictions_2)
    if longitud==1: ordered_preds= predictions_2[0]['text']
    else:

      predictions_3 = list(distinguish_rows(predictions_2, thresh))
      #print(f"predictions_3->{predictions_3}")
      # Remove all empty rows
      predictions_3_f = list(filter(lambda x:x!=[], predictions_3))
      #print(f"predictions_3_f->{predictions_3_f}")
      # Order text detections in human readable format

      ylst = ['yes', 'y']
      for pr in predictions_3_f:
          if order in ylst:
              row = sorted(pr, key=lambda x:x['distance_from_origin'])
              for each in row:
                  ordered_preds.append(each['text'])
    #print(f"ordered_preds->{ordered_preds}")

    with open('texto.txt','a+') as f:
      for word in ordered_preds:
        f.write(word+' ')
    text=''
    for word in ordered_preds:

      text=text+' '+word

    #print(f"text->{text}")
    return predictions, ordered_preds, text
###############################################################################
def preds_to_pd(predictions):
    data_list = []

    for word, coords in predictions:
        upper_left_X, upper_left_Y, lower_left_X, lower_left_Y, lower_right_X, lower_right_Y, upper_right_X, upper_right_Y = coords.flatten()
        data_list.append([word, upper_left_X, upper_left_Y, lower_left_X, lower_left_Y, lower_right_X, lower_right_Y, upper_right_X, upper_right_Y])

    df = pd.DataFrame(data_list, columns=['word', 'UL_X', 'UL_Y', 'LL_X', 'LL_Y', 'LR_X', 'LR_Y', 'UR_X', 'UR_Y'])
    df = df.astype({col: np.float32 for col in df.columns[1:]})

    return df

def write_to_pickle(df, path):
    """Write the DataFrame to a Pickle file (*.pkl) at the specified path"""
    with open(path, 'wb') as f:
        pickle.dump(df, f)
        print(f"Dataframe successfully written to: {path}")

def read_from_pickle(path):
    """Read the Pickle file into a Pandas dataframe"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def write_preds_fileset(predictions, ordered_preds, word_preds, gdrive_path, base_name):
    aim_util.write_text(str(ordered_preds), gdrive_path + base_name + "_word_list.txt")
    aim_util.write_text(word_preds, gdrive_path + base_name + "_word_string.txt")

    preds_df = preds_to_pd(predictions)

    pickle_path = gdrive_path + base_name + "_bbox.pkl"
    write_to_pickle(preds_df, pickle_path)
###############################################################################
def test_preds_to_pickle(predictions, gdrive_path, base_name):
    preds_df = preds_to_pd(predictions)
    print(f"gen preds->{preds_df.head()}")

    pickle_path = gdrive_path + base_name + "_bbox.pkl"

    preds_read_df = read_from_pickle(pickle_path)
    print(f"read preds->{preds_read_df.head()}")

    # Compare the two dataframes
    result = preds_df.equals(preds_read_df)
    print("Result of comparison:", result)
###############################################################################
