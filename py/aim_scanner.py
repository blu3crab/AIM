###############################################################################
# AIM scanner
import pickle
import numpy as np
import pandas as pd

import keras_ocr
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import aim_util

gdrive_path = '/content/gdrive/MyDrive/AIM/'

###############################################################################
# keras OCR
# !pip install keras_ocr

###############################################################################
def get_pipeline():
    pipeline_OCR = keras_ocr.pipeline.Pipeline()
    return pipeline_OCR

###############################################################################
###############################################################################
def Detect(image_path, pipeline):
    """OCR for text detection"""

    # Read in image path
    read_image = keras_ocr.tools.read(image_path)
    # prediction_groups is a list of (word, box) tuples
    prediction_groups = pipeline.recognize([read_image])
    return prediction_groups[0]

###############################################################################
def Distance(predictions):
    """
    Returns dictionary with (key,value):

    """
    # Point of origin
    x0, y0 = 0, 0
    # Generate dictionary
    detections = []
    # trace_count = 0
    for group in predictions:

        # Get center point of bounding box
      top_left_x, top_left_y = group[1][0]
      #bottom_right_x, bottom_right_y = group[1][1]
      bottom_right_x, bottom_right_y = group[1][2]
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
                          'distance_y':distance_y,
                          'top_left_x': top_left_x,
                          'top_left_y': top_left_y,
                          'bottom_right_x': bottom_right_x,
                          'bottom_right_y': bottom_right_y,
                      })
      # if trace_count < 4:
      #   print(f"{group}")
      #   print(f"top_left_x, top_left_y->{top_left_x}, {top_left_y}--bottom_right_x, bottom_right_y->{bottom_right_x}, {bottom_right_y}")
      #   trace_count = trace_count + 1
    return detections

###############################################################################
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

###############################################################################
def OCR(image_path, pipeline, trim_length=1, order='yes', thresh=6):
    """
    Function returns predictions in human readable order
    from left to right & top to bottom
    """
    ordered_word_list = []
    ordered_word_list_with_newline = []

    predictions_raw = Detect(image_path, pipeline)
    #print(f"predictions (raw)->{predictions_raw}")
    predictions_2 = Distance(predictions_raw)
    #print(f"predictions_2->{predictions_2}")
    longitud=len(predictions_2)
    if longitud==1:
        ordered_word_string = predictions_2[0]['text']
    else:

        predictions_3 = list(distinguish_rows(predictions_2, thresh))
        #print(f"predictions_3->{predictions_3}")
        # Remove all empty rows
        predictions_3_f = list(filter(lambda x:x!=[], predictions_3))
        #print(f"predictions_3_f->{predictions_3_f}")

        # Order text detections in human-readable format
        pred_sort_dist = []
        #breakout = False
        ylst = ['yes', 'y']
        for pr in predictions_3_f:
            #print(f"pr->{pr}")
            if order in ylst:
                row = sorted(pr, key=lambda x:x['distance_from_origin'])
                #print(f"sorted row->{row}")
                row_added = False
                for each in row:
                    if len(each['text']) > trim_length:
                        ordered_word_list.append(each['text'])
                        ordered_word_list_with_newline.append(each['text'])
                        #print(f"each->{each}, each-text->{each['text']}")

                        # append bbox preds
                        pred_sort_dist.append([each['text'], each['center_x'],each['center_y'],each['distance_from_origin'],each['distance_y'],each['top_left_x'],each['top_left_y'],each['bottom_right_x'],each['bottom_right_y']])
                        row_added = True
                if row_added:
                    ordered_word_list_with_newline.append("\n")
                # breakout = True
                # if breakout:
                #     break
        #print(pred_sort_dist)
        #print(f"ordered_word_string->{ordered_word_string}")

    print(f"ordered_word_list->{ordered_word_list}")
    print(f"ordered_word_list_with_newline->{ordered_word_list_with_newline}")

    # with open('texto.txt','a+') as f:
    #     for word in ordered_word_list:
    #         f.write(word + ' ')
    # word_list = ''
    ordered_word_string = ""
    ordered_word_string_with_newline =""
    for word in ordered_word_list:
        ordered_word_string = ordered_word_string + ' ' + word
    for word in ordered_word_list_with_newline:
        ordered_word_string_with_newline = ordered_word_string_with_newline + ' ' + word

    #print(f"ordered_preds_with_newline->{ordered_preds_with_newline}")
    return predictions_raw, pred_sort_dist, ordered_word_string, ordered_word_string_with_newline, ordered_word_list
###############################################################################
def predict(gdrive_path, transform_basename_list, pipeline, trim_length=1):
    for base_name in transform_basename_list:
        image_path = gdrive_path + base_name + ".jpg"
        img = mpimg.imread(image_path)
        plt.imshow(img)
        print(f"scanning {image_path}")
        predictions_raw, pred_sort_dist, ordered_word_string, ordered_word_string_with_newline, word_list = OCR(image_path, pipeline, trim_length, order='yes', thresh=16)
        write_preds_fileset(predictions_raw, pred_sort_dist, ordered_word_string, ordered_word_string_with_newline, word_list, gdrive_path, base_name)

    return predictions_raw, pred_sort_dist, ordered_word_string, ordered_word_string_with_newline, word_list
###############################################################################

def pred_to_pd(predictions, raw=True):
    data_list = []

    if raw:
        for word, coords in predictions:
            upper_left_X, upper_left_Y, lower_left_X, lower_left_Y, lower_right_X, lower_right_Y, upper_right_X, upper_right_Y = coords.flatten()
            data_list.append([word, upper_left_X, upper_left_Y, lower_left_X, lower_left_Y, lower_right_X, lower_right_Y, upper_right_X, upper_right_Y])

        df = pd.DataFrame(data_list, columns=['word', 'UL_X', 'UL_Y', 'LL_X', 'LL_Y', 'LR_X', 'LR_Y', 'UR_X', 'UR_Y'])
        df = df.astype({col: np.float32 for col in df.columns[1:]})
    else:
        for word, center_x, center_y, distance_from_origin, distance_y, top_left_x, top_left_y, bottom_right_x, bottom_right_y in predictions:
            data_list.append([word, center_x, center_y, distance_from_origin, distance_y, top_left_x, top_left_y, bottom_right_x, bottom_right_y])

        df = pd.DataFrame(data_list, columns=['word', 'CENTER_X', 'CENTER_Y', 'DIST_ORIGEN', 'DIST_Y', 'UL_X', 'UL_Y', 'LR_X', 'LR_Y'])
        df = df.astype({col: np.float32 for col in df.columns[1:]})

    #print(df.head())
    return df

###############################################################################
def write_to_pickle(df, path):
    """Write the DataFrame to a Pickle file (*.pkl) at the specified path"""
    with open(path, 'wb') as f:
        pickle.dump(df, f)
        print(f"Dataframe successfully written to: {path}")

def read_from_pickle(path):
    """Read the Pickle file into a Pandas dataframe"""
    with open(path, 'rb') as f:
        return pickle.load(f)

###############################################################################
def write_preds_fileset(predictions_raw, pred_sort_dist, ordered_word_string, ordered_word_string_with_newline, word_list, gdrive_path, base_name):

    pred_raw_df = pred_to_pd(predictions_raw, raw=True)
    pickle_path = gdrive_path + base_name + "_raw_bbox.pkl"
    write_to_pickle(pred_raw_df, pickle_path)

    pred_sort_dist_df = pred_to_pd(pred_sort_dist, raw=False)
    pickle_path = gdrive_path + base_name + "_sort_dist_bbox.pkl"
    write_to_pickle(pred_sort_dist_df, pickle_path)

    aim_util.write_text(ordered_word_string, gdrive_path + base_name + "_word_string.txt")
    aim_util.write_text(ordered_word_string_with_newline, gdrive_path + base_name + "_newline_word_string.txt")
    aim_util.write_text(str(word_list), gdrive_path + base_name + "_word_list.txt")
###############################################################################
def test_preds_to_pickle(predictions, gdrive_path, base_name, raw=True):
    preds_df = pred_to_pd(predictions, raw)
    print(f"gen preds->{preds_df.head()}")

    pickle_path = gdrive_path + base_name + "_bbox.pkl"

    preds_read_df = read_from_pickle(pickle_path)
    print(f"read preds->{preds_read_df.head()}")

    # Compare the two dataframes
    result = preds_df.equals(preds_read_df)
    print("Result of comparison:", result)
###############################################################################
