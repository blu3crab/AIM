###############################################################################
# AIM Overlayer - overlay image with predictions for words and bounding boxes.
#
import pickle
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%matplotlib inline

def overlay_preds(image_path, preds_path, overlay_image_path, font_path, raw_preds=True, bbox_color=(0, 0, 255), text_color=(0, 255, 0)):
    # Load the pickle file
    with open(preds_path, 'rb') as f:
        preds_df = pickle.load(f)
    print(preds_df.head())

    # Convert the data to a pandas DataFrame
    #preds_df = pd.DataFrame(preds, columns=['word', 'upper_left_X', 'upper_left_Y', 'lower_left_X', 'lower_left_Y', 'lower_right_X', 'lower_right_Y', 'upper_right_X', 'upper_right_Y'])

    # Filter words with less than 3 characters
    preds_df = preds_df[preds_df.word.str.len() >= 3]

    print("Remaining list of words:")
    print(preds_df.word.tolist())

    # Load the image using PIL
    img = Image.open(image_path)

    # Create a new copy of the image for overlay
    overlay = img.copy()

    # Set the font and size for the text overlay
    font = ImageFont.truetype(font_path, 30)

    fig, (ax1) = plt.subplots(1, 1, figsize=(15, 15))
    ax1.set_title("Bounding Boxes and Words Overlay")
    sns.set_style("white")

    count_overlays = 0

    # Iterate over the rows of the DataFrame
    for idx, row in preds_df.iterrows():
        # Extract the bounding box coordinates and word
        x1, y1, text = row['UL_X'], row['UL_Y'], row['word']
        #x2, y2 = row['LL_X'], row['LL_Y']
        x3, y3 = row['LR_X'], row['LR_Y']
        #x4, y4 = row['UR_X'], row['UR_Y']

        if raw_preds:
            # raw preds include the lower-left and upper-right coords
            x2, y2 = row['LL_X'], row['LL_Y']
            x4, y4 = row['UR_X'], row['UR_Y']
            print(f"x2, y2->{x2}, {y2}")
            print(f"x4, y4->{x2}, {y2}")

        else:
            # calulate the lower-left and upper-right coords
            x2, y2 = row['UL_X'], row['LR_Y']
            x4, y4 = row['LR_X'], row['UL_Y']

        # Calculate the bounding box rectangle to draw.
        min_x, min_y = min(x1, x2, x3, x4), min(y1, y2, y3, y4)
        max_x, max_y = max(x1, x2, x3, x4), max(y1, y2, y3, y4)

        draw = ImageDraw.Draw(overlay)
        draw.rectangle((min_x, min_y, max_x, max_y), outline=(0, 255, 0))

        # Get the text bounding box to ensure it fits inside the rectangle
        text_bbox = draw.textbbox((min_x, min_y), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        # Adjust the text position to center it within the bounding box
        if max_x - min_x > text_width and max_y - min_y > text_height:
            text_x = min_x + ((max_x - min_x) / 2) - (text_width / 2)
            text_y = min_y + ((max_y - min_y) / 2) - (text_height / 2)
            draw.text((text_x, text_y), text, (0, 0, 255), font=font)
            count_overlays += 1

    # Display the imaged with the overlays
    ax1.imshow(overlay)
    plt.show()

    print(f"Count of bounding box/word overlays: {count_overlays}")
    overlay.save(overlay_image_path)