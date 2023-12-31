"""
visualization_utils.py

Core rendering functions shared across visualization scripts
"""

#%% Constants and imports
import os
import os.path as osp
import shutil

from io import BytesIO
from typing import Union
import time

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image, ImageFile, ImageFont, ImageDraw

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_ROTATIONS = {
    3: 180,
    6: 270,
    8: 90
}

TEXTALIGN_LEFT = 0
TEXTALIGN_RIGHT = 1


# Retry on blob storage read failures
n_retries = 10
retry_sleep_time = 0.01
error_names_for_retry = ['ConnectionError']
                

#%% Functions

def open_image(input_file: Union[str, BytesIO]) -> Image:
    """
    Opens an image in binary format using PIL.Image and converts to RGB mode.
    
    Supports local files or URLs.

    This operation is lazy; image will not be actually loaded until the first
    operation that needs to load it (for example, resizing), so file opening
    errors can show up later.

    Args:
        input_file: str or BytesIO, either a path to an image file (anything
            that PIL can open), or an image as a stream of bytes

    Returns:
        an PIL image object in RGB mode
    """
    if (isinstance(input_file, str)
            and input_file.startswith(('http://', 'https://'))):
        try:
            response = requests.get(input_file)
        except Exception as e:
            print(f'Error retrieving image {input_file}: {e}')
            success = False
            if e.__class__.__name__ in error_names_for_retry:
                for i_retry in range(0,n_retries):
                    try:
                        time.sleep(retry_sleep_time)
                        response = requests.get(input_file)        
                    except Exception as e:
                        print(f'Error retrieving image {input_file} on retry {i_retry}: {e}')
                        continue
                    print('Succeeded on retry {}'.format(i_retry))
                    success = True
                    break
            if not success:
                raise
        try:
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            print(f'Error opening image {input_file}: {e}')
            raise

    else:
        image = Image.open(input_file)
    if image.mode not in ('RGBA', 'RGB', 'L', 'I;16'):
        raise AttributeError(
            f'Image {input_file} uses unsupported mode {image.mode}')
    if image.mode == 'RGBA' or image.mode == 'L':
        # PIL.Image.convert() returns a converted copy of this image
        image = image.convert(mode='RGB')

    # Alter orientation as needed according to EXIF tag 0x112 (274) for Orientation
    #
    # https://gist.github.com/dangtrinhnt/a577ece4cbe5364aad28
    # https://www.media.mit.edu/pia/Research/deepview/exif.html
    #
    try:
        exif = image._getexif()
        orientation: int = exif.get(274, None)  # 274 is the key for the Orientation field
        if orientation is not None and orientation in IMAGE_ROTATIONS:
            image = image.rotate(IMAGE_ROTATIONS[orientation], expand=True)  # returns a rotated copy
    except Exception:
        pass

    return image


def load_image(input_file: Union[str, BytesIO]) -> Image:
    """
    Loads the image at input_file as a PIL Image into memory.

    Image.open() used in open_image() is lazy and errors will occur downstream
    if not explicitly loaded.

    Args:
        input_file: str or BytesIO, either a path to an image file (anything
            that PIL can open), or an image as a stream of bytes

    Returns: PIL.Image.Image, in RGB mode
    """
    image = open_image(input_file)
    image.load()
    return image

COLORS = [
    'AliceBlue', 'Red', 'RoyalBlue', 'Gold', 'Chartreuse', 'Aqua', 'Azure',
    'Beige', 'Bisque', 'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue',
    'AntiqueWhite', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson',
    'Cyan', 'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'RosyBrown', 'Aquamarine', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def crop_image(detections, image, confidence_threshold=0.8, expansion=0):
    """
    Crops detections above *confidence_threshold* from the PIL image *image*,
    returning a list of PIL images.

    *detections* should be a list of dictionaries with keys 'conf' and 'bbox';
    see bbox format description below.  Normalized, [x,y,w,h], upper-left-origin.

    *expansion* specifies a number of pixels to include on each side of the box.
    """

    ret_images = []
    categories = []
    for detection in detections:

        score = float(detection['best_probability'])

        if score >= confidence_threshold:

            x1, y1, w_box, h_box = detection['bbox']
            ymin,xmin,ymax,xmax = y1, x1, y1 + h_box, x1 + w_box

            # Convert to pixels so we can use the PIL crop() function
            im_width, im_height = image.size
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)

            if expansion > 0:
                left -= expansion
                right += expansion
                top -= expansion
                bottom += expansion

            # PIL's crop() does surprising things if you provide values outside of
            # the image, clip inputs
            left = max(left,0); right = max(right,0)
            top = max(top,0); bottom = max(bottom,0)

            left = min(left,im_width-1); right = min(right,im_width-1)
            top = min(top,im_height-1); bottom = min(bottom,im_height-1)

            ret_images.append(image.crop((left, top, right, bottom)))
            categories.append(detection['best_class'])
        # ...if this detection is above threshold

    # ...for each detection

    return ret_images, categories


def render_detection_bounding_boxes(detections, image,
                                    classification_label_map={},
                                    confidence_threshold=0.8, thickness=4, expansion=0,
                                    classification_confidence_threshold=0.3,
                                    max_classifications=3,
                                    colormap=COLORS,
                                    textalign=TEXTALIGN_LEFT):
    """
    Renders bounding boxes, label, and confidence on an image if confidence is above the threshold.

    This works with the output of the batch processing API.

    Supports classification, if the detection contains classification results according to the
    API output version 1.0.

    Args:

        detections: detections on the image, example content:
            [
                {
                    "category": "2",
                    "conf": 0.996,
                    "bbox": [
                        0.0,
                        0.2762,
                        0.1234,
                        0.2458
                    ]
                }
            ]

            ...where the bbox coordinates are [x, y, box_width, box_height].

            (0, 0) is the upper-left.  Coordinates are normalized.

            Supports classification results, if *detections* has the format
            [
                {
                    "category": "2",
                    "conf": 0.996,
                    "bbox": [
                        0.0,
                        0.2762,
                        0.1234,
                        0.2458
                    ]
                    "classifications": [
                        ["3", 0.901],
                        ["1", 0.071],
                        ["4", 0.025]
                    ]
                }
            ]

        image: PIL.Image object

        label_map: optional, mapping the numerical label to a string name. The type of the numerical label
            (default string) needs to be consistent with the keys in label_map; no casting is carried out.

        classification_label_map: optional, mapping of the string class labels to the actual class names.
            The type of the numerical label (default string) needs to be consistent with the keys in
            label_map; no casting is carried out.

        confidence_threshold: optional, threshold above which the bounding box is rendered.
        thickness: line thickness in pixels. Default value is 4.
        expansion: number of pixels to expand bounding boxes on each side.  Default is 0.
        classification_confidence_threshold: confidence above which classification result is retained.
        max_classifications: maximum number of classification results retained for one image.

    image is modified in place.
    """

    display_boxes = []
    display_strs = []  # list of lists, one list of strings for each bounding box (to accommodate multiple labels)
    classes = []  # for color selection

    for detection in detections:
        if detection['best_probability'] >= confidence_threshold:
            x1, y1, w_box, h_box = detection['bbox']
            display_boxes.append([y1, x1, y1 + h_box, x1 + w_box])
            clss = detection['best_class']

            # ...if we have detection results
            display_strs.append([detection['name']])
            classes.append(clss)

        # ...if the confidence of this detection is above threshold

    # ...for each detection
    display_boxes = np.array(display_boxes)

    draw_bounding_boxes_on_image(image, display_boxes, classes,
                                 display_strs=display_strs, thickness=thickness, 
                                 expansion=expansion, colormap=colormap, textalign=textalign)


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 classes,
                                 thickness=4,
                                 expansion=0,
                                 display_strs=(),
                                 colormap=COLORS,
                                 textalign=TEXTALIGN_LEFT):
    """
    Draws bounding boxes on an image.

    Args:
      image: a PIL.Image object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
             The coordinates are in normalized format between [0, 1].
      classes: a list of ints or strings (that can be cast to ints) corresponding to the class labels of the boxes.
             This is only used for selecting the color to render the bounding box in.
      thickness: line thickness in pixels. Default value is 4.
      expansion: number of pixels to expand bounding boxes on each side.  Default is 0.
      display_strs: list of list of strings.
                             a list of strings for each bounding box.
                             The reason to pass a list of strings for a
                             bounding box is that it might contain
                             multiple labels.
    """

    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        # print('Input must be of size [N, 4], but is ' + str(boxes_shape))
        return  # no object detection on this image, return
    for i in range(boxes_shape[0]):
        if display_strs:
            display_str_list = display_strs[i]
            draw_bounding_box_on_image(image,
                                       boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3],
                                       classes[i],
                                       thickness=thickness, expansion=expansion,
                                       display_str_list=display_str_list,
                                       colormap=colormap,
                                       textalign=textalign)


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               clss=None,
                               thickness=4,
                               expansion=0,
                               display_str_list=(),
                               use_normalized_coordinates=True,
                               label_font_size=16,
                               colormap=COLORS,
                               textalign=TEXTALIGN_LEFT):
    """
    Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box - upper left.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    clss: str, the class of the object in this bounding box - will be cast to an int.
    thickness: line thickness. Default value is 4.
    expansion: number of pixels to expand bounding boxes on each side.  Default is 0.
    display_str_list: list of strings to display in box
        (each to be shown on its own line).
        use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    label_font_size: font size to attempt to load arial.ttf with
    """
    if clss is None:
        color = colormap[1]
    else:
        color = colormap[int(clss) % len(colormap)]

    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    if expansion > 0:
        left -= expansion
        right += expansion
        top -= expansion
        bottom += expansion

        # Deliberately trimming to the width of the image only in the case where
        # box expansion is turned on.  There's not an obvious correct behavior here,
        # but the thinking is that if the caller provided an out-of-range bounding
        # box, they meant to do that, but at least in the eyes of the person writing
        # this comment, if you expand a box for visualization reasons, you don't want
        # to end up with part of a box.
        #
        # A slightly more sophisticated might check whether it was in fact the expansion
        # that made this box larger than the image, but this is the case 99.999% of the time
        # here, so that doesn't seem necessary.
        left = max(left,0); right = max(right,0)
        top = max(top,0); bottom = max(bottom,0)

        left = min(left,im_width-1); right = min(right,im_width-1)
        top = min(top,im_height-1); bottom = min(bottom,im_height-1)

    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)


    # try:
    #     font = ImageFont.truetype('arial.ttf', label_font_size)
    # except IOError:
    #     font = ImageFont.load_default()
    #
    # # If the total height of the display strings added to the top of the bounding
    # # box exceeds the top of the image, stack the strings below the bounding box
    # # instead of above.
    # display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    #
    # # Each display_str has a top and bottom margin of 0.05x.
    # total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
    #
    # if top > total_display_str_height:
    #     text_bottom = top
    # else:
    #     text_bottom = bottom + total_display_str_height
    #
    # # Reverse list and print from bottom to top.
    # for display_str in display_str_list[::-1]:
    #     text_width, text_height = font.getsize(display_str)
    #
    #     text_left = left
    #
    #     if textalign == TEXTALIGN_RIGHT:
    #         text_left = right - text_width
    #
    #     margin = np.ceil(0.05 * text_height)
    #
    #     draw.rectangle(
    #         [(text_left, text_bottom - text_height - 2 * margin), (text_left + text_width,
    #                                                           text_bottom)],
    #         fill=color)
    #
    #     draw.text(
    #         (text_left + margin, text_bottom - text_height - margin),
    #         display_str,
    #         fill='black',
    #         font=font)
    #
    #     text_bottom -= (text_height + 2 * margin)


def move_save_img(img_file, mv_dir):
    split_name = osp.basename(img_file).split('_')

    if len(split_name) < 2:
        not_split_mv_dir = osp.join(mv_dir, 'test_mv')
        if not osp.exists(not_split_mv_dir):
            os.makedirs(not_split_mv_dir, exist_ok=True)

        not_split_mv_file_name = osp.join(not_split_mv_dir, osp.basename(img_file))

        if osp.exists(not_split_mv_file_name):
            os.remove(img_file)
        else:
            shutil.move(img_file, not_split_mv_file_name)

        return not_split_mv_file_name
    else:
        split_mv_dir = osp.join(mv_dir, split_name[0])

        if not osp.exists(split_mv_dir):
            os.makedirs(split_mv_dir, exist_ok=True)

        mv_file_name = osp.join(split_mv_dir, '_'.join(split_name[1:]))

        if osp.exists(mv_file_name):
            os.remove(img_file)
        else:
            shutil.move(img_file, mv_file_name)

        return mv_file_name


