from glob import glob
import os
from PIL import ImageFont, ImageDraw, Image
import torch
from torch_geometric.data import Data
from trainer.data.util import sparse_to_dense
from trainer.global_configs import DATA_DIR, FONTS_DIR, LABEL_MAP, CANVAS_WIDTH, CANVAS_LENGTH, IMAGE_FILE_TYPES, MODEL_LABELS

def get_normalized_dimensions(image_path, canvas_width = CANVAS_WIDTH, canvas_length = CANVAS_LENGTH):
    """
    this function gets the image path and canvas size and returns tuple of normalized width and height
    """
    try:
        image = Image.open(image_path)
        width, height = image.size
        #normalise 
        normalized_width = width / canvas_width  #canvas width
        normalized_height = height / canvas_length #canvas height
        ## new approach (testing)
        #if (width > canvas_width) or (height > canvas_length):
        #    image.thumbnail((canvas_width, canvas_length), Image.ANTIALIAS)

        return normalized_width, normalized_height
    
    except IOError:
        print(f"Unable to open image file: {image_path}")
        return None
    

def get_normalized_dimensions_of_text(text, font_path=f'{FONTS_DIR}/KohSantepheap-Regular.ttf', font_size=20, canvas_width = CANVAS_WIDTH, canvas_length = CANVAS_LENGTH):
    """
    this function gets the text, font, size, and canvas size and canvas size then bbbox it  and returns tuple of normalized width and height
    """

    # load the font
    font = ImageFont.truetype(font_path, font_size)    
    
    # dummy image
    img = Image.new ('RGB', (1, 1))
    draw = ImageDraw.Draw(img)

    # split the text 
    lines = text.splitlines()
    
    # get bboxes for each line
    bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]   
    
    # output
    bbox_width = max(bbox[2] for bbox in bboxes)
    bbox_height = sum(bbox[3] for bbox in bboxes)
    
    # Normalize the width and height by dividing by the maximum possible value which is the canvas w,h
    norm_width = float(bbox_width) / canvas_width 
    norm_height = float(bbox_height) / canvas_length
    
    return norm_width, norm_height


def generate_input_from_directory(model_name='layoutdm_rico', verbatim=False):
    '''
    Model labels:
        - layoutdm_publaynet:
            ['text', 'title', 'list', 'table', 'figure']
        - layoutdm_rico:
            ['Text', 'Image', 'Icon', 'Text Button', 'List
            Item', 'Input', 'Background Image', 'Card', 'Web View', 'Radio Button',
            'Drawer', 'Checkbox', 'Advertisement', 'Modal', 'Pager Indicator', 'Slider',
            'On/Off Switch', 'Button Bar', 'Toolbar', 'Number Stepper', 'Multi-Tab',
            'Date Picker', 'Map View', 'Video', 'Bottom Navigation']
    '''

    # setup
    backgrounds = []
    images = []
    for image_file_type in IMAGE_FILE_TYPES:
        backgrounds.extend(glob(f"{DATA_DIR}/images/*background{image_file_type}"))
        images.extend(glob(f"{DATA_DIR}/images/*product{image_file_type}"))
        images.extend(glob(f"{DATA_DIR}/images/*logo{image_file_type}"))
    headers = glob(f"{DATA_DIR}/headers/*.header")
    texts = glob(f"{DATA_DIR}/texts/*.txt")

    tensor_list = []
    labels_list = []
    list_files = []
    
    # CANVAS dimensions taken from the background image, there should be one by default but multiple are accepted, change that in the future
    background = backgrounds[0]
    canvas_dimensions = Image.open(background).size


    # 0.processing the background
    for bckgrnd in backgrounds:
        bckgrond_label = LABEL_MAP['backgrounds'][model_name]
        
        dimensions = get_normalized_dimensions(bckgrnd, canvas_width = canvas_dimensions[0], canvas_length = canvas_dimensions[1])

        if dimensions:
            width, height = dimensions
            tensor_list.append([0.5, 0.5, width, height])
        else:
            tensor_list.append([0.5, 0.5, 0, 0])
        labels_list.append(bckgrond_label)##################################################################################
        list_files.append(bckgrnd)


    #1.processing the images
    for img in images:
        image_label = LABEL_MAP['images'][model_name]
        dimensions = get_normalized_dimensions(img, canvas_width = canvas_dimensions[0], canvas_length = canvas_dimensions[1])

        if dimensions:
            width, height = dimensions

            tensor_list.append([0.5,0.5,width, height])
            labels_list.append(image_label)##################################################################################
            list_files.append(img)
        else:
            tensor_list.append([0.5,0.5,0, 0])
            labels_list.append(image_label)##################################################################################
            list_files.append(img)
            
            
    #2. processing the headers
    for header in headers:
        header_label = LABEL_MAP['headers'][model_name]

        # Check if there is text in the title
        with open(header, 'r') as f:
            title_text = f.read().strip()
            if title_text:
                # If there is text in the title, put it in a box and measure its width and height
                width, height = get_normalized_dimensions_of_text(title_text, font_size=50, canvas_width = canvas_dimensions[0], canvas_length = canvas_dimensions[1])
                # Append a tensor representing the title to the tensor list
                tensor_list.append([0.5, 0.5, width, height])
                labels_list.append(header_label) ##################################################################################
                list_files.append(header)
            else:
                # If there is no text in the title, append a tensor with zero width and height
                tensor_list.append([0.5, 0.5, 0, 0])
                labels_list.append(header_label) ##################################################################################
                list_files.append(header)

    #3. processing the sub_headers/text
    for text in texts:
        text_label = LABEL_MAP['texts'][model_name]

        # Check if there is text in the title
        with open(text, 'r') as f:
            title_text = f.read().strip()
            if title_text:
                # If there is text in the title, put it in a box and measure its width and height
                width, height = get_normalized_dimensions_of_text(title_text,font_size=20, canvas_width = canvas_dimensions[0], canvas_length = canvas_dimensions[1])
                # Append a tensor representing the title to the tensor list
                tensor_list.append([0.5, 0.5, width, height])
                labels_list.append(text_label) ##################################################################################
                list_files.append(text)
            else:
                # If there is no text in the title, append a tensor with zero width and height
                tensor_list.append([0.5, 0.5, 0, 0])
                labels_list.append(text_label) ##################################################################################
                list_files.append(text)
        
    bboxes = torch.FloatTensor(tensor_list)
    labels = torch.LongTensor(labels_list)
    assert bboxes.size(0) == labels.size(0) and bboxes.size(1) == 4
    ## set some optional attributes by a dummy value (False)
    attr = {k: torch.full((1,), fill_value=False) for k in ["filtered", "has_canvas_element", "NoiseAdded"]}
    data = Data(x=bboxes, y=labels, attr=attr)

    if verbatim:
        print('\n', 20*'#', ' Input generated based on objects in input/ subdirectories ', 20*'#')
        for filepath, bbox, label in zip(list_files, data.x, data.y):
            #print(f"bboxes:\n{data.x}\nlabels: {data.y}\n")
            filename = os.path.basename(filepath)
            print(f"\n{filename}\n\tbbox:\t{bbox}\n\tlabel:\t'{MODEL_LABELS[model_name][label]}' ({model_name}[{label}])")

        
    return data, list_files, canvas_dimensions  # based on this need to attribute the result in a variable then[0] for data and [1] to map