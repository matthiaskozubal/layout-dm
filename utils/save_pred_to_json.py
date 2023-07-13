import json
import os
from trainer.global_configs import OUTPUT_DIR, CANVAS_LENGTH, CANVAS_WIDTH


def save_pred_to_json(list_files, pred, OUTPUT_DIR=OUTPUT_DIR, verbatim=False, which_page_to_save='all'):
    '''
    Save pred dict with tensors to json
    To Do:
        - rescaling
    Example usage:
        _ = save_pred_to_json(pred, OUTPUT_DIR=OUTPUT_DIR, verbatim=False)
    '''
    
    # translate pred to bbox_list and label_list
    ## how many objects per page
    n_objects = len(list_files)
    ## bbox
    bbox_list = pred['bbox'].tolist()
    n_pages = len(bbox_list)
    bbox_list = [[object for object in page[:n_objects]] for page in bbox_list]
    ## labels
    label_list = pred['label'].tolist()
    label_list = label_list[:n_objects]

    # translate bbox_list and label_list into a dict and save
    output = dict()
    for page in range(n_pages):
        ## append page to dict
        output[page] = {}
        ## print
        if verbatim:
            print(f'\nPage {page}:')
        for object_in_page in range(n_objects):
            ## for given page: grab a bbox tuple, resize, and add to a dict 
            file_name = list_files[object_in_page]
            bbox_tuple = tuple(bbox_list[page][object_in_page])
            bbox_tuple_resized = (bbox_tuple[0]*CANVAS_WIDTH, bbox_tuple[1]*CANVAS_LENGTH, bbox_tuple[2]*CANVAS_WIDTH, bbox_tuple[3]*CANVAS_LENGTH)
            bbox_tuple_resized_rounded = tuple(round(elem) for elem in bbox_tuple_resized)
            output[page][file_name] = bbox_tuple_resized_rounded   
            ### dump
            #output_file_path = os.path.join(OUTPUT_DIR, 'output.json')
            #with open(output_file_path, 'w') as output_file:
            #    json.dump(output, output_file)
            ## print
            if verbatim:
                print(f'{os.path.basename(list_files[object_in_page])}:\t\t{tuple([round(elem, 3) for elem in bbox_list[page][object_in_page]])}')
            
    # save
    output_file_path = os.path.join(OUTPUT_DIR, 'predicted_layouts.json')
    with open(output_file_path, 'w') as output_file:
        if which_page_to_save == 'all':
            pass
        elif isinstance(which_page_to_save, int):
            output = output[which_page_to_save]
        else:
            raise TypeError("Please enter 'all' or an integer")
        json.dump(output, output_file)
            
    return output



