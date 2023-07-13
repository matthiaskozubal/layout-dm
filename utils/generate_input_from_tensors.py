# Setup
import torch
from torch_geometric.data import Data
from utils import print_attrs


# Custom dataset
def generate_input_from_tensors(
    bboxes_tensor = torch.FloatTensor([
        [0.5, 0.5, 0.1, 0.6],
        [0.5, 0.5, 0.6, 0.1],
        [0.5, 0.5, 0.2, 0.2],
        [0.5, 0.5, 0.4, 0.4],
        [0.8, 0.8, 0.1, 0.1],
        [0.2, 0.2, 0.1, 0.1]
    ]),
    labels_tensor = torch.LongTensor([0, 1, 6, 4, 0, 0])
    , verbatim=False):
    
    bboxes = bboxes_tensor
    ## see .labels of each dataset class for name-index correspondense
    labels = labels_tensor
    assert bboxes.size(0) == labels.size(0) and bboxes.size(1) == 4
    ## set some optional attributes by a dummy value (False)
    attr = {k: torch.full((1,), fill_value=False) for k in ["filtered", "has_canvas_element", "NoiseAdded"]}
    custom_data = Data(x=bboxes, y=labels, attr=attr)  # can be used as an alternative for `dataset[target_index]` in demo.ipynb
    if verbatim:
        print(f"bboxes:\n{custom_data.x}\nlabels: {custom_data.y}\n")
        
    return custom_data
