
import scipy.io

labels_needed = ['x', 'y', 'visibility']

def load_mat_file(mat_filename):
    """
    Returns a dictionary with attributes:
        action: 'tennis serve'
        pose: 'back'
        x: 151x13
        y: 151x13
        visibility: 151x13
        train: 
        bbox
        dimensions
        nframes
    """
    return scipy.io.loadmat(mat_filename)


def get_needed_annotations(all_annotations, req_labels):
    new_dict = {}
    for label_name in req_labels:
        new_dict[label_name] = all_annotations[label_name]
    return new_dict


def get_penn_action_annotations(mat_filename):
    annotations = load_mat_file(mat_filename)
    new_dict = get_needed_annotations(annotations, labels_needed)
    return new_dict
