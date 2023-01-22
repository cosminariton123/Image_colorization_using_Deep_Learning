from config import INPUT_SIZE, GROUND_THRUTH_SIZE

def convert_from_image_to_numpy_notation(size_tuple):
    size = list(size_tuple)
    size =  [size[1], size[0]] + size[2:]
    
    return tuple(size)

def compute_input_size_numpy():
    return convert_from_image_to_numpy_notation(INPUT_SIZE)

def compute_ground_truth_size_numpy():
    return convert_from_image_to_numpy_notation(GROUND_THRUTH_SIZE)

INPUT_SIZE_NUMPY = compute_input_size_numpy()

GROUND_TRUTH_SIZE_NUMPY = compute_ground_truth_size_numpy()
