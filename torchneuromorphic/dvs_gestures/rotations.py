import numpy as np

def make_new_classes(sample, label, num_gestures=11):

    degs = [90., 180., 270.]
    samples = [sample]
    labels = [label]
    for d_idx, d in enumerate(degs):
        homogeneous_samples = np.hstack((samples, 
                                         np.ones((samples.shape[0],1)))).T
        mat = make_rot_matrix(np.deg2rad(d))
        rotated_samples = mat @ homogeneous_samples
        # hack; 4 should not be hard coded
        samples.append( (rotated_samples.T)[:,:4]) 
        labels.append(label + d_idx * num_gestures)
    return samples, labels


def make_rot_matrix(angle, x_c=64, y_c=64):
    '''
    This function is assuming the original data is in the format
    (time, x, y, polarity, 1) and that samples are contained in 
    columns, not rows.
    '''
    sub_mat = np.array([[1, 0, 0, 0, 0], # time
                        [0, 1, 0, 0, -x_c],
                       [0, 0, 1, 0, -y_c],
                       [0, 0, 0, 1, 0]]) # polarity
                       [0, 0, 0, 0, 1]])

    rot_mat = np.array([
        [  1    ,               0      , 0, 0, 0], # time
        [  0,  np.cos(angle), np.sin(angle),  0, 0],
        [  0,   -np.sin(angle), np.cos(angle), 0, 0],
        [  0    ,               0      , 0, 1, 0], # polarity
        [  0    ,               0      , 0, 0, 1]])
                        

    add_mat = np.array([
                       [1, 0, 0, 0, 0], # time
                       [0, 1, 0, 0, x_c],
                       [0, 0, 1, 0, y_c],
                       [0, 0, 0, 1, 0]]) # polarity
                       [0, 0, 0, 0, 1]])

    return add_mat @ rot_mat @ sub_mat  # compose the transformations.
