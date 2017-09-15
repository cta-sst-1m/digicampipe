import numpy as np


def compute_gain_drop(pedestal, type='std'):
    if type == 'mean':

        return np.ones(pedestal.shape[0])
    elif type == 'std':

        return np.ones(pedestal.shape[0])
    else:
        raise('Unknown type %s' % type)


def compute_nsb_rate(pedestal, type='std'):
    if type == 'mean':

        return np.ones(pedestal.shape[0]) * 1.e9
    elif type == 'std':

        return np.ones(pedestal.shape[0]) * 1.e9
    else:
        raise('Unknown type %s' % type)


def get_gains():
    return np.ones(1296) * 23. # TODO, replace gain of 23 by calib array of gain