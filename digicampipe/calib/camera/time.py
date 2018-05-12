import numpy as np


def compute_time_from_max(events):

    bin_time = 4  # 4 ns between samples

    for count, event in enumerate(events):

        adc_samples = event.data.adc_samples
        reconstructed_time = np.argmax(adc_samples, axis=-1)
        reconstructed_time *= bin_time

        new_shape = reconstructed_time.shape + (1, )
        reconstructed_time = reconstructed_time.reshape(new_shape)
        event.data.reconstructed_time = reconstructed_time

        yield event
