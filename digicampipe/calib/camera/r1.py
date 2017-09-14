import numpy as np


def calibrate_to_r1(event_stream):

    pixel_list = list(range(1296))

    for event in event_stream:

        for telescope_id in event.r0.tels_with_data:

            r0_camera = event.r0.tel[telescope_id]
            adc_samples = np.array(list(r0_camera.adc_samples.values()))
            baseline_mean, baseline_std = compute_baseline(adc_samples)

            r1_camera = event.r1.tel[telescope_id]
            r1_camera.pedestal_mean = dict(zip(pixel_list, baseline_mean))
            r1_camera.pedestal_std = dict(zip(pixel_list, baseline_std))
            r1_camera.gain_drop = None
            r1_camera.nsb = None
            gain = np.ones(adc_samples.shape[0]) * 5.8
            charge = compute_charge(adc_samples, gain, baseline_mean, baseline_std)
            r1_camera.pe_samples = dict(zip(pixel_list, charge))

            yield event


def compute_baseline(adc_samples, start=0, end=-1):

    baseline_mean = np.mean(adc_samples[..., start:end], axis=-1)
    baseline_std = np.std(adc_samples[..., start:end], axis=-1)

    return baseline_mean, baseline_std


def compute_charge(adc_samples, gain, baseline_mean, baseline_std=None, type='max'):

    if type == 'max':

        return (np.max(adc_samples) - baseline_mean) / gain

    else:

        print('Unknown type %s' %type)

def compute_gain_drop(pedestal, type='std'):

    if type == 'mean':

        pass

    elif type == 'std':

        pass

    else:

        print('Unknown type %s' % type)

    return

def compute_nsb_rate(pedestal, type='std'):

    if type == 'mean':

        pass

    elif type == 'std':

        pass

    return


