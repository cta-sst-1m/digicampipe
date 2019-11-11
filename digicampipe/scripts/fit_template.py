from digicampipe.utils.fitter import PulseTemplateFitter
from digicampipe.utils.pulse_template import NormalizedPulseTemplate

from pkg_resources import resource_filename
import os
import numpy as np
import matplotlib.pyplot as plt

TEMPLATE_FILENAME = resource_filename(
    'digicampipe',
    os.path.join(
        'tests',
        'resources',
        'pulse_SST-1M_pixel_0.dat'
    )
)
template = NormalizedPulseTemplate.load(TEMPLATE_FILENAME)

true_charge = 10.4
true_baseline = 40.4
true_time = 12.4588
times = np.arange(0, 50, 1) * 4
sigma = 0.1
noise = np.random.normal(0, sigma, size=len(times))
data = true_charge * template(times - true_time) + true_baseline + noise

fitter = PulseTemplateFitter(data=data, error=np.ones(data.shape)*sigma)
fitter.fit(sigma_t=1, )
fitter.plot_likelihood('t_0', x_label='Time [ns]')
fitter.plot_likelihood('baseline', x_label='Baseline [LSB]')
fitter.plot_likelihood('charge', x_label='Charge [LSB]')
fitter.plot_likelihood('t_0', 'charge', x_label='Baseline [LSB]', y_label='Charge [LSB]', size=(100, 100))
fitter.plot()
print(fitter)
plt.show()
