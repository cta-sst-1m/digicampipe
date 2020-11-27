import numpy as np

from digicampipe.utils.pdf import gaussian2d


def time_log_likelihood(times, amplitudes, t_0, template):

    # times = np.append(times, times.max() + np.diff(times)[0])
    # time_bin = np.diff(times)
    # times = times[:-1] + time_bin / 2
    t = times[:, np.newaxis] - t_0
    y = template.pdf(t)
    mask = (y > 0)
    y[~mask] = 1E-40

    y = y / np.trapz(y, t, axis=0)
    y = y.T
    # amplitudes = (amplitudes / time_bin) / np.sum(amplitudes, axis=-1)
    amplitudes = amplitudes - np.min(amplitudes, axis=-1)[..., None]
    amplitudes = amplitudes # / np.sum(amplitudes, axis=-1)[..., None]

    log_lh = amplitudes * np.log(y)
    log_lh = np.nansum(log_lh, axis=-1)

    return log_lh


def velocity_likelihood(times, amplitudes, long, v, x0, template):

    p = [v, x0]
    t_0 = np.polyval(p, long)
    log_lh = time_log_likelihood(times, amplitudes, t_0, template)
    log_lh = np.nansum(log_lh, axis=-1)

    return log_lh


def gaussian_likelihood(photo_electrons, x, y, x_cm, y_cm, width, length, psi):

    scale_w = 1. / (2. * width**2)
    scale_l = 1. / (2. * length**2)
    a = np.cos(psi)**2 * scale_l + np.sin(psi)**2 * scale_w
    b = np.sin(2 * psi) * (scale_w - scale_l) / 2.
    c = np.cos(psi)**2 * scale_w + np.sin(psi)**2 * scale_l

    norm = 1. / (2 * np.pi * width * length)

    log_lh = - (a * (x - x_cm)**2 - 2 * b * (x - x_cm) * (y - y_cm) + c * (y - y_cm)**2) + np.log(norm)
    log_lh = log_lh * photo_electrons
    log_lh = np.sum(log_lh)

    return log_lh


def pulse_log_likelihood(times, amplitudes, charge, baseline, t_0, template, sigma=1):

    norm = np.sqrt(1 / (2 * np.pi)) / sigma
    y_fit = charge * template(times - t_0) + baseline
    log_lh = - ((y_fit - amplitudes) / sigma)**2 * 0.5
    log_lh = np.sum(log_lh, axis=-1) + np.log(norm)

    return log_lh


def pulse_chisquare(amplitudes, times, charge, t0, baseline, template):

    y = charge * template(times - t0) + baseline

    chi2 = (y - amplitudes)**2
    chi2 = np.sum(chi2, axis=-1)

    return chi2


def combined_log_likelihood(times, amplitudes, template, pix_x, pix_y, charge,
                            baseline, t_cm, x_cm, y_cm, width, length,
                            psi, v, sigma=1):

    dx = (pix_x - x_cm)
    dy = (pix_y - y_cm)
    long = dx * np.cos(psi) + dy * np.sin(psi)
    p = [v, t_cm]
    t = np.polyval(p, long)

    A = gaussian2d(photo_electrons=1,
                   x=pix_x, y=pix_y, x_cm=x_cm,
                   y_cm=y_cm, width=width, length=length,
                   psi=psi)
    t = times[..., np.newaxis] - t
    t = t.T
    y_fit = charge * A[..., np.newaxis] * template(t) + baseline
    norm = np.sqrt(1 / (2 * np.pi)) / sigma
    log_lh = - ((y_fit - amplitudes) / sigma)**2 * 0.5
    log_lh = np.sum(log_lh) + np.log(norm)

    return log_lh
