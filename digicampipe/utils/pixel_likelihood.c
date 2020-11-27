#include <stdint.h>
#include <stdio.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_spline.h>


double log_generalized_poisson(double k, double mu, double mu_xt, double baseline, double gain)
{

    double k_hat;
    k_hat = (k - baseline) / gain;
    if ((mu <= 0.0) || (gain <= 0.0 ))
    {
        return GSL_NAN;
    }

    else if (k_hat < 0.0)
    {
        return -10.0;
    }


    double x = mu + mu_xt * k_hat;
    double log_pdf = gsl_sf_log(mu) - gsl_sf_lngamma(k_hat + 1.0) + (k_hat - 1.0) * gsl_sf_log(x) - x - gsl_sf_log(gain);

    return log_pdf;
}

double generalized_poisson(double k, double mu, double mu_xt, double baseline, double gain)
{
    double log_pdf = log_generalized_poisson(k, mu, mu_xt, baseline, gain);
    if ((gsl_isnan(log_pdf)) || (log_pdf < -300))
    {
        return 0.0;
    }
    else
    {
        return gsl_sf_exp(log_pdf);
    }

}

double log_normal_distribution(double x, double mean, double sigma)
{
    double log_pdf = - gsl_pow_2(x - mean) / (2.0 * gsl_pow_2(sigma));
    log_pdf = log_pdf - gsl_sf_log(sqrt(2.0 * M_PI) * sigma);
    return log_pdf;

}

double normal_distribution(double x, double mean, double sigma)
{
    return gsl_sf_exp(log_normal_distribution(x, mean, sigma));
}

double pixel_loglikelihood(double x, double mu, double mu_xt, double gain, double sigma_e, double sigma_s, double baseline)
{

    // printf("x : %.2f \t mu : %.2f \t mu_xt : %.2f \t gain : %.2f \t sigma_e : %.2f \t sigma_s : %.2f \t baseline : %.2f \n  ", x, mu, mu_xt, gain, sigma_e, sigma_s, baseline);

    double llh = 0.0;
    double pdf = 0.0;
    double mean;
    double sigma;
    double log_1;
    double log_2;
    double log_pdf;
    double p_threshold = 1e-8;
    double k_min;
    double k_max;
    double mean_hat = mu / (1.0 - mu_xt) * gain + baseline;
    double sigma_hat = sqrt(mu / gsl_pow_3(1.0 - mu_xt) ) * gain;
    double mu_max = 5.0 * (gsl_pow_2(gain / 2.0) - gsl_pow_2(sigma_e)) / sigma_s;

    mu_max = ceil(mu_max);
    // printf("mu_max : %.3f\n", mu_max);
    // printf("mu : %.3f\n", mu);
    if (gain <= 0.001)
    {
        llh = log_normal_distribution(x, baseline, sigma_e);
        // printf("I went here !");

    }
    else if (mu > mu_max)
    {

        llh = log_generalized_poisson(x, mu, mu_xt, baseline, gain);
        // printf("I am here !");
    }

    else
    {
        double width_gauss = sqrt(2.0) * sigma_hat * sqrt(-gsl_sf_log(p_threshold * sqrt(2.0 * M_PI) * sigma_hat));
        k_min = (mean_hat - width_gauss - baseline) / gain;
        k_max = (mean_hat + width_gauss - baseline) / gain;
        k_min = GSL_MAX(k_min, 0.0);
        k_max = GSL_MAX(mu_max, k_max);
        k_max *= 10.0;
        k_max = ceil(k_max);

        // printf("%.10f\n", p_threshold);
        // printf("mean %.1f\n" , mean_hat);
        // printf("sigma %.1f\n" , sigma_hat);
        // printf("width %.1f\n" , width_gauss);
        // printf("k_min %.1f\n" , k_min);
        // printf("k_max %.1f\n" , k_max);
        // printf("OMG !");

        for ( double k = k_min; k <= k_max; k++ )
        {
            // printf("k = %.1f\n", k);
            mean = k * gain + baseline;
            sigma = sqrt(gsl_pow_2(sigma_e) + k * gsl_pow_2(sigma_s));
            log_1 = log_generalized_poisson(k, mu, mu_xt, 0.0, 1.0);
            // printf("Log-Poisson %.9f\n", log_1);

            log_2 = log_normal_distribution(x, mean, sigma);
            // printf("Log-Normal %.9f\n", log_2);

            log_pdf = log_1 + log_2;
            // printf("Log pdf %.9f\n", log_pdf);
            if (log_pdf > -600.0)
            {
                pdf += gsl_sf_exp(log_pdf);
                // printf("Hello I am %.1f\n", k);
            }

        }

        llh = gsl_sf_log(pdf);
    }
    return llh;

}

void pixel_likelihood_1d(const double* x, double* out, const double mu, const double mu_xt, const double gain, const double sigma_e, const double sigma_s, const double baseline, const unsigned int n_points)
{

    for (int i = 0; i < n_points; i++)
    {

       out[i] = gsl_sf_exp(pixel_loglikelihood(x[i], mu, mu_xt, gain, sigma_e, sigma_s, baseline));
    }
}

double B(const double x, const int k, const int i, const double* t)
{
    double c_1 = 0.0;
    double c_2 = 0.0;

    if (k == 0)
    {
        if ((t[i] <= x ) && (x < t[i+1]))
        {
            return 1.0;
        }
        else
        {
            return 0.0;
        }

    }
    if (t[i+k] == t[i])
    {
        c_1 = 0.0;
    }
    else
    {
        c_1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t);
    }
    if (t[i+k+1] == t[i+1])
    {
        c_2 = 0.0;
    }
    else
    {

        c_2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t);
    }


    return c_1 + c_2;

}

double bspline(const double x, const double* t, const double* c, const int k, const int n_knots)
{
    unsigned int n = n_knots - k - 1;
    double out = 0.0;
    for (unsigned int i = 0; i < n; i++)
    {
        out += c[i] * B(x, k, i, t);
    }
}

double log_gaussian_2d(const double x, const double y,
                   const double x_cm, const double y_cm,
                   const double psi, const double width,
                   const double length, const double size)
{
    double scale_w = 1.0 / (2.0 * width * width);
    double scale_l = 1.0 / (2.0 * length * length);
    double a = gsl_sf_cos(psi) * gsl_sf_cos(psi) * scale_l + gsl_sf_sin(psi) * gsl_sf_sin(psi) * scale_w;
    double b  = gsl_sf_sin(2.0 * psi) * (scale_w - scale_l) / 2.0;
    double c = gsl_sf_cos(psi) * gsl_sf_cos(psi) * scale_w + gsl_sf_sin(psi) * gsl_sf_sin(psi) * scale_l;

    double norm = 1.0 / (2.0 * M_PI * width * length);

    double log_pdf = - (a * (x - x_cm) * (x - x_cm) - 2.0 * b * (x - x_cm) * (y - y_cm) + c * (y - y_cm) * (y - y_cm));

    log_pdf += gsl_sf_log(norm) + gsl_sf_log(size);

    return log_pdf;

}

double gaussian_2d(const double x, const double y,
                   const double x_cm, const double y_cm,
                   const double psi, const double width,
                   const double length, const double size)
{

    return gsl_sf_exp(log_gaussian_2d(x, y, x_cm, y_cm, psi, width, length, size));
}

// void pulse_template(const double* x, double* y, const double* coefficients, )

double image_loglikelihood(
        const double* waveform,
        const double* pixel_x,
        const double* pixel_y,
        const double* pixel_area,
        const unsigned int n_pixels,
        const unsigned int n_samples,
        const double t_start,
        const double dt,
        const double* sigma_e,
        const double* sigma_s,
        const double* mu_xt,
        const double* gain,
        const double* baseline,
        const double* t,
        const double* c,
        const int k,
        const int n_knots,
        const double t_cm,
        const double v,
        const double x_cm,
        const double y_cm,
        const double psi,
        const double width,
        const double length,
        const double size)
{

    double llh = 0.0;
    double pixel_lh;
    unsigned int index;
    double image_model;
    double gain_model;
    double time = t_start;
    double t_pixel;

    double n_points = 0.0;
    for (unsigned int i = 0; i < n_pixels; i++)
    {

        // printf("pixel_x: %.3f, pixel_y: %.3f, pixel_area: %.3f, x_cm: %.3f, y_cm: %.3f, psi: %.3f, width: %.3f, length: %.3f, size: %.3f\n", pixel_x[i], pixel_y[i], pixel_area[i], x_cm, y_cm, psi, width, length, size);

        image_model = gaussian_2d(pixel_x[i], pixel_y[i], x_cm, y_cm, psi, width, length, size);
        image_model *= pixel_area[i];
        t_pixel = v * ((pixel_x[i] - x_cm) * gsl_sf_cos(psi) +  (pixel_y[i] - y_cm) * gsl_sf_sin(psi)) + t_cm;
        // printf("amplitude: %.3f\n, t_pixel: %.3f", image_model, t_pixel);


        time = t_start;

        for (unsigned int j = 0; j < n_samples; j++)
        {
            index = i * n_samples + j;
            // printf("%d\n", index);
            // printf("Gain: %.2f, time: %.2f\n", gain[i], time);
            // printf("Waveform: %.3f, Amplitude: %.3f, Crosstalk: %.3f, Gain: %.3f, sigma_e: %.3f, sigma_s: %.3f, baseline: %.3f\n", waveform[index], image_model,mu_xt[i], gain_model, sigma_e[index], sigma_s[i], baseline[i]);
            gain_model = gain[i] * bspline(time - t_pixel , t, c, k, n_knots);
            // printf("time: %.3f, dt: %.3f, gain: %.3f\n",time,  time - t_pixel, gain_model);
            pixel_lh = pixel_loglikelihood(waveform[index], image_model, mu_xt[i], gain_model, sigma_e[i], sigma_s[i], baseline[i]);
            // printf("Pixel LH: %.8f\n", pixel_lh);
            if (!gsl_isnan(pixel_lh))
            {
                llh += pixel_lh;
                // pixel_lh = gsl_sf_log(pixel_lh);
                // llh += pixel_lh;
                n_points += 1.0;

            }
            time += dt;

        }


    }

    // printf("LLH: %.2f\n", llh);
    return llh / n_points;
}


int main() {

    // double a = normal_distribution(1, 0, 3);
    // double b = generalized_poisson(0, 1, 2, 3, 4);
    // double c = generalized_poisson(10., 2., 3., 2., 5.);
    // printf("%.9f\n", a);
    // printf("%.9f\n", b);
    // printf("%.9f\n", c);
    // printf("%.9f\n", gsl_sf_log(10.));

    double xx = 0.0;
    double dx = 1.0;
    double x_0 = 10.0;
    double x[100]= {};
    double y[100] = {};
    double mu = 10.0;
    double mu_xt = 0.2;
    double gain = 10.0;
    double sigma_e = 0.5;
    double sigma_s = 1;
    double baseline = 100.0;

    xx += x_0;
    // printf("%.2f\n", xx);
    for (int i = 0; i < 100; i++)
    {
        xx += dx;
        x[i] = xx;
        y[i] = gsl_sf_exp(pixel_loglikelihood(x[i], mu, mu_xt, gain, sigma_e, sigma_s, baseline));
        // printf("%.9f", pixel_likelihood(x[i], mu, mu_xt, gain, sigma_e, sigma_s, baseline));
        // printf("%.9f\t", x[i]);
        // printf("%.9f\n", y[i]);
    }
    // pixel_likelihood_1d(&x, mu, mu_xt, gain, sigma_e, sigma_s, baseline, y);

    return 0;

}