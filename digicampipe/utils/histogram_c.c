#include <stdint.h>
#include <stdio.h>

void histogram(const float *data, unsigned int *hist, unsigned int *underflow, unsigned int *overflow, const float *bins, const unsigned int n_pixels, const unsigned int n_samples, const unsigned int n_bins)
{

    unsigned int index_hist, index_data, index_bin, first, last;

    for (unsigned int i = 0; i < n_pixels; ++i)
    {

        for (unsigned int j = 0; j < n_samples; ++j)

        {

            index_data = i * (n_samples) + j;
            // printf("%d\n", index_data);

            if (data[index_data] >= (bins[0]) && (data[index_data] < bins[n_bins - 1]))
            {

                first = 0;
                last = n_bins - 1;
                index_bin = (first + last) / 2;

                while(first <= last)
                {

                    if ((data[index_data] >= bins[index_bin]) && (data[index_data] < bins[index_bin + 1]))

                    {
                        index_hist = i * (n_bins - 1) + index_bin;
                        // printf("%d\n", index_hist);
                        ++hist[index_hist];
                        break;
                    }

                    else if (data[index_data] > bins[index_bin])
                    {

                        first = index_bin + 1;
                    }

                    else
                    {

                        last = index_bin - 1;
                    }

                    index_bin = (first + last) / 2;
                }

            }

            else if(data[index_data] < bins[0])

            {

            ++underflow[i];

            }

            else if(data[index_data] >= bins[n_bins - 1])

            {

            ++overflow[i];

            }




        }
    }
}
