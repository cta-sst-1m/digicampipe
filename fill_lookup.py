import numpy as np


def fill_lookup(size_bins_edges, impact_bins_edges,
                impact_parameter, size, data):

    binned_data = []

    for i in range(len(impact_bins_edges)-1):

        imp_edge_min = impact_bins_edges[i]
        imp_edge_max = impact_bins_edges[i+1]

        data_impactbinned = data[(impact_parameter >= imp_edge_min) &
                                     (impact_parameter < imp_edge_max)]
        size_impactbinned = size[(impact_parameter >= imp_edge_min) &
                                 (impact_parameter < imp_edge_max)]

        for j in range(len(size_bins_edges)-1):
            
            siz_edge_min = size_bins_edges[j]
            siz_edge_max = size_bins_edges[j+1]
            
            if len(data_impactbinned[(size_impactbinned >= siz_edge_min) &
                   (size_impactbinned < siz_edge_max)]) > 0:

                mean_data = np.mean(data_impactbinned[
                    (size_impactbinned >= siz_edge_min) &
                    (size_impactbinned < siz_edge_max)
                    ])
                std_data = np.std(data_impactbinned[
                    (size_impactbinned >= siz_edge_min) &
                    (size_impactbinned < siz_edge_max)
                    ])
                n_data = len(data_impactbinned[
                    (size_impactbinned >= siz_edge_min) &
                    (size_impactbinned < siz_edge_max)
                    ])
            else:
                mean_data = np.nan
                std_data = np.nan
                n_data = np.nan

            binned_data.append(
                ((imp_edge_max-imp_edge_min)/2.0 + imp_edge_min,
                (siz_edge_max-siz_edge_min)/2.0 + siz_edge_min,
                mean_data, std_data, n_data))

    binned_data = np.array(binned_data,
                             dtype=[('impact', 'f8'),
                             ('size', 'f8'),
                             ('mean', 'f8'),
                             ('std', 'f8'),
                             ('n_data', 'f8')])

    return binned_data
