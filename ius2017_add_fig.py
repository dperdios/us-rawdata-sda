import os
import pickle
import matplotlib.pyplot as plt
import matplotlib

# Load computed metrics (i.e. ius2017_results.py MUST have been processed before)
metrics_path = os.path.join('results', 'ius2017')
file_list = [f for f in os.listdir(path=metrics_path) if f.endswith('_metrics.pickle')]
file_list.sort()
metrics_list = []
for file in file_list:
    with open(os.path.join(metrics_path, file), 'rb') as handle:
        metrics_list.append(pickle.load(handle))

# Computed metrics from the compressed sensing (CS) approach
#   Note: the codes used to generate the CS results cannot be provided for the moment. Sorry for the incovenience.
cs_psnr_list = [[39.24, 36.08, 31.73, 27.13, 22.98, 19.64, 17.74, 16.64, 16.27, 15.70],  # Carotid cross
                [31.56, 30.01, 27.87, 23.79, 20.61, 18.14, 16.14, 15.07, 14.55, 14.17],  # Carotid long
                [33.12, 30.50, 27.51, 24.88, 22.42, 20.54, 19.25, 18.42, 17.73, 16.84],  # In vitro type 1
                [33.30, 30.29, 26.94, 24.00, 21.71, 19.76, 18.67, 18.03, 17.65, 17.37],  # In vitro type 2
                [33.38, 30.31, 27.27, 24.44, 22.21, 20.46, 19.22, 18.48, 17.90, 17.33],  # In vitro type 3
                [33.21, 30.94, 28.21, 25.26, 22.51, 20.07, 17.93, 16.39, 14.83, 13.83]   # Numerical
                ]

# Find max and min PSNR values for the figure axes
max_psnr = max(max(map(max, cs_psnr_list)), max([m['bmode_psnr'] for metrics in metrics_list for m in metrics]))
min_psnr = min(min(map(min, cs_psnr_list)), min([m['bmode_psnr'] for metrics in metrics_list for m in metrics]))

# Extract metrics for each PICMUS test case
filenames = [n.replace('.pickle', '.pdf') for n in file_list]

for metrics, cs_bmode_psnr, fn in zip(metrics_list, cs_psnr_list, filenames):
    cp_list = [m['model_config']['compression_percent'] for m in metrics if m['model_config']['learn_mm']]
    undersampling_ratios = [(100 - cp) / 100 for cp in cp_list]
    # PSNR
    sda_cl_bmode_psnr = [m['bmode_psnr'] for m in metrics if m['model_config']['learn_mm']]
    sda_cnl_bmode_psnr = [m['bmode_psnr'] for m in metrics if not m['model_config']['learn_mm']]
    # Figures
    matplotlib.rcParams.update({'font.size': 14})
    linewidth = 3
    marker = '.'
    markersize = 10
    plt.figure()
    plt.plot(undersampling_ratios, sda_cl_bmode_psnr, label='SDA–CL', linewidth=linewidth, marker=marker, markersize=markersize)
    plt.plot(undersampling_ratios, sda_cnl_bmode_psnr, label='SDA–CNL', linewidth=linewidth, marker=marker, markersize=markersize)
    plt.plot(undersampling_ratios, cs_bmode_psnr, label='CS', linewidth=linewidth, marker=marker, markersize=markersize)
    plt.grid()
    plt.legend(loc='upper left')
    plt.ylim([int(min_psnr), int(max_psnr) + 1])
    ax = plt.gca()
    ax.set_xlabel('Undersampling ratio M/N [–]')
    ax.set_ylabel('PSNR [dB]')
    # Save figure
    plt.savefig(os.path.join(metrics_path, fn))
