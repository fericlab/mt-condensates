import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Define file paths
file_paths = glob.glob(r'C:/Users/Nidhi/OneDrive - The Pennsylvania State University/Lab work/Research/analysis/intensity line profile/IMT1B experiment/plotting script/raw data/*.csv')

# Define the 3 channels to analyze
channels = ['mtDNA','mtRNA', 'MRG', ]
colors = ['#FFDE21', '#e800cc', '#23D5D5', ]
markers = ['o', 's', '^']  # Different marker for each channel

# Initialize a dictionary to track the global maximum intensity per channel
global_max = {channel: 0 for channel in channels}
data_frames = {}

for file in file_paths:
    df = pd.read_csv(file)
    data_frames[file] = df
    
    # Store max intensity per channel in current file
    channel_max_values = {}

    for channel in channels:
        if channel in df.columns:
            channel_max_values[channel] = df[channel].max()

    for channel, max_value in channel_max_values.items():
        global_max[channel] = max(global_max[channel], max_value)
    
    print(f"Max intensities for {os.path.basename(file)}: {channel_max_values}")

# Generate separate plots for each file
for file in file_paths:
    df = data_frames[file]
    file_name = os.path.basename(file).replace('.csv', '')

    plt.figure(figsize=(5, 4))  # Slightly larger figure
    plt.rcParams['pdf.fonttype'] = 42
    font_properties = {'family': 'Arial', 'size': 14, 'weight': 'normal', 'style': 'normal'}

    for i, channel in enumerate(channels):
        if channel not in df.columns:
            print(f"Skipping {channel} in {file} (Column not found)")
            continue

        intensities = df[channel].dropna()
        if intensities.empty:
            print(f"Skipping {channel} in {file} (No valid data)")
            continue

        x = df.iloc[:len(intensities), 0]
        norm_intensities = intensities / global_max[channel]

        # Plot with markers
        plt.plot(
            x, norm_intensities,
            label=channel,
            alpha=1,
            linewidth=2,
            color=colors[i],
            marker=markers[i],
            markersize=10,
            markerfacecolor=colors[i],    
            markeredgecolor=colors[i],    
            markeredgewidth=1
        )
        # Label and legend handling
        plt.xlabel('Distance (µm)', **font_properties)
        plt.ylabel('Normalized intensity (a.u.)', **font_properties)
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.00),
            ncol=len(channels),
            frameon=False,
            fontsize=12,
            handletextpad=0.6,   # reduce spacing between marker and text
            columnspacing=1.0    # reduce space between columns
            )
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.ylim(0, 1.3)
        plt.tight_layout()
 
    # Adjust internal margins before saving
    plt.subplots_adjust(left=0.18, right=0.95, top=0.85, bottom=0.18)

    # Save the figure
    output_filename = f"{file_name}_ind-norm_intensity_plot"
    plt.savefig(f'{output_filename}.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_filename}.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_filename}.pdf and {output_filename}.eps")
