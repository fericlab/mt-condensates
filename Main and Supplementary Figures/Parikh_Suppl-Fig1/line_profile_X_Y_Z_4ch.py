import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import glob
import os
import re


def lighten_color(hex_color, amount=0.55):
    """Returns a lighter (pastel) tint of hex_color by blending it toward
    white. amount=0 returns the original color, amount=1 returns white."""
    rgb = mcolors.to_rgb(hex_color)
    return tuple(c + (1.0 - c) * amount for c in rgb)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Folder containing your exported profile files, all together, named like:
#   X_bead1.csv / X_bead1.xlsx        -> before alignment, horizontal (X) line profile
#   X_bead1_aligned.csv / .xlsx       -> after alignment
#   Y_bead1.xlsx, Y_bead1_aligned.xlsx   -> vertical (Y) line profile
#   Z_bead1.xlsx, Z_bead1_aligned.xlsx   -> fixed-point axial (Z) profile
# Both .csv and .xlsx are supported, and can be mixed within the same folder.
# Each file needs a Distance column and one <channel>_Gray_Value column per
# channel, e.g.: Distance_(microns), 647_Gray_Value, 561_Gray_Value, 488_Gray_Value, 405_Gray_Value
base_dir = r'D:/OneDrive - The Pennsylvania State University/Lab work/Paper #1 Revisions/analysis/bead line profile/20260815/profiles'

# Define the 4 channels to analyze (beads imaged in 4 laser lines)
channels = ['405', '488', '561', '647']

# Fluorescence pseudo-colors for each channel
colors = ['#4363d8', '#3cb44b', '#e6194B', '#e800cc']  # 405=blue, 488=green, 561=red, 647=magenta

markers = ['o', 's', '^', 'D']  # Different marker for each channel
marker_size = 5  # shared marker size, used identically on individual and before/after plots
line_width = 1.3  # shared curve line thickness

font_properties = {'family': 'Arial', 'size': 14, 'weight': 'normal', 'style': 'normal'}

# Which profile-type prefixes to process, and the x-axis label to use for each.
# X and Y are lateral (in-plane) line profiles; Z is the fixed-point axial profile.
PROFILE_TYPES = {
    'X': 'X distance (nm)',
    'Y': 'Y distance (nm)',
    'Z': 'Axial distance, Z (nm)',
}


# ---------------------------------------------------------------------------
# FILE LOADING / COLUMN NORMALIZATION
# ---------------------------------------------------------------------------

def load_profile_file(path):
    """Reads a .csv or .xlsx profile export and returns a DataFrame with
    standardized columns: 'Distance' (in nm) plus one column per channel
    present (named '405', '488', '561', '647').

    Handles both export formats without manual editing:
      - Line profiles (X/Y): distance column 'Distance_(microns)' and channel
        columns '<ch>_Gray_Value' (optionally with a '_z<slice>' suffix).
      - Z-axis profiles: distance column '[micron]' and channel columns
        '<ch>_Mean' (Fiji's Plot Z-axis Profile reports mean intensity)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(path)
    elif ext in ('.xlsx', '.xls'):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    rename_map = {}
    distance_col = None
    for col in df.columns:
        cl = col.strip().lower()
        if cl.startswith('distance') or 'micron' in cl:
            distance_col = col
            continue
        # Accept '<ch>_Gray_Value' (line profiles) or '<ch>_Mean' (Z profiles).
        m = re.match(r'^(\d{3})_(?:Gray_Value|Mean)', col.strip())
        if m:
            rename_map[col] = m.group(1)

    if distance_col is None:
        raise ValueError(f"Couldn't find a Distance column in: {path} (columns: {list(df.columns)})")

    df = df.rename(columns=rename_map)
    df = df.rename(columns={distance_col: 'Distance'})

    # Convert microns -> nm to match the plot's nm axis.
    df['Distance'] = df['Distance'] * 1000

    keep_cols = ['Distance'] + [c for c in channels if c in df.columns]
    return df[keep_cols]


def discover_files(folder_path, prefix):
    """Finds every '<prefix>_<basename>.(csv|xlsx)' file in folder_path and
    returns {basename: filepath}, e.g. {'bead1': '.../X_bead1.csv',
    'bead1_aligned': '.../X_bead1_aligned.xlsx'}."""
    found = {}
    for ext in ('csv', 'xlsx', 'xls'):
        pattern = os.path.join(folder_path, f'{prefix}_*.{ext}')
        for file in glob.glob(pattern):
            base = os.path.basename(file)
            base = base[len(prefix) + 1:]
            base = re.sub(r'\.(csv|xlsx|xls)$', '', base, flags=re.IGNORECASE)
            found[base] = file
    return found


# ---------------------------------------------------------------------------
# PEAK LABELING (shared by both individual and before/after plots, so both
# plot types annotate peak nm values identically)
# ---------------------------------------------------------------------------

def label_cluster_threshold(x_range):
    """The nm gap below which two peaks are considered 'too close together'
    for separate inline text labels. Scaled to the plotted x-range (rather
    than a fixed nm value) because whether two labels visually collide
    depends on how much plot width each nm occupies -- e.g. X/Y (lateral)
    profiles span only ~300 nm while Z (axial) profiles span ~1000+ nm, so a
    fixed nm threshold would be far too strict for one and far too loose for
    the other."""
    return max(20, 0.12 * x_range)


def cluster_peaks(peak_values, threshold):
    """Groups a {channel: peak_x} dict into clusters of channels whose peaks
    sit within `threshold` (nm) of each other, sorted by peak position."""
    items = sorted(peak_values.items(), key=lambda kv: kv[1])
    clusters = []
    for item in items:
        if clusters and abs(item[1] - clusters[-1][-1][1]) <= threshold:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    return clusters


def decide_label_style(peak_value_dicts, thresholds):
    """Looks at one or more {channel: peak_x} dicts (one per panel -- a
    single-item list for an individual plot, two for a before/after pair
    that shares a y-axis) plus each panel's own label_cluster_threshold, and
    decides, once for all of them, whether peaks are spread out enough for
    inline labels to stay readable, or whether a corner list is needed
    instead. Returns (use_inline_labels, ylim_top)."""
    max_stack = max(
        (len(cluster)
         for peak_values, threshold in zip(peak_value_dicts, thresholds)
         for cluster in cluster_peaks(peak_values, threshold)),
        default=1,
    )
    use_inline_labels = max_stack <= 2
    ylim_top = 1.35 if use_inline_labels else 1.125
    return use_inline_labels, ylim_top


def draw_peak_labels(ax, peak_values, use_inline_labels, threshold):
    """Draws each channel's peak position as a thin dashed guide line, plus
    its nm value -- either as an inline label near the top of the curve
    (used when peaks are spread out enough to stay legible) or as a compact
    color-coded list in the corner of the panel (used when peaks cluster too
    closely together for inline labels to avoid overlapping)."""
    if use_inline_labels:
        for cluster in cluster_peaks(peak_values, threshold):
            base_y = 1.30
            for j, (channel, peak_x) in enumerate(cluster):
                i = channels.index(channel)
                label_y = base_y - 0.09 * j
                ax.plot([peak_x, peak_x], [0, label_y - 0.03], linestyle='--',
                        linewidth=1, color='0.55', zorder=1)
                ax.text(peak_x, label_y, f'{peak_x:.0f} nm', color=colors[i],
                        fontsize=8.5, ha='center', va='bottom')
    else:
        for i, channel in enumerate(channels):
            if channel not in peak_values:
                continue
            peak_x = peak_values[channel]
            ax.plot([peak_x, peak_x], [0, 1.0], linestyle='--', linewidth=1,
                    color='0.55', zorder=1)
        for i, channel in enumerate(channels):
            if channel not in peak_values:
                continue
            ax.text(
                0.03, 0.97 - 0.075 * i,
                f'{channel}: {peak_values[channel]:.0f} nm',
                color=colors[i], fontsize=9, ha='left', va='top',
                transform=ax.transAxes,
            )


def make_channel_legend_handles():
    """Builds the shared color-coded channel legend handles (one per
    channel), used identically on individual and before/after plots. Each
    handle shows the channel's line AND its marker, so the legend matches
    the curves in the plot."""
    return [
        mlines.Line2D([], [], color=colors[i], linestyle='-', linewidth=line_width + 0.5,
                      marker=markers[i], markersize=marker_size,
                      markerfacecolor=colors[i], markeredgecolor=colors[i],
                      label=channel)
        for i, channel in enumerate(channels)
    ]


# ---------------------------------------------------------------------------
# CORE PIPELINE (used for each of X, Y, Z)
# ---------------------------------------------------------------------------

def process_profile_type(folder_path, prefix, x_label):
    """Runs the full pipeline (individual plots + before/after overlay plots)
    for every '<prefix>_<basename>.(csv|xlsx)' file found in folder_path."""
    file_map = discover_files(folder_path, prefix)
    if not file_map:
        print(f"[{prefix}] No '{prefix}_<basename>.(csv|xlsx)' files found in: {folder_path}")
        return

    # ---- Pass 1: load every file once, print raw per-channel max intensities ----
    # Each channel is normalized to its own maximum within its own file, so
    # every plot shows every channel peaking at 1.0, keeping the focus on
    # peak position rather than absolute brightness. The printed values below
    # are each file's raw (un-normalized) per-channel peak intensity.
    data_frames = {}

    for base, file in file_map.items():
        df = load_profile_file(file)
        data_frames[base] = df

        channel_max_values = {channel: df[channel].max() for channel in channels if channel in df.columns}
        print(f"[{prefix}] Max intensities for {base}: {channel_max_values}")

    # ---- Part 1: individual plots (one plot per file, 4 channels) ----
    for base, df in data_frames.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        plt.rcParams['pdf.fonttype'] = 42

        peak_values = {}  # channel -> peak_x, for the shared peak-labeling helpers

        for i, channel in enumerate(channels):
            if channel not in df.columns:
                print(f"[{prefix}] Skipping {channel} in {base} (Column not found)")
                continue

            intensities = df[channel].dropna()
            if intensities.empty:
                print(f"[{prefix}] Skipping {channel} in {base} (No valid data)")
                continue

            x = df['Distance'].iloc[:len(intensities)]
            norm_intensities = intensities / intensities.max()
            marker_every = max(1, len(x) // 12)  # same marker density as the before/after panels

            ax.plot(
                x, norm_intensities,
                label=channel,
                alpha=1,
                linewidth=line_width,
                color=colors[i],
                marker=markers[i],
                markersize=marker_size,
                markevery=marker_every,
                markerfacecolor=colors[i],
                markeredgecolor=colors[i],
                markeredgewidth=1
            )

            peak_idx = int(norm_intensities.values.argmax())
            peak_x = x.iloc[peak_idx]
            peak_values[channel] = peak_x

        # Same peak nm annotation logic as the before/after plots: inline
        # labels when peaks are spread out, a corner list when they cluster.
        threshold = label_cluster_threshold(df['Distance'].max())
        use_inline_labels, ylim_top = decide_label_style([peak_values], [threshold])
        draw_peak_labels(ax, peak_values, use_inline_labels, threshold)

        ax.set_xlabel(x_label, **font_properties)
        ax.set_ylabel('Normalized intensity (a.u.)', **font_properties)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylim(0, ylim_top)

        # Legend lives outside the axes (figure-level) so it doesn't
        # compete with the peak labels for space at the top of the plot.
        channel_legend = fig.legend(
            handles=make_channel_legend_handles(),
            loc='upper center',
            bbox_to_anchor=(0.5, 1.05),
            ncol=len(channels),
            frameon=False,
            fontsize=12,
            handletextpad=0.6,
            columnspacing=1.0,
        )

        plt.tight_layout()
        fig.subplots_adjust(top=0.82)

        output_filename = f"{prefix}_{base}_ind-norm_intensity_plot"
        plt.savefig(f'{output_filename}.pdf', format='pdf', dpi=300, bbox_inches='tight',
                    bbox_extra_artists=(channel_legend,))
        plt.savefig(f'{output_filename}.eps', format='eps', dpi=300, bbox_inches='tight',
                    bbox_extra_artists=(channel_legend,))
        plt.show()
        print(f"Saved: {output_filename}.pdf and {output_filename}.eps")

    # ---- Part 2: before/after overlay plots, paired by '_aligned' suffix ----
    before_bases = {}
    after_bases = {}

    for base in data_frames:
        if base.endswith('_aligned'):
            bead_id = base[: -len('_aligned')]
            after_bases[bead_id] = base
        else:
            before_bases[base] = base

    paired_beads = sorted(set(before_bases) & set(after_bases))
    unmatched_before = sorted(set(before_bases) - set(after_bases))
    unmatched_after = sorted(set(after_bases) - set(before_bases))

    if unmatched_before:
        print(f"[{prefix}] No '_aligned' match found for: {unmatched_before}")
    if unmatched_after:
        print(f"[{prefix}] No unaligned match found for: {unmatched_after}")

    for bead_id in paired_beads:
        df_before = data_frames[before_bases[bead_id]]
        df_after = data_frames[after_bases[bead_id]]

        plt.rcParams['pdf.fonttype'] = 42
        fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)

        panels = [(ax_before, df_before, 'Before alignment'), (ax_after, df_after, 'After alignment')]
        panel_peaks = []  # one {channel: peak_x} dict per panel
        panel_thresholds = []  # each panel's own label_cluster_threshold (x-axes aren't shared)

        for ax, df, panel_title in panels:
            peak_values = {}
            panel_thresholds.append(label_cluster_threshold(df['Distance'].max()))

            for i, channel in enumerate(channels):
                if channel not in df.columns:
                    continue
                intensities = df[channel].dropna()
                if intensities.empty:
                    continue
                x = df['Distance'].iloc[:len(intensities)]
                norm_intensities = intensities / intensities.max()
                marker_every = max(1, len(x) // 12)  # dispersed markers, regardless of point count

                ax.plot(
                    x, norm_intensities,
                    alpha=1,
                    linewidth=line_width,
                    linestyle='-',
                    color=colors[i],
                    marker=markers[i],
                    markersize=marker_size,
                    markevery=marker_every,
                    markerfacecolor=colors[i],
                    markeredgecolor=colors[i],
                    markeredgewidth=1,
                )

                peak_idx = int(norm_intensities.values.argmax())
                peak_x = x.iloc[peak_idx]
                peak_values[channel] = peak_x

            panel_peaks.append(peak_values)

            ax.set_title(panel_title, fontsize=13, **{k: v for k, v in font_properties.items() if k != 'size'})
            ax.set_xlabel(x_label, **font_properties)
            ax.tick_params(axis='both', which='major', labelsize=14)

        # Both panels share a y-axis, so the peak-labeling style/limits have
        # to be decided once for the pair -- same shared logic used for the
        # individual plots above.
        use_inline_labels, ylim_top = decide_label_style(panel_peaks, panel_thresholds)

        for (ax, df, panel_title), peak_values, threshold in zip(panels, panel_peaks, panel_thresholds):
            draw_peak_labels(ax, peak_values, use_inline_labels, threshold)
            ax.set_ylim(0, ylim_top)

        ax_before.set_ylabel('Normalized intensity (a.u.)', **font_properties)

        channel_legend = fig.legend(
            handles=make_channel_legend_handles(),
            loc='upper center',
            bbox_to_anchor=(0.5, 1.08),
            ncol=len(channels),
            frameon=False,
            fontsize=11,
            handletextpad=0.5,
            columnspacing=0.9
        )

        plt.tight_layout()
        fig.subplots_adjust(top=0.82, wspace=0.08)

        output_filename = f"{prefix}_{bead_id}_before-after-norm_intensity_plot"
        plt.savefig(f'{output_filename}.pdf', format='pdf', dpi=300, bbox_inches='tight',
                    bbox_extra_artists=(channel_legend,))
        plt.savefig(f'{output_filename}.eps', format='eps', dpi=300, bbox_inches='tight',
                    bbox_extra_artists=(channel_legend,))
        plt.show()
        print(f"Saved: {output_filename}.pdf and {output_filename}.eps")


# ---------------------------------------------------------------------------
# RUN: once for each of X, Y, Z
# ---------------------------------------------------------------------------

for prefix, x_label in PROFILE_TYPES.items():
    process_profile_type(base_dir, prefix, x_label)