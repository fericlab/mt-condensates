import czifile as czifile
import matplotlib.pyplot as plt
import trackpy as tp
import pandas as pd
import numpy as np

plt.rcParams['pdf.fonttype'] = 42 

# -------- USER SETTINGS (edit these) --------
CZI_PATH = r'C:/Users/Nidhi/OneDrive - The Pennsylvania State University/Lab work/Research/analysis/tracking_script/20240904_HeLa_live_FASTKD2_mScarlet_pg_W2_63x_AL_8x_60cyc_2sint-08_processed_aligned_annotated-ID02-Gauss-s1.6.czi'
FRAME_INTERVAL_S = 2              # seconds per frame
PX_TO_UM = 0.0328                 # microns per pixel 
DIAMETER = (17, 15)               # detection window (odd ints; (dx, dy) allowed)
MEMORY = 5                        # linker memory (frames)

# IDs to use for the single-particle distance computation
PID_CH0 = 0   # <- set the desired particle ID from channel 0
PID_CH1 = 2   # <- set the desired particle ID from channel 1
# --------------------------------------------

file = czifile.imread(CZI_PATH)
print("CZI shape:", file.shape)

def tracking(channel, p_threshold, search_R, diameter, edge_qc_frame=None):
    """Locate + link for one channel (edge-safe via padding). Returns (linked_df, frames)."""
    tp_df = pd.DataFrame()
    frames = file[0,0, :, channel, 0, :, :, 0]

    # derive padding radius from diameter (supports int or (dx,dy))
    if isinstance(diameter, tuple):
        dx, dy = int(diameter[0]), int(diameter[1])
    else:
        dx = dy = int(diameter)
    rad_x = dx // 2
    rad_y = dy // 2

    # STEP 1: Locate (with padding)
    for i in range(len(frames)):
        image = frames[i]
        thr_value = tp.find.percentile_threshold(image, p_threshold)

        # reflect-pad so the detection window fits at edges
        image_p = np.pad(image, ((rad_y, rad_y), (rad_x, rad_x)), mode='reflect')

        f = tp.locate(image_p, diameter=diameter, threshold=thr_value, preprocess=True)

        # shift back to original coords
        f['x'] = f['x'] - rad_x
        f['y'] = f['y'] - rad_y

        # keep only detections that lie inside the real image
        h, w = image.shape
        f = f[(f['x'] >= 0) & (f['x'] < w) & (f['y'] >= 0) & (f['y'] < h)]

        # metadata
        f['frame'] = int(i)
        f['time']  = i * FRAME_INTERVAL_S
        tp_df = pd.concat([tp_df, f], ignore_index=True)

        # (optional) quick per-frame annotate/QC
        tp.annotate(f, image, plot_style={'markersize': 15})
        plt.imshow(image, cmap='gray'); plt.show()

        # (optional) one-time QC showing padded vs original alignment
        if edge_qc_frame is not None and i == edge_qc_frame:
            plt.figure(figsize=(10,4))
            # left: padded coords
            plt.subplot(1,2,1); plt.imshow(image_p, cmap='gray')
            plt.scatter(f['x']+rad_x, f['y']+rad_y, s=60, facecolors='none', edgecolors='r')
            plt.title("Detections in padded coords"); plt.gca().invert_yaxis()
            # right: shifted back
            plt.subplot(1,2,2); plt.imshow(image, cmap='gray')
            plt.scatter(f['x'], f['y'], s=60, facecolors='none', edgecolors='lime')
            plt.title("Detections in original coords"); plt.gca().invert_yaxis()
            plt.show()

    # STEP 2: LINK
    linked = tp.link(tp_df, search_range=search_R, memory=MEMORY)

    # Save full trajectories CSV
    linked.to_csv(f"trajectories_channel{channel}.csv", index=False)

    # Plot & save overlay of all trajectories
    plt.figure()
    tp.plot_traj(linked, superimpose=frames[0], label=False)
    plt.title(f"TrackPy trajectories – Channel {channel}")
    #plt.savefig(f"tp_traj_channel{channel}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Save each particle: CSV + PNG over first-appearance frame
    for pid, tr in linked.groupby('particle'):
        tr = tr.sort_values('frame').copy()
        # add µm coords (useful downstream)
        tr['x_um'] = tr['x'] * PX_TO_UM
        tr['y_um'] = tr['y'] * PX_TO_UM

        tr.to_csv(f"track_pid{int(pid)}_channel{channel}.csv", index=False)

        first_f = int(tr['frame'].min())
        bg = frames[first_f]
        plt.figure()
        plt.imshow(bg, cmap='gray')
        plt.plot(tr['x'], tr['y'], linewidth=2)
        plt.plot(tr.iloc[0]['x'], tr.iloc[0]['y'], 'o', markersize=6)
        plt.plot(tr.iloc[-1]['x'], tr.iloc[-1]['y'], 'x', markersize=8)
        plt.gca().invert_yaxis()
        plt.title(f"Particle {int(pid)} – Channel {channel} (len={len(tr)})")
        plt.xlabel("x (px)"); plt.ylabel("y (px)")
        plt.savefig(f"channel{channel}_track_pid{int(pid)}.png", dpi=300, bbox_inches='tight')
        plt.close()

    return linked, frames

# ---- Run tracking
ch0_linked, ch0_frames = tracking(channel=0, p_threshold=60, diameter=(17,15), search_R=16)
ch1_linked, ch1_frames = tracking(channel=1, p_threshold=70, diameter=(17,15), search_R=16)

# ---- Select ONLY the specified particles for distance computation ----
ch0_particle = ch0_linked[ch0_linked['particle'] == PID_CH0].sort_values('frame').copy()
ch1_particle = ch1_linked[ch1_linked['particle'] == PID_CH1].sort_values('frame').copy()

# ---- Merge and compute distance (µm) vs time (s) ----
merged = pd.merge(ch0_particle, ch1_particle, on='frame', how='inner', suffixes=('0','1'))
merged['distance_px'] = np.sqrt((merged['x1'] - merged['x0'])**2 + (merged['y1'] - merged['y0'])**2)
merged['distance_um'] = merged['distance_px'] * PX_TO_UM

plt.figure()
fig, ax = plt.subplots()

# --- plot data first ---
ax.plot(
    merged['time0'], merged['distance_um'],
    color='black', linewidth=2, alpha=0.85, zorder=2
)
ax.scatter(
    merged['time0'], merged['distance_um'],
    facecolors='#71797E', edgecolors='black', linewidths=0.8, s=80, zorder=3
)

# --- set & freeze y-limits from data BEFORE shading ---
ymin_data = 0.0
ymax_data = max(0.3, merged['distance_um'].max() * 1.10)  
ax.set_ylim(ymin_data, ymax_data)
ax.autoscale(enable=False, axis='y')  # freeze y autoscale so hspans won't move it

# --- background shading using your colors ---
ax.axhspan(0.0, 0.1, facecolor='#bdbdbd', alpha=0.6, zorder=0)  # dark gray (mixed)
ax.axhspan(0.1, 0.2, facecolor='#d9d9d9', alpha=0.6, zorder=0)  # medium gray (wetting)
ax.axhspan(0.2, ymax_data, facecolor='#eeeeee', alpha=0.6, zorder=0)  # light gray (demixed)

# --- labels, ticks, title ---
ax.set_xlabel('time (s)', fontsize=18)
ax.set_ylabel('distance (µm)', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_title(f"Distance vs time (ch0 pid={PID_CH0}, ch1 pid={PID_CH1})", fontsize=18)

plt.show()


fig.savefig("distance_vs_time.png", dpi=300, bbox_inches='tight')
fig.savefig("distance_vs_time.eps", bbox_inches='tight')
fig.savefig("distance_vs_time.pdf", bbox_inches='tight')
plt.show()
merged.to_csv(f"channel0_pid{PID_CH0}_channel1_pid{PID_CH1}_distance_um.csv", index=False)


