import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import tool_functions as tf
import plotly.graph_objects as go
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import pandas as pd
import paths_and_flags as pf
from sklearn.metrics import r2_score
from matplotlib.widgets import Button
import copy
from pybaselines.whittaker import asls
from prettyprinter import pprint

""" Analyse the reconstructed data. 
Derive wave propagation in cranial-caudal and left-right directions"""

# Plot the 2D points
# define polylines
# calculate distance to polylines find closest points
# find the principle frequency in all points with fft
# get wave propagation from phase shift
# get temporal parameters along each point
proceed_with_DP = False  # Helper bool to get feedback for the last calculation


class Analyse:
    def __init__(self, pnts, frame):
        self.points = pnts
        self.ref_frame = frame
        self.polylines = []  # [polyline] [node on line] [0 = x, 1 = y]
        self.wave_propagation_pts = []  # [with respect to line] [point number]
        self.calculation = {}

    def create_polylines(self):
        fig, ax = tf.plt.subplots()
        ax.plot(self.points[:, 0, self.ref_frame],
                self.points[:, 1, self.ref_frame], 'k.', markersize=6)
        ax.set_aspect('equal')

        current_line = []
        lines = []

        def onclick(event):
            nonlocal current_line, lines

            if event.button == 1:
                color = np.random.random(3)
                current_line.append((event.xdata, event.ydata))
                if len(current_line) == 1:
                    line, = ax.plot([], [], color=color, marker='o')
                    lines.append(line)
                lines[-1].set_data(zip(*current_line))
                fig.canvas.draw()

            elif event.button == 3 and current_line:
                self.polylines.append(current_line[:])
                current_line = []

        fig.canvas.mpl_connect('button_press_event', onclick)
        tf.plt.show()
        current_line = []

    def wave_propagation(self, exp_path):
        self.redraw_pnts(exp_path)
        for line in tf.tqdm(self.wave_propagation_pts, 'line analysis'):
            wave = []
            br_rate_m = []
            heart_rate_m = []

            for pnt in tf.tqdm(line, 'points in line'):
                # Filter in frequency domain and get parameters
                self.calculation = {}
                br_rate, phase, heart_rate, filtered_signal= get_parameters_in_frequency(self.points[pnt, 2, :],
                                                                                          pnt, exp_path,
                                                                                          Verbose=True)
                wave.append(phase)
                br_rate_m.append(br_rate)
                heart_rate_m.append(heart_rate)

                # Get other parameters in temporal domain
                ev, time = get_parameters_in_temporal(filtered_signal, pnt, exp_path, br=br_rate,
                                                          Verbose=True)
                self.calculation['Time [ms]'] = time  # Time in ms
                self.calculation['Displacement [mm]'] = filtered_signal  # Z displacement

                self.calculation['Ti [ms]'] = [e.Ti for e in ev]  # Total inspiration time
                self.calculation['Ti_slope'] = [e.Ti_slope for e in ev]  # Inspiration slope
                self.calculation['Ti_fit'] = [e.Ti_fit for e in ev]  # R^2 linear fit inspiration
                self.calculation['Ti_AuC [mm*ms]'] = [e.Ti_AuC for e in ev]  # Inspiration area under curve

                self.calculation['Tp_[ms]'] = [e.Tp for e in ev]
                self.calculation['Tp_AuC [mm*ms]'] = [e.Tp_AuC for e in ev]

                self.calculation['Te [ms]'] = [e.Te for e in ev]  # Total expiration time
                self.calculation['Te_fit'] = [e.Te_fit for e in ev]  # R^2 linear fit inspiration
                self.calculation['Te_AuC [mm*ms]'] = [e.Te_AuC for e in ev]  # Inspiration area under curve

                self.calculation['Te_FE [ms]'] = [e.Te_FE for e in ev]  # Total inspiration time
                self.calculation['Te_FE_slope'] = [e.Te_FE_slope for e in ev]  # Inspiration slope
                self.calculation['Te_FE_fit'] = [e.Te_FE_fit for e in ev]  # R^2 linear fit inspiration
                self.calculation['Te_FE_AuC [mm*ms]'] = [e.Te_FE_AuC for e in ev]  # Inspiration area under curve

                self.calculation['Te_SE [ms]'] = [e.Te_SE for e in ev]  # Total inspiration time
                self.calculation['Te_SE_ppt [a, t, c]'] = [e.Te_SE_ppt for e in ev]  # Inspiration slope
                self.calculation['Te_SE_T1/2 [ms]'] = [e.Te_SE_half for e in ev]  # Inspiration slope
                self.calculation['Te_SE_fit'] = [e.Te_SE_fit for e in ev]  # R^2 linear fit inspiration
                self.calculation['Te_SE_AuC [mm*ms]'] = [e.Te_SE_AuC for e in ev]  # Inspiration area under curve

                output = cleanup_dataframe(self.calculation)
                output.to_csv(exp_path + f"{pnt}_analysis.csv")
                # pprint(self.calculation)
                # clean_dict = {k: v for k, v in self.calculation.items() if v is not None}
            # Calculate wave
            wave = np.asarray(wave)
            wave = np.unwrap(wave)
            wave = wave - wave[0]
            wave = np.rad2deg(wave)

            self.calculation['BR [Hz]'] = br_rate_m
            self.calculation['HR [bpm]'] = heart_rate_m
            self.calculation['Wave propagation [deg]'] = wave
            output_last = cleanup_dataframe(self.calculation)
            output_last.to_csv(exp_path + f"{pnt}_last_analysis.csv")
        return print('Finished')

    def draw(self):
        fig = go.Figure()
        for line in self.wave_propagation_pts:
            for idx, pnt in enumerate(line):
                time = np.arange(self.points.shape[2])
                loc = np.full(time.shape, fill_value=idx)
                disp = self.points[pnt, 2, :]
                fig.add_trace(go.Scatter3d(x=loc, y=time, z=disp, mode='lines', marker=dict(size=1)))
            # fig.update_layout(width=800, height=700, autosize=True)
            fig.show()

    def redraw_pnts(self, export_path):
        x_vals = [item[0][self.ref_frame] for item in self.points]
        y_vals = [item[1][self.ref_frame] for item in self.points]
        z_vals = [item[2][self.ref_frame] for item in self.points]

        # Plot all the points
        trace_all_points = go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='markers',
            marker=dict(size=3, color='blue')
        )

        # Create scatter plots for the IDs from each line list
        traces_scatter = [trace_all_points]
        colors = ['red', 'green', 'yellow', 'purple', 'magenta', 'orange', 'pink']
        for idx, line in enumerate(self.wave_propagation_pts):
            x_line = [x_vals[i] for i in line]
            y_line = [y_vals[i] for i in line]
            z_line = [z_vals[i] for i in line]

            trace = go.Scatter3d(
                x=x_line,
                y=y_line,
                z=z_line,
                mode='markers+text',
                marker=dict(size=5, color=colors[idx % len(colors)]),
                text=line,
                textposition="top center"
            )
            traces_scatter.append(trace)

        layout = go.Layout(
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=70, y=40, z=10)
            ),

            margin=dict(l=0, r=0, b=0, t=0)
        )

        fig = go.Figure(data=traces_scatter, layout=layout)
        fig.write_html(export_path+"points.html")
        fig.write_json(export_path+"points.json")
        #fig.show()

    def compute_closest_points(self):
        """ This function will search for the closest point to a given segment on the polyline.
        The more segments are in the polyline, the more points will be taken into account for the wave propagation"""
        if len(self.polylines) == 0:
            return 'no polylines were defined, try again'

        # Create a KDTree for efficient nearest neighbor search
        point_coords = self.points[:, :2, self.ref_frame]
        tree = cKDTree(point_coords)

        for line in self.polylines:
            closest_points = []

            for i in range(len(line) - 1):
                segment_start = line[i]
                segment_end = line[i + 1]
                midpoint = ((segment_start[0] + segment_end[0]) / 2, (segment_start[1] + segment_end[1]) / 2)

                # Using query to find the index of the nearest point to the segment's midpoint
                _, idx = tree.query(midpoint)

                closest_points.append(idx)

            self.wave_propagation_pts.append(closest_points)

class Breathing_Event:
    def __init__(self, pnt, peak, interval1, interval2):
        self.pnt = pnt
        self.peak = peak
        self.interv1 = interval1
        self.interv2 = interval2

        self.Ti = None
        self.Ti_fit = None
        self.Ti_slope = None
        self.Ti_AuC = None

        self.Te = None
        self.Te_fit = None
        self.Te_AuC = None

        self.Tp =None
        self.Tp_AuC = None

        self.DP_x = None
        self.DP_y = None

        self.Te_FE = None
        self.Te_FE_fit = None
        self.Te_popt = None
        self.Te_FE_slope = None
        self.Te_FE_AuC = None

        self.Te_SE = None
        self.Te_SE_fit = None
        self.Te_SE_ppt = None
        self.Te_SE_AuC = None
        self.Te_SE_half = None



def point_to_segment_distance(P, A, B):
    """Calculate the Euclidean distance between point P and segment AB."""
    # If the segment is just a point
    if A == B:
        return np.linalg.norm(np.array(P) - np.array(A))
    # if proper segment
    vecAB = np.array(B) - np.array(A)
    vecAP = np.array(P) - np.array(A)

    dot_product = np.dot(vecAP, vecAB)
    len_squared = np.dot(vecAB, vecAB)

    # Check if the projection of point P is outside segment AB
    if dot_product <= 0:
        return np.linalg.norm(vecAP)
    if dot_product >= len_squared:
        return np.linalg.norm(np.array(P) - np.array(B))

    proj_len = dot_product / len_squared
    proj_point = A + proj_len * vecAB
    return np.linalg.norm(np.array(P) - proj_point)


def get_parameters_in_frequency(y, pnt, exp_path, Verbose=False):
    # Smooth the data
    y = tf.adaptiveMovAverage(y, 3)

    # Glue mirrorred version of the data to assure contunuity
    mir = np.concatenate((y, np.flip(y)), axis=None)
    Y = np.fft.fft(mir)
    frequencies = np.fft.fftfreq(len(Y), d=1 / 100)

    # Set aplitude of all frequencies below low_pass_threshold to zero
    low_pass_threshold = 0.6
    low_pass = np.abs(frequencies) >= low_pass_threshold
    Y_lp = copy.deepcopy(Y)
    Y_lp[~low_pass] = 0

    # Inverse back to temporal domain
    filtered_signal = np.fft.ifft(Y_lp).real
    filtered_signal = filtered_signal[:len(y)]
    if Verbose:
        fig, axs = plt.subplots(2, 2, figsize=(18, 12), dpi=300)
        axs[0, 0].plot(y, 'g-x', markersize=3, label='original trace')
        axs[0, 0].set_title('Temporal: ' + str(pnt))
        axs[0, 0].legend()
        axs[0, 0].grid()

        axs[1, 0].plot(filtered_signal, linestyle='--', marker='o', color='b', markersize=3, label="Reconstructed")
        axs[1, 0].legend()
        axs[1, 0].grid()

        axs[0, 0].set(xlabel='Time [frame]', ylabel='Displacement [mm]')
        axs[1, 0].set(xlabel='Time [frame]', ylabel='Displacement [mm]')

        axs[0, 1].plot((frequencies[frequencies > 0]), np.abs(Y[np.nonzero(frequencies > 0)]), 'r-', label='Raw')
        axs[0, 1].set_title('Fourier space ' + str(pnt))
        axs[0, 1].legend()
        axs[0, 1].grid()

        axs[1, 1].plot((frequencies[frequencies > 0]), np.abs(Y_lp[np.nonzero(frequencies > 0)]), 'y-',
                       label=f"Low pass at {low_pass_threshold}")
        axs[1, 1].sharex(axs[0, 1])
        axs[1, 1].sharey(axs[0, 1])
        axs[1, 1].legend()
        axs[1, 1].grid()

        axs[0, 1].set(xlabel='Frequency [Hz]', ylabel='Amplitude')
        axs[1, 1].set(xlabel='Frequency [Hz]', ylabel='Amplitude')
        plt.savefig(exp_path + f"{pnt}_freq.png")
        plt.close(fig)
        # fig.tight_layout()
        # fig.show()

    # Exclude DC component
    non_dc_component = np.abs(Y_lp[1:])

    # Get principal frequency
    principal_frequency_index = np.argmax(non_dc_component) + 1  # Add 1 to account for DC exclusion

    # Look for prominent peaks
    peaks, _ = find_peaks(non_dc_component, height=np.max(non_dc_component) * 0.75, distance=3)
    if len(peaks) >= 1:
        breathing_freq = frequencies[peaks[0] + 1]
        phase = np.angle(Y_lp[peaks[0] + 1])
        # Looking for the heart rate
        principal_amplitude = np.abs(Y_lp[peaks[0] + 1])
        mask = (frequencies >= 6) & (frequencies <= 12)  # Range where we look for the heart rate (360-720 bpm)
        valid_frequencies = frequencies[mask]
        valid_amplitudes = np.abs(Y_lp[mask])
        significant_amplitudes = valid_amplitudes[valid_amplitudes >= 0.3 * principal_amplitude]
        if len(significant_amplitudes) > 0:
            max_amplitude_index = np.argmax(significant_amplitudes)
            heart_rate = valid_frequencies[valid_amplitudes >= 0.3 * principal_amplitude][
                             max_amplitude_index] * 60.0
        else:
            heart_rate = None  # No valid heart rate detected

    # Criteria unsatisfied
    else:
        phase = None
        breathing_freq = None
        heart_rate = None
        # principal_amplitude = np.abs(Y[principal_frequency_index])

    return breathing_freq, phase, heart_rate, filtered_signal


def get_parameters_in_temporal(y, pnt, exp_path, br, Verbose=True):
    events = []
    peak_threshold = 0.21
    event_num = 9

    m_guess = 1.6
    b_guess = 0.2

    p0_guess = 0.0
    p1_guess = 10.0
    p2_guess = 0.1
    p3_guess = 8.6
    p4_guess = 0.7

    baseline, _ = asls(y)
    y_corr = y-baseline
    x = np.arange(0, len(y) / 100.0, 0.01)  # time in ms
    dist = 1.0/br*100
    '''fig, (ax0, ax1) = plt.subplots(figsize=(18, 10), nrows=2)
    ax0.plot(y, 'r-')
    ax0.plot(baseline, 'b-')
    ax1.plot(y_corr, 'g-')
    plt.show()
'''

    peaks, _ = find_peaks(y_corr, height=peak_threshold, distance=60)
    interv = tf.find_intervals(y_corr, peaks, peak_threshold)
    print(peaks)
    print(interv)
    mean_max = np.mean(y_corr[peaks])
    mean_min = np.mean(y_corr[interv])
    y_corr = (y_corr-mean_min)/(mean_max-mean_min)
    if len(peaks) >= event_num and len(interv) >= event_num:
        if Verbose:
            fig, (ax0, ax1) = plt.subplots(figsize=(18, 10), nrows=2)
            ax0.plot(y, 'r-', label='post frequency analysis')
            ax0.plot(baseline, 'b-', label='calculated baseline')
            ax0.plot(peaks, y[peaks], 'ro', markersize=12, label='detected peaks')
            ax0.plot(interv, y[interv], 'rx', markersize=12, label='detected peaks')
            ax0.set(xlabel='Time [frame]', ylabel='Displacement [mm]')
            ax1.set(xlabel='Time [frame]', ylabel='Displacement [mm]')
            ax1.plot(y_corr, 'g-', label='baseline corrected')
            plt.legend()
            plt.savefig(exp_path + f"{pnt}_peaks.png", bbox_inches='tight')
            plt.close(fig)
            #plt.show()

        for idx in range(1, event_num):
            event = Breathing_Event(pnt=pnt, peak = peaks[idx], interval1= interv[idx-1], interval2=interv[idx])
            try:
                #max_peak = np.max(y_corr[peaks[idx]:interv[idx]])
                #min_peak = np.mean(y_corr[peaks[idx]:interv[idx]])
                #exp_seg_y = (y_corr[peaks[idx]:interv[idx]]-min_peak)/(max_peak-min_peak)
                exp_seg_y = y_corr[peaks[idx]:interv[idx]]
                exp_seg_x = np.linspace(0, len(exp_seg_y) / 100.0, len(exp_seg_y))
                popt, pcov = curve_fit(tf.doubleGaussian, exp_seg_x, exp_seg_y, p0=[p0_guess, p1_guess, p2_guess, p3_guess, p4_guess])
                fit_y = tf.doubleGaussian(exp_seg_x, *popt)
                # Get r^2 val
                event.Te_fit = r2_score(exp_seg_y, fit_y)
                event.Te_popt = popt
                event.Te_AuC = np.trapz(y=exp_seg_y, x=exp_seg_x)
                p0_guess = popt[0]
                p1_guess = popt[1]
                p2_guess = popt[2]
                p3_guess = popt[3]
                p4_guess = popt[4]
                events.append(event)
            except Exception as e:
                print('Whole exp fit failed', f"Error - {str(e)}")
                del event

        #Sort events according to expiration fit, pick three best
        events = sorted(events, key=lambda fit: abs(fit.Te_fit - 1))[:3]

        m_guess = -1.0
        b_guess = 0.3

        a_guess = 0.2
        t_guess = 30.0
        c_guess = 0.0
        for e in events:
            # Get inspiration time
            e.Ti = x[e.peak] - x[e.interv1]
            # Get expiration time
            e.Te = x[e.interv2] - x[e.peak]
            height_30 = y_corr[e.peak]*0.30
            range_peak = y_corr[e.interv1:e.interv2]
            range_above = np.argwhere(range_peak>height_30).ravel()
            e.Tp = x[range_above[-1]]-x[range_above[0]]
            range_above = range_peak[range_above].ravel()
            range_x = np.linspace(0, ((len(range_above)) / 100.0), num=len(range_above))
            e.Tp_AuC = np.trapz(y=range_above, x=range_x)

            # Get slope of inspiration and coefficient of determination
            try:
                y_feed = y_corr[e.interv1:e.peak]
                x_feed = np.linspace(0, len(y_feed) / 100.0, len(y_feed))
                popt, pcov = curve_fit(tf.linearFit, x_feed, y_feed, p0=[m_guess, b_guess])
                m_guess = popt[0]
                b_guess = popt[1]
                y_fit = tf.linearFit(x_feed, *popt)
                e.Ti_fit = r2_score(y_feed, y_fit)
                e.Ti_slope = popt[0]
            except:
                print('FA fit failed')
            e.Ti_AuC = np.trapz(y=np.abs(y_feed), x=np.linspace(0, ((len(y_feed)) / 100.0), num=len(y_feed)))

            # Get the dampening point
            exp_seg_y = y_corr[e.peak:e.interv2]
            exp_seg_x = np.linspace(0, len(exp_seg_y) / 100.0, len(exp_seg_y))

            fdy = tf.doubleGfdy(exp_seg_x, e.Te_popt)
            f2dy = tf.doubleGf2dy(exp_seg_x, e.Te_popt)
            f3dy = tf.doubleGf3dy(exp_seg_x, e.Te_popt)
            e.DP_x = exp_seg_x[np.nanargmin(f3dy)]
            e.DP_y = tf.doubleGaussian(e.DP_x, *e.Te_popt)

            # Plot for visual inspection
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(figsize=(18, 10), ncols=4)
            ax0.plot(exp_seg_x, exp_seg_y, 'b.', label='data')
            ax0.plot(exp_seg_x, tf.doubleGaussian(exp_seg_x, *e.Te_popt), 'r-', label='fit')
            ax0.plot((e.DP_x, e.DP_x), (0, e.DP_y), 'gx-')
            ax0.plot((0, exp_seg_x[-1]), (e.DP_y, e.DP_y), 'r-x', label='DP')
            ax0.legend()
            ax0.set_title('DP calculation')

            ax1.plot(exp_seg_x, fdy)
            ax1.plot((e.DP_x, e.DP_x), (0, tf.doubleGfdy(e.DP_x, e.Te_popt)), 'gx-')
            ax1.set_title('Velocity')

            ax2.plot(exp_seg_x, f2dy)
            ax2.plot((e.DP_x, e.DP_x), (0, tf.doubleGf2dy(e.DP_x, e.Te_popt)), 'gx-')
            ax2.set_title('~ Force')

            ax3.plot(exp_seg_x, f3dy)
            ax3.plot((e.DP_x, e.DP_x), (0, tf.doubleGf3dy(e.DP_x, e.Te_popt)), 'gx-')
            ax3.set_title('Change in force')
            plt.savefig(exp_path + f"{pnt}_{e.peak}_DP.png", bbox_inches='tight')
            plt.close(fig)

            combined = np.vstack((exp_seg_x, exp_seg_y)).T
            fast_expiraiton = combined[combined[:, 0] < e.DP_x]
            slow_expiration = combined[combined[:, 0] >= e.DP_x]
            slow_expiration[:, 0] -= e.DP_x

            try:
                popt, pcov = curve_fit(tf.linearFit, fast_expiraiton[:, 0], fast_expiraiton[:, 1],
                                       p0=[m_guess, b_guess])
                fe_y_fit = tf.linearFit(fast_expiraiton[:, 0], *popt)
                e.Te_FE_fit = r2_score(fast_expiraiton[:, 1], fe_y_fit)
                e.Te_FE_slope = popt[0]
                e.Te_FE = fast_expiraiton[-1, 0] - fast_expiraiton[0, 0]
                e.Te_FE_AuC = np.trapz(y=np.abs(fast_expiraiton[:, 1]), x=fast_expiraiton[:, 0])
                m_guess = popt[0]
                b_guess = popt[1]

                e.Te_SE_AuC = np.trapz(y=np.abs(slow_expiration[:, 1]), x=slow_expiration[:, 0])
                e.Te_SE = slow_expiration[-1, 0] - slow_expiration[0, 0]

                popt1, pcov1 = curve_fit(tf.expDecay, xdata=slow_expiration[:, 0], ydata=slow_expiration[:, 1],
                                         p0=[a_guess, t_guess, c_guess])
                a_guess = popt1[0]
                t_guess = popt1[1]
                c_guess = popt1[2]
                se_y_fit = tf.expDecay(slow_expiration[:, 0], *popt1)
                e.Te_SE_fit = r2_score(slow_expiration[:, 1], se_y_fit)
                e.Te_SE_ppt = popt1
                e.Te_SE_half = -np.math.log(0.5) / popt1[1]
            except Exception as e:
                print('SE fit failed', f"Error - {str(e)}")
    return events, x




def cleanup_dataframe(dict):
    # Replace None with nans
    for key, values in dict.items():
        if values is None:
            dict[key] = np.nan
        else:
            # Check if the value is an iterable type
            if isinstance(values, (list, tuple, np.ndarray)):
                dict[key] = [np.nan if v is None else v for v in values]
            else:
                # If it's a singular value, assign it directly
                dict[key] = values
    # Whereever there's a single value, tripple it (to make dataframe compatible)
    for key, value in dict.items():
        # Check if the value is not an iterable type (excluding strings, as they're iterable)
        if not isinstance(value, (list, tuple, np.ndarray)):
            # Replicate the value 3 times and reassign
            dict[key] = [value] * 3
    df = pd.DataFrame.from_dict(dict, orient='index')
    df = df.transpose()
    return df
