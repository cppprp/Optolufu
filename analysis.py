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
                br_rate, phase, heart_rate, filtered_signal = get_parameters_in_frequency(self.points[pnt, 2, :],
                                                                                          pnt, exp_path,
                                                                                          Verbose=True)
                wave.append(phase)
                br_rate_m.append(br_rate)
                heart_rate_m.append(heart_rate)

                # Get other parameters in temporal domain
                exp_x, exp_y = get_parameters_in_temporal(filtered_signal, pnt, exp_path, br=br_rate, calculations=self.calculation,
                                                          Verbose=True)
                if type(exp_x).__name__ == 'ndarray' and type(exp_y).__name__ == 'ndarray':
                    DP = breaking_pnt(exp_x, exp_y, pnt, exp_path, self.calculation, Verbose=True)
                else:
                    DP = None
                if DP is not None:
                    proceed_DP(exp_x, exp_y, DP[0], self.calculation)
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


def get_parameters_in_temporal(y, pnt, exp_path, calculations, br, Verbose=False):
    Ti_m = []
    Ti_AuC_m = []
    Te_m = []
    dInsp_m = []
    dInsp_fit_m = []
    exp_buffer_y = []
    exp_buffer_x = []
    peak_threshold = 0.075
    event_num = 10
    m_guess = 1.6
    b_guess = 0.2

    x = np.arange(0, len(y) / 100.0, 0.01)  # by FPS
    #dist = 1.0/br*100
    peaks, _ = find_peaks(y, height=peak_threshold, distance=60)
    interv = tf.find_intervals(y, peaks, peak_threshold)
    if len(peaks) > event_num and len(interv) > event_num:
        if Verbose:
            fig = tf.plt.figure(figsize=(20, 4))
            tf.plt.plot(np.arange(len(y)), y, '-')
            tf.plt.plot(peaks, y[peaks], 'ro', markersize=12, label='detected peaks')
            tf.plt.plot(interv, y[interv], 'rx', markersize=12, label='detected peaks')
            tf.plt.legend()
            tf.plt.xlabel('Time[frames]')
            tf.plt.ylabel('Displacement[mm]')
            plt.savefig(exp_path + f"{pnt}_peaks.png", bbox_inches='tight')
            plt.close(fig)
            # tf.plt.show()

        for idx in range(1, event_num + 1):
            if interv[idx] < peaks[idx]:
                Ti = x[peaks[idx]] - x[interv[idx]]
                Te = x[interv[idx + 1]] - x[peaks[idx]]
                dInsp = (y(peaks[idx]) - y(interv[idx])) / (x[peaks[idx]] - x[interv[idx]])
            if peaks[idx] < interv[idx]:
                print(peaks[idx], interv[idx])
                # Get inspiration time
                Ti = x[peaks[idx + 1]] - x[interv[idx]]
                # Get expiration time
                Te = x[interv[idx]] - x[peaks[idx]]
                # Get slope of inspiration and coefficient of determination
                # dInsp = (y[peaks[idx + 1]] - y[interv[idx]]) / (x[peaks[idx + 1]] - x[interv[idx]])
                x_feed = x[interv[idx]:peaks[idx + 1]]
                y_feed = y[interv[idx]:peaks[idx + 1]]
                popt, pcov = curve_fit(tf.linearFit, x_feed, y_feed, p0=[m_guess, b_guess])
                m_guess = popt[0]
                b_guess = popt[1]
                y_fit = tf.linearFit(x_feed, *popt)
                dInsp_fit = r2_score(y_feed, y_fit)
                dInsp = popt[0]
                Ti_AuC = np.trapz(y=np.abs(y_feed), x=np.linspace(0, ((len(y_feed)) / 100.0), num=len(y_feed)))
                # Make expiration segments for downstream analysis
                exp_seg_y = y[peaks[idx]:interv[idx]]
                exp_seg_x = np.arange(len(exp_seg_y)) / 100.0
            exp_buffer_y.append(exp_seg_y)
            exp_buffer_x.append(exp_seg_x)
            Ti_m.append(Ti)
            Ti_AuC_m.append(Ti_AuC)
            Te_m.append(Te)
            dInsp_m.append(dInsp)
            dInsp_fit_m.append(dInsp_fit)

        # Perform the breaking point calculation
        exp_buffer_y = np.concatenate(exp_buffer_y).ravel()
        exp_buffer_x = np.concatenate(exp_buffer_x).ravel()
        '''tf.plt.figure(figsize=(10, 4))
        tf.plt.plot(exp_buffer_x, exp_buffer_y, '.')
        tf.plt.legend()
        tf.plt.xlabel('Time[frames]')
        tf.plt.ylabel('Displacement[mm]')'''

        calculations['Time [ms]'] = x  # Time in ms
        calculations['Displacement [mm]'] = y  # Z displacement
        calculations['SegExp_time [ms]'] = exp_seg_x  # Time for expiration segments
        calculations['SegExp_disp [mm]'] = exp_seg_y  # Z displacement of all expiration segments
        calculations['Peaks [ind]'] = peaks  # Peaks inspeiration points
        calculations['Intervals [ind]'] = interv  # End of breathing event points
        calculations['T_i [ms]'] = Ti_m  # Total inspiration time
        calculations['T_e [ms]'] = Te_m  # Total expiration time
        calculations['Ti_slope'] = dInsp_m  # Inspiration slope
        calculations['Ti_fit'] = dInsp_fit_m  # R^2 linear fit inspiration
        calculations['Ti_AuC [mm*ms]'] = Ti_AuC_m  # Inspiration area under curve

    else:
        calculations['Time [ms]'] = x  # Time in ms
        calculations['Displacement [mm]'] = y  # Z displacement
        calculations['Peaks [ind]'] = peaks  # Peaks inspeiration points
        calculations['Intervals [ind]'] = interv  # End of breathing event points
        print('Did not detect 10 peaks: ' + str(pnt))
    return exp_buffer_x, exp_buffer_y


def breaking_pnt(x, y, pnt, exp_path, calculations, Verbose=True):
    # Fit a double gaussian to the curve
    DP = None
    SegExp_fit = None
    p0_guess = 0.0
    p1_guess = 10.0
    p2_guess = 0.1
    p3_guess = 8.6
    p4_guess = 0.7
    try:
        popt, pcov = curve_fit(tf.doubleGaussian, x, y, p0=[p0_guess, p1_guess, p2_guess, p3_guess, p4_guess])
        fit_y = tf.doubleGaussian(x, *popt)
        # Get r^2 val
        SegExp_fit = r2_score(y, fit_y)
        # Get 1°, 2°, 3° derivatives
        fit_x = np.linspace(np.min(x), np.max(x), len(y))
        fdy = tf.doubleGfdy(fit_x, popt)
        f2dy = tf.doubleGf2dy(fit_x, popt)
        f3dy = tf.doubleGf3dy(fit_x, popt)
        # f3dy = np.nan_to_num(f3dy, nan=float('inf'))
        # Define the 'dampening point'
        xDP = fit_x[np.nanargmin(f3dy)]
        xDP = fit_x[np.nanargmax(f2dy)]
        yDP = tf.doubleGaussian(xDP, *popt)
        DP = (xDP, yDP)

        # Plot for visual inspection
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(figsize=(18, 10), ncols=4)
        ax0.plot(x, y, 'b.', label='data')
        ax0.plot(fit_x, tf.doubleGaussian(fit_x, *popt), 'r-', label='fit')
        ax0.plot((xDP, xDP), (0, yDP), 'gx-')
        ax0.plot((0, x[-1]), (yDP, yDP), 'r-x', label='DP')
        ax0.legend()
        ax0.set_title('DP calculation')

        ax1.plot(fit_x, fdy)
        ax1.plot((xDP, xDP), (0, tf.doubleGfdy(xDP, popt)), 'gx-')
        ax1.set_title('Velocity')

        ax2.plot(fit_x, f2dy)
        ax2.plot((xDP, xDP), (0, tf.doubleGf2dy(xDP, popt)), 'gx-')
        ax2.set_title('~ Force')

        ax3.plot(fit_x, f3dy)
        ax3.plot((xDP, xDP), (0, tf.doubleGf3dy(xDP, popt)), 'gx-')
        ax3.set_title('Change in force')

        # button_width, button_height = 0.2, 0.1
        # button_spacing = 0.025
        # axprev = plt.axes([0.4 - button_spacing / 2 - button_width, 0.01, button_width, button_height])
        # axnext = plt.axes([0.6 + button_spacing / 2, 0.01, button_width, button_height])
        plt.savefig(exp_path + f"{pnt}_DP.png", bbox_inches='tight')
        # Decide whether to proceed with further calcualtions
        # Inline definition of button handling
        # bnext = Button(axnext, 'Proceed')
        # bnext.on_clicked(lambda event: globals().update(proceed_with_DP=True))
        # bprev = Button(axprev, 'Skip')
        # bprev.on_clicked(lambda event: globals().update(proceed_with_DP=False))
        fig.show()
        plt.close(fig)
    except Exception as e:
        print(f"Error - {str(e)}")

    calculations['Dampening [ind]'] = DP  # Dampening point
    calculations['2G fit R^2'] = SegExp_fit  # Dampening point
    return DP


'''
    if fit is 'Decay':
        a_guess = np.max(y) - np.min(y)
        b_guess = 1e-4  # an example value, might need tweaking
        c_guess = np.min(y)
        popt, pcov = curve_fit(tf.expDecay, x, y, p0=[a_guess, b_guess, c_guess])
        fit_y = tf.expDecay(x, *popt)
    if fit is 'dSink':
        a_guess = np.pi
        b_guess = np.pi / 2.0 / y[np.argmin(x)]
        c_guess = 0.0
        popt, pcov = curve_fit(tf.squaredSinc, x, y, p0=[a_guess, b_guess, c_guess])
        fit_y = tf.squaredSinc(x, *popt)
    if Verbose:
        fit_x = np.linspace(np.min(x), np.max(x), 100)
        plt.plot(x, y, 'rx', label='values')
        plt.plot(fit_x, fit_y, 'b-', label='regression')
        plt.xlabel('Time[frames]')
        plt.ylabel('Displacement[mm]')
        plt.show()'''


def proceed_DP(x, y, xDP, calculations):
    T_fe = None
    dExp_fe = None
    dExp_fe_fit = None
    Fe_AuC = None

    T_se = None
    dExp_se = None
    dExp_se_fit = None
    Se_AuC = None

    combined = np.vstack((x, y)).T
    sorted_combined = combined[combined[:, 0].argsort()]
    fast_expiraiton = sorted_combined[sorted_combined[:, 0] < xDP]
    slow_expiration = sorted_combined[sorted_combined[:, 0] >= xDP]
    slow_expiration[:, 0] -= xDP

    m_guess = -1.0
    b_guess = 0.3
    try:
        popt, pcov = curve_fit(tf.linearFit, fast_expiraiton[:, 0], fast_expiraiton[:, 1], p0=[m_guess, b_guess])
        fe_y_fit = tf.linearFit(fast_expiraiton[:, 0], *popt)
        dExp_fe_fit = r2_score(fast_expiraiton[:, 1], fe_y_fit)
        dExp_fe = popt[0]
        T_fe = fast_expiraiton[-1, 0] - fast_expiraiton[0, 0]
        Fe_AuC = np.trapz(y=np.abs(fast_expiraiton[:, 1]), x=fast_expiraiton[:, 0])
        a_guess = 10.0
        b_guess = -0.3
        c_guess = 0.0
        Se_AuC = np.trapz(y=np.abs(slow_expiration[:, 1]), x=slow_expiration[:, 0])
        T_se = slow_expiration[-1, 0] - slow_expiration[0, 0]

        popt1, pcov1 = curve_fit(tf.one_over_x, slow_expiration[:, 0], slow_expiration[:, 1], p0=[a_guess, b_guess])
        se_y_fit = tf.one_over_x(slow_expiration[:, 0], *popt)
        dExp_se_fit = r2_score(slow_expiration[:, 1], se_y_fit)
        dExp_se = popt1[0]
    except Exception as e:
        print(f"Error - {str(e)}")

    calculations['T_fe [ms]'] = T_fe  # Fast expiration time
    calculations['Fe_slope'] = dExp_fe  # Fast expiration slope
    calculations['Fe_AuC [mm*ms]'] = Fe_AuC  # Fast expiration area under curve
    calculations['Fe_fit'] = dExp_fe_fit  # R^2 linear fit fast expiration

    calculations['T_se [ms]'] = T_se  # Slow expiration time
    calculations['Se_decay'] = dExp_se  # Slow inspiration decay
    calculations['Se_AuC [mm*ms]'] = Se_AuC  # Slow expiration area under curve
    calculations['Se_fit'] = dExp_se_fit  # R^2 exponential decay fit slow expiration


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
