import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.signal.windows import hann
from multiprocessing import Pool, cpu_count


def annotate_peaks(ax, peak_frequencies, peak_amplitudes):
    # Convert lists to NumPy arrays if not already
    peak_frequencies = np.array(peak_frequencies)
    peak_amplitudes = np.array(peak_amplitudes)

    # Dictionary with ranges and the number of peaks to search for
    ranges = {
        (0, 500): 1,
        (500, 1000): 1,
        (1000, 2000): 2,
        (2500, 3500): 3
    }

    # Find the maximum peak amplitude
    max_peak_amplitude = np.max(peak_amplitudes)
    two_max_peaks = np.argsort(peak_amplitudes)[-2:]  # Indices of the two highest peaks

    # Get the limits of the Y-axis for bounds checking
    ylim_lower, ylim_upper = ax.get_ylim()

    # Function for annotation with bounds checking and X-axis adjustment
    def add_annotation(ax, freq, amp, label, offset_y, offset_x=0, draw_arrow=False, shorten_arrow=False):
        # Calculate the annotation position
        text_y = amp + offset_y

        # Ensure annotation doesn't exceed Y-axis limits
        if text_y > ylim_upper:  # If the annotation is above the upper limit, move it inside and shift sideways
            text_y = ylim_upper - (ylim_upper * 0.05)
            offset_x = 40  # Shift annotation to the right to avoid covering the peak
        elif text_y < ylim_lower:  # If the annotation is below the lower limit, raise it
            text_y = ylim_lower + (ylim_upper * 0.05)
        
        # Special check for peaks with frequencies < 200
        if freq < 200:
            offset_x = 40  # Shift annotation to the right to avoid going off-chart
            text_y = max_peak_amplitude * 0.5  # Raise annotation, but check against the upper bound
            if text_y > ylim_upper:
                text_y = ylim_upper - (ylim_upper * 0.1)

        ax.text(freq + offset_x, text_y, label, fontsize=12, 
                rotation=90, ha='center')

        # Optionally add an arrow
        if draw_arrow:
            arrow_length_factor = 0.8 if shorten_arrow else 1.0  # Shorten the arrow by 20%
            ax.annotate('', xy=(freq, amp), xytext=(freq + offset_x, text_y * arrow_length_factor),
                        arrowprops=dict(facecolor='none', edgecolor='red', linestyle='dashed', 
                                        lw=0.5, shrink=0.05))

    annotations = []  # Store annotation data
    for (low, high), num_peaks in ranges.items():
        # Find peaks within the current range
        current_range = np.where((peak_frequencies > low) & (peak_frequencies <= high))[0]
        print(f"Range {low}-{high}: Found {len(current_range)} peaks, expected {num_peaks}")
        
        if len(current_range) == 0:
            print(f"No peaks found in the range {low}-{high}")
            continue

        # Find the top N peaks within the range
        top_peaks = np.argsort(peak_amplitudes[current_range])[-num_peaks:]
        selected_indices = current_range[top_peaks]

        # Add data for annotation
        for idx in selected_indices:
            freq = peak_frequencies[idx]
            amp = peak_amplitudes[idx]
            annotations.append((freq, amp))

    # Sort annotations by frequency for consistent display
    annotations = sorted(annotations, key=lambda x: x[0])

    # Process annotations
    for i, (freq, amp) in enumerate(annotations):
        offset_y = max_peak_amplitude * 0.06  # Offset for annotation above the peak
        offset_x = 0
        draw_arrow = False  # Flag for drawing arrow
        shorten_arrow = False  # Flag for shortening arrow

        # Special case for frequencies < 200
        if freq < 200:
            offset_y = max_peak_amplitude * 0.5  # Move to the 0.5 Y-axis level
            add_annotation(ax, freq, amp, f'{freq:.2f}', offset_y, offset_x=40)
            continue

        # Special case for frequencies < 1000
        if freq < 1000:
            offset_y = max_peak_amplitude * 0.06  # Smaller offset to stay within graph bounds
            add_annotation(ax, freq, amp, f'{freq:.2f}', offset_y, offset_x=-40)
            continue

        # Check if peaks are close to each other (< 350 cm⁻¹ apart)
        if i > 0 and (freq - annotations[i - 1][0]) < 350:
            # Raise annotation by amp_max * 0.26
            offset_y = max_peak_amplitude * 0.26
            draw_arrow = True  # Add arrow
            shorten_arrow = True  # Shorten arrow

            # Shift annotation left or right depending on frequency position
            if freq < annotations[i - 1][0]:
                offset_x = -40
            else:
                offset_x = 40

        # For the two highest peaks, keep the annotation at the peak's amplitude
        if i in two_max_peaks:
            offset_y = 0  # Set annotation level to the peak's amplitude
            draw_arrow = False  # No arrow for these peaks

        # General case for annotations, with the option of shortened arrows
        add_annotation(ax, freq, amp, f'{freq:.2f}', offset_y, offset_x, draw_arrow=draw_arrow, shorten_arrow=shorten_arrow)

    return ax
