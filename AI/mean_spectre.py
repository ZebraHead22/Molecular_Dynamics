import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Configuration
INPUT_DIR = os.getcwd()
OUTPUT_DIR = os.getcwd()
DPI = 300

def detect_peaks(xf_filtered, smoothed_spectrum, original_spectrum):
    """Detect peaks using sliding window and iterative filtering."""
    max_amp = np.max(original_spectrum)
    # Step 1: Find all local maxima with 3-point window
    peaks = []
    for i in range(1, len(smoothed_spectrum) - 1):
        if smoothed_spectrum[i] > smoothed_spectrum[i - 1] and smoothed_spectrum[i] > smoothed_spectrum[i + 1]:
            peaks.append((xf_filtered[i], smoothed_spectrum[i]))
    if not peaks:
        return []
    # Step 2: Iteratively filter until less than 20 peaks remain
    current_peaks = peaks.copy()
    while len(current_peaks) >= 1000:
        new_peaks = []
        n = len(current_peaks)
        if n == 0:
            break
        for i in range(n):
            if n == 1:
                new_peaks.append(current_peaks[i])
                break
            if i == 0:
                if current_peaks[i][1] > current_peaks[i + 1][1]:
                    new_peaks.append(current_peaks[i])
            elif i == n - 1:
                if current_peaks[i][1] > current_peaks[i - 1][1]:
                    new_peaks.append(current_peaks[i])
            else:
                if current_peaks[i][1] > current_peaks[i - 1][1] and current_peaks[i][1] > current_peaks[i + 1][1]:
                    new_peaks.append(current_peaks[i])
        if len(new_peaks) == len(current_peaks):
            break
        current_peaks = new_peaks
    # Filter peaks below 1.5% of max amplitude from original spectrum
    current_peaks = [peak for peak in current_peaks if peak[1] >= 0.015 * max_amp]
    return current_peaks

def main():
    # Find all spectrum CSV files
    csv_files = glob.glob(os.path.join(INPUT_DIR, '*_spectrum.csv'))
    if not csv_files:
        print("Error: No spectrum CSV files found in input directory.")
        return

    print(f"Found {len(csv_files)} spectrum files. Processing...")

    # Load and validate all spectra
    spectra_data = []
    reference_freq = None
    for file in csv_files:
        df = pd.read_csv(file)
        if 'Frequency_cm-1' not in df or 'Amplitude' not in df:
            print(f"Error: Invalid columns in file {file}")
            return
        
        freq = df['Frequency_cm-1'].values
        amp = df['Amplitude'].values
        
        # Check frequency axis consistency
        if reference_freq is None:
            reference_freq = freq
        else:
            if not np.allclose(freq, reference_freq, atol=0.01):
                print("Error: Frequency axes mismatch between files")
                return
        
        spectra_data.append(amp)

    # Calculate mean spectrum
    mean_spectrum = np.mean(spectra_data, axis=0)

    # Detect peaks using original spectrum for both smoothed and original arguments
    detected_peaks = detect_peaks(reference_freq, mean_spectrum, mean_spectrum)

    # Save mean spectrum
    output_csv = os.path.join(OUTPUT_DIR, 'mean_spectre_AI.csv')
    pd.DataFrame({
        'Frequency_cm-1': reference_freq,
        'Amplitude': mean_spectrum
    }).to_csv(output_csv, index=False)
    print(f"Saved mean spectrum to {output_csv}")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(reference_freq, mean_spectrum, 'k-', lw=0.8, label='Mean Spectrum')
    
    # Plot peaks with unique labels
    seen_labels = set()
    for freq, amp in detected_peaks:
        label = f"{freq:.2f} cm⁻¹"
        if label not in seen_labels:
            seen_labels.add(label)
            plt.scatter(freq, amp, color='red', marker='x', s=100, label=label)
        else:
            plt.scatter(freq, amp, color='red', marker='x', s=100)
    
    plt.xlabel("Frequency (cm⁻¹)")
    plt.ylabel("Spectral ACF EDM Amplitude (a. u.)")
    plt.title("Mean Spectrum with Detected Peaks")
    plt.grid(True, alpha=0.3)
    plt.legend(title='Peak Positions', loc='upper right', ncol=2, fontsize=8)
    plt.tight_layout()
    
    output_img = os.path.join(OUTPUT_DIR, 'mean_spectrum_with_peaks.png')
    plt.savefig(output_img, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_img}")

if __name__ == '__main__':
    main()