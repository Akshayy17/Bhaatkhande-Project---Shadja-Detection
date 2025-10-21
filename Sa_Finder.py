import os
import csv
import numpy as np
import librosa
from tqdm import tqdm
from collections import Counter
import math

# ========== CONFIGURATION ==========

INPUT_FOLDER = "./downloaded_songs"
OUTPUT_CSV = "Result.csv"
MAX_SONGS = 100000000000
TOP_N = 12
SUPPORTED_EXTENSIONS = [".mp3", ".wav"]
SR = 44100
DIVISIONS = 3 #Spect. gonna have (div*duration*dividingFactorOfHopLen(here 5) + 1) time axis shape
N_MELS = 1700 #for 1900 result is 13.53% error,1700 = 12.31%
GLOBAL_THRESH = 10
COLUMN_THRESH = 10
RATIO_LIST = [1, 2, 3, 4]
BACKGROUND_VALUE = -80
TOP_FREQ_PERCENTAGE = 15



BIN_EDGES = np.array([110, 116.39891589, 123.29601869, 130.6662243,
 138.49894393, 146.80959813, 155.58328037, 164.84548598,
 174.57431776, 184.8136729 , 195.8082243 , 207.45629907,
 220])



bins_13 = [
    (0.49, 0.515), (0.52, 0.545), (0.55, 0.58),
    (0.585, 0.6025), (0.6075, 0.6425), (0.6475, 0.6825),
    (0.6875, 0.725), (0.7325, 0.77), (0.7775, 0.8125),
    (0.8175, 0.865), (0.875, 0.915), (0.92, 0.965),
    (0.98, 1.03), (1.04, 1.09), (1.10, 1.16),
    (1.17, 1.205), (1.215, 1.285), (1.295, 1.365),
    (1.375, 1.45), (1.465, 1.54), (1.555, 1.625),
    (1.635, 1.73), (1.75, 1.83), (1.84, 1.93),
    (1.96, 2.05)
]


bins_8 = [
    (0.9800, 1.0333), (1.0333, 1.1625), (1.1625, 1.2917), (1.2917, 1.453),
    (1.4530, 1.5500), (1.5500, 1.7340), (1.7340, 1.9400), (1.9400, 2.050)
]

BIN_LABELS = ['2A', '2A#', '2B', '3C', '3C#', '3D', '3D#', '3E', '3F', '3F#', '3G', '3G#']

# ========== FUNCTION: filename parser ==========
def parse_filename(filename):
    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    serial = parts[0]
    thaat = parts[1]

    raga_terms = []
    i = 2
    while i < len(parts) and '(' not in parts[i]:
        raga_terms.append(parts[i])
        i += 1
    raga = "_".join(raga_terms)

    taal = parts[i] if i < len(parts) else ""
    i += 1

    song_title = "_".join(parts[i:]) if i < len(parts) else ""

    return [serial, thaat, raga, taal, song_title]

# ========== AUDIO PROCESSING FUNCTIONS ==========

def compute_features(y, sr, n_fft, hop_length, n_mels, global_thresh, per_column_thresh):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                               n_mels=n_mels, window='hann')
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    global_thresh_val = np.percentile(mel_spec_db, 100 - global_thresh)
    mel_spec_db[mel_spec_db < global_thresh_val] = BACKGROUND_VALUE

    for i in range(mel_spec_db.shape[1]):
        col_thresh = np.percentile(mel_spec_db[:, i], 100 - per_column_thresh)
        mel_spec_db[mel_spec_db[:, i] < col_thresh, i] = BACKGROUND_VALUE

    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr / 2)
    cutoff_bin = np.argmax(mel_freqs > 2500)
    mel_spec_db[cutoff_bin:, :] = BACKGROUND_VALUE

    freq_bins = np.fft.rfftfreq(n_fft, d=1 / sr)
    low_cut_bin = np.argmin(freq_bins < 150) #freq. converts to 100 hz for this value(150) for 1900 n_mels
    mel_spec_db[:low_cut_bin, :] = BACKGROUND_VALUE

    return mel_spec_db

def modify_mel_spec_top_to_bottom(mel_spec_db):
    modified_spec = mel_spec_db.copy()
    rows, cols = mel_spec_db.shape
    for i in range(cols):
        for j in range(1, rows):
            if modified_spec[j - 1, i] < modified_spec[j, i]:
                modified_spec[j - 1, i] = BACKGROUND_VALUE
    return modified_spec

def modify_mel_spec_bottom_to_top(mel_spec_db):
    modified_spec = mel_spec_db.copy()
    rows, cols = mel_spec_db.shape
    for i in range(cols):
        for j in range(rows - 2, -1, -1):
            if modified_spec[j + 1, i] < modified_spec[j, i]:
                modified_spec[j + 1, i] = BACKGROUND_VALUE
    return modified_spec

def custom_and_operation(spec1, spec2):
    background_mask = (spec1 == BACKGROUND_VALUE) & (spec2 == BACKGROUND_VALUE)
    averaged_spec = (spec1 + spec2)
    result_spec = np.where(background_mask, BACKGROUND_VALUE, averaged_spec)
    return result_spec

def find_best_frequencies(spectrogram, sr, ratio_list):
    num_freq_bins, num_time_frames = spectrogram.shape
    freqs = librosa.mel_frequencies(n_mels=num_freq_bins, fmin=0, fmax=sr / 2)
    overtone_matrix = np.full((len(ratio_list), num_time_frames, 2), [0, BACKGROUND_VALUE], dtype=np.float32)

    for col in range(num_time_frames):
        valid_indices = np.where(spectrogram[:, col] > BACKGROUND_VALUE)[0]
        if len(valid_indices) < len(ratio_list):
            continue

        freq_values = freqs[valid_indices]
        intensity_values = spectrogram[valid_indices, col]

        best_sequence, best_indices, best_intensity_sum = None, None, -np.inf

        for i in range(len(freq_values)):
            x1 = freq_values[i]
            expected_freqs = np.array([x1 * r for r in ratio_list])
            tolerance = expected_freqs * 0.02

            matches, intensities, match_indices = [], [], []
            for ef, tol in zip(expected_freqs, tolerance):
                mask = np.abs(freq_values - ef) <= tol
                if not np.any(mask):
                    break
                best_idx = np.argmax(intensity_values[mask])
                matches.append(float(freq_values[mask][best_idx]))
                intensities.append(
                    float(intensity_values[mask][best_idx]) * float(freq_values[mask][best_idx])**0.1
                )
                match_indices.append(np.where(mask)[0][best_idx])

            if len(matches) == len(ratio_list):
                total_intensity = sum(intensities)
                if total_intensity > best_intensity_sum:
                    best_sequence, best_indices, best_intensity_sum = matches, match_indices, total_intensity

        if best_sequence:
            for i, f in enumerate(best_sequence):
                original_idx = valid_indices[best_indices[i]]
                overtone_matrix[i, col, 0] = freqs[original_idx]
                overtone_matrix[i, col, 1] = spectrogram[original_idx, col]

    return overtone_matrix

def fold_frequency(freq):
    if freq == 0:
        return 0
    while freq < 110:
        freq *= 2
    while freq > 220:
        freq /= 2
    return freq

def hz_to_note(freq):
    midi = librosa.hz_to_midi(freq)
    return librosa.midi_to_note(midi, octave=False)

def prepare_freq_distribution(overtone_matrix_0, TOP_FREQ_PERCENTAGE=10):
    """Returns filtered [(freq, count)] list, selecting top x% of frequencies."""
    freqs = overtone_matrix_0[:, 0]
    # Remove zeros (i.e., background or padding)
    freqs = freqs[freqs > 0]
    #print(freqs)
    # Round to 2 decimals for grouping similar freqs
    rounded_freqs = [round(f, 2) for f in freqs]
    freq_counts = Counter(rounded_freqs)

    # Sort by count descending
    sorted_freqs = sorted(freq_counts.items(), key=lambda x: x[1], reverse=True)

    # Pick top X% entries
    cutoff = max(1, int(len(sorted_freqs) * (TOP_FREQ_PERCENTAGE / 100.0)))
    top_freqs = sorted_freqs[:cutoff]

    return top_freqs  # [(freq, count), ...]

def convert_counts_to_pairs(top_freqs):
    return [(math.floor(freq) if freq - int(freq) < 0.5 else math.ceil(freq), count) for freq, count in top_freqs]
    #return [(int(freq), count) for freq, count in top_freqs]


def evaluate_all(pairs, bins, mode="normal"):
    results = []

    for sa_freq, _ in pairs:
        bins_dict = {}
        lower_sa = None

        for freq, count in pairs:
            #if freq == sa_freq:
                #continue
            ratio = round(freq / sa_freq, 3)
            for b_idx, (low, high) in enumerate(bins_13, start=1):
                if low <= freq / sa_freq <= high:
                    if b_idx == 1:  # lower Sa
                        lower_sa = freq
                    bins_dict.setdefault(b_idx, []).append((freq, count, ratio))
                    break

        # calculate total score and per-bin scores
        bin_scores = {b: sum(c for _, c, _ in vals) for b, vals in bins_dict.items()}
        total_score = sum(bin_scores.values())

        # --- Calculate Main Score ---
        lower_bins = sum(score for b, score in bin_scores.items() if b <= 13)
        upper_bins = sum(score for b, score in bin_scores.items() if b >= 13)
        main_score = max(lower_bins, upper_bins)

        sa_output = lower_sa if lower_sa else sa_freq
        results.append((sa_output, sa_freq, total_score, main_score, bins_dict, bin_scores))

    # sort by total score and pick top 3
    top3 = sorted(results, key=lambda x: x[3], reverse=True)[:3]

    return top3


def compute_top_bins(overtone_matrix, top_n):
    folded_freqs = np.array([fold_frequency(f) for f in overtone_matrix[0][:, 0]])
    bin_indices = np.digitize(folded_freqs, BIN_EDGES)
    bin_indices[folded_freqs == 0] = -1

    hist = np.zeros(12, dtype=int)
    for idx in bin_indices:
        if 1 <= idx <= 12:
            hist[idx - 1] += 1

    total = hist.sum()
    if total == 0:
        return []

    top_indices = np.argsort(hist)[::-1][:top_n]
    top_bins_with_percent = [
        f"{BIN_LABELS[i]} ({round((hist[i] / total) * 100, 2)}%)"
        for i in top_indices
    ]
    return top_bins_with_percent 

def process_all_songs(TOP_FREQ_PERCENTAGE=10):
    files = sorted([f for f in os.listdir(INPUT_FOLDER) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS])
    files = files[:min(MAX_SONGS, len(files))]

    if not files:
        print("No audio files found.")
        return

    with open(OUTPUT_CSV, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for filename in tqdm(files, desc="Processing files", unit="file"):
            file_path = os.path.join(INPUT_FOLDER, filename)
            try:
                y, sr = librosa.load(file_path, sr=SR)
                duration = int(len(y) / sr) + 1
                y = np.pad(y, (0, sr * duration - len(y)), mode='constant')

                n_fft = int(sr / DIVISIONS)
                hop_length = int(n_fft / 5)

                mel_spec_db = compute_features(y, sr, n_fft, hop_length, N_MELS, GLOBAL_THRESH, COLUMN_THRESH)
                top_mod = modify_mel_spec_top_to_bottom(mel_spec_db)
                bot_mod = modify_mel_spec_bottom_to_top(mel_spec_db)
                mel_and = custom_and_operation(top_mod, bot_mod)
                overtone_matrix = find_best_frequencies(mel_and, sr, RATIO_LIST)
                #print(overtone_matrix)
                top_bins_with_percent = compute_top_bins(overtone_matrix, TOP_N)

                parsed = parse_filename(filename)

                top_freqs = prepare_freq_distribution(overtone_matrix[0], TOP_FREQ_PERCENTAGE=TOP_FREQ_PERCENTAGE)
                potential_values = convert_counts_to_pairs(top_freqs)
                #print(potential_values)

                #all_results_13 = evaluate_all(potential_values, bins_13, mode="normal")
                #all_results_8 = evaluate_all(potential_values, bins_8 , mode="normal")
                all_results_13v8 = evaluate_all(potential_values, bins_13, mode="13v8")[:3]
                #print(all_results_13v8)

                def get_top_freqs(results):
                    #print(results[0])
                    #print(results[1][0])
                    #print(results[2][0])
                    return [f"{res[1]} ({hz_to_note(res[1])})" for res in results[:3]]
                

                #row = parsed + get_top_freqs(all_results_13) + get_top_freqs(all_results_8) + get_top_freqs(all_results_13v8)
                row = parsed + get_top_freqs(all_results_13v8)
                writer.writerow(row)
                csvfile.flush()

            except Exception as e:
                print(f"✗ Failed {filename}: {e}")

# ========== RUN ==========
if __name__ == "__main__":
    process_all_songs(TOP_FREQ_PERCENTAGE)

