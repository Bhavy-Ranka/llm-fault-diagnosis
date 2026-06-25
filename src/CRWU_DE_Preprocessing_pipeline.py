import os
import json
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.stats import skew, kurtosis
from pathlib import Path

CWRU_FILES = {
    # Normal baseline 
    ("NO",  "none", 0): "archive/cwru/Normal_0.mat",
    ("NO",  "none", 1): "archive/cwru/Normal_1.mat",
    ("NO",  "none", 2): "archive/cwru/Normal_2.mat",
    ("NO",  "none", 3): "archive/cwru/Normal_3.mat",
 
    # Inner Race Fault 
    ("IRF", "007",  0): "archive/cwru/IR007_0.mat",
    ("IRF", "007",  1): "archive/cwru/IR007_1.mat",
    ("IRF", "007",  2): "archive/cwru/IR007_2.mat",
    ("IRF", "007",  3): "archive/cwru/IR007_3.mat",
    ("IRF", "014",  0): "archive/cwru/IR014_0.mat",
    ("IRF", "014",  1): "archive/cwru/IR014_1.mat",
    ("IRF", "014",  2): "archive/cwru/IR014_2.mat",
    ("IRF", "014",  3): "archive/cwru/IR014_3.mat",
    ("IRF", "021",  0): "archive/cwru/IR021_0.mat",
    ("IRF", "021",  1): "archive/cwru/IR021_1.mat",
    ("IRF", "021",  2): "archive/cwru/IR021_2.mat",
    ("IRF", "021",  3): "archive/cwru/IR021_3.mat",
 
    # Outer Race Fault (@6 o'clock position)
    ("ORF", "007",  0): "archive/cwru/OR0076_0.mat",
    ("ORF", "007",  1): "archive/cwru/OR0076_1.mat",
    ("ORF", "007",  2): "archive/cwru/OR0076_2.mat",
    ("ORF", "007",  3): "archive/cwru/OR0076_3.mat",
    ("ORF", "014",  0): "archive/cwru/OR0146_0.mat",
    ("ORF", "014",  1): "archive/cwru/OR0146_1.mat",
    ("ORF", "014",  2): "archive/cwru/OR0146_2.mat",
    ("ORF", "014",  3): "archive/cwru/OR0146_3.mat",
    ("ORF", "021",  0): "archive/cwru/OR0216_0.mat",
    ("ORF", "021",  1): "archive/cwru/OR0216_1.mat",
    ("ORF", "021",  2): "archive/cwru/OR0216_2.mat",
    ("ORF", "021",  3): "archive/cwru/OR0216_3.mat",
 
    # Rolling Element / Ball Fault 
    ("REF", "007",  0): "archive/cwru/B007_0.mat",
    ("REF", "007",  1): "archive/cwru/B007_1.mat",
    ("REF", "007",  2): "archive/cwru/B007_2.mat",
    ("REF", "007",  3): "archive/cwru/B007_3.mat",
    ("REF", "014",  0): "archive/cwru/B014_0.mat",
    ("REF", "014",  1): "archive/cwru/B014_1.mat",
    ("REF", "014",  2): "archive/cwru/B014_2.mat",
    ("REF", "014",  3): "archive/cwru/B014_3.mat",
    ("REF", "021",  0): "archive/cwru/B021_0.mat",
    ("REF", "021",  1): "archive/cwru/B021_1.mat",
    ("REF", "021",  2): "archive/cwru/B021_2.mat",
    ("REF", "021",  3): "archive/cwru/B021_3.mat",
}

def load_and_segment(file_path: str, window_size: int = 2048,
                     overlap: float = 0.2, channel: str = "DE_time"):
    mat = scipy.io.loadmat(file_path)
    key = next((k for k in mat if channel in k), None)    
    signal = mat[key].flatten()
    
    # REMOVED GLOBAL NORMALIZATION: Retaining physical metrics (G's) for fault severity tracking
    step = int(window_size * (1 - overlap))
    
    segments = [signal[i: i + window_size]
                for i in range(0, len(signal) - window_size, step)]
    return np.array(segments)

def compute_fft_magnitude(segment: np.ndarray):
    L = len(segment)
    X = fft(segment)
    magnitude = np.abs(X)[:L // 2]      
    return magnitude / L  


# Statistical feature extraction    
TIME_FEATURE_NAMES = ["mean", "RMS", "standard_deviation", "crest_factor", "skewness", "shape_factor", "kurtosis", "peak_to_peak", "energy_factor", "impulse_factor"]
FREQ_FEATURE_NAMES = ["peak_frequency", "spectral_centroid", "spectral_kurtosis", "spectral_bandwidth", "spectral_skewness"]
 
def time_features(x: np.ndarray) -> list:
    mean       = np.mean(x)
    rms        = np.sqrt(np.mean(x ** 2))
    std        = np.std(x)
    peak       = np.max(np.abs(x))
    ptp        = np.ptp(x)
    sk         = skew(x)
    kurt_val   = kurtosis(x)
    crest_f    = peak / (rms + 1e-12)
    shape_f    = rms / (np.mean(np.abs(x)) + 1e-12)
    impulse_f  = peak / (np.mean(np.abs(x)) + 1e-12)
    energy_f   = np.sum(x ** 2) / (np.sum(np.abs(x)) ** 2 + 1e-12)
    return [mean, rms, std, crest_f, sk, shape_f, kurt_val, ptp, energy_f, impulse_f]
 
 
def frequency_features(mag: np.ndarray, freq: np.ndarray) -> list:
    peak_freq   = freq[np.argmax(mag)]
    
    # Corrected: Found actual spectral centroid instead of mean of the static frequency array bounds
    centroid    = np.sum(freq * mag) / (np.sum(mag) + 1e-12)
    
    # Corrected: Subtracted center-of-mass centroid instead of absolute frequency midpoint
    spec_bw     = np.sqrt(
        np.sum((freq - centroid) ** 2 * mag) / (np.sum(mag) + 1e-12)
    )
    
    spec_kurt   = kurtosis(mag)
    spec_skew   = skew(mag)
    
    return [peak_freq, centroid, spec_kurt, spec_bw, spec_skew]

    
def extract_all_features(segments: np.ndarray, fs: int = 12000):
    N = segments.shape[1]
    freq = np.fft.fftfreq(N, 1 / fs)[: N // 2]
    rows = []
    for seg in segments:
        mag = compute_fft_magnitude(seg)
        tf  = time_features(seg)
        ff  = frequency_features(mag, freq)
        rows.append(tf + ff)
    cols = TIME_FEATURE_NAMES + FREQ_FEATURE_NAMES
    return pd.DataFrame(rows, columns=cols)


# String based FFT tokenisation
def encode_fft(magnitude: np.ndarray, separator=",", D=3):
    scale = 10 ** D
    quantised = [int(v * scale) for v in magnitude]
    tokens = [str(q) if not np.isnan(q) else "NaN" for q in quantised]
    return separator.join(tokens)


FFT_INSTRUCTION_TEMPLATE = (
    "Given machine information: {equip_info}; and working conditions: "
    "{load} hp, {speed} rpm, please predict the operating status of the "
    "bearing based on the following FFT vector."
)
 
STAT_INSTRUCTION_TEMPLATE = (
    "Given machine information: {equip_info}; and working conditions: "
    "{load} hp, {speed} rpm, please predict the operating status of the "
    "bearing based on the following time-domain and frequency-domain features."
)

LOAD_TO_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
EQUIP_INFO = (
    "CWRU Bearing Dataset; "
    "Device: Reliance Electric 2 hp motor; "
    "Bearing model: SKF deep-groove ball bearing 6205-2RS JEM; "
    "Pitch diameter: 1.748 in; Ball diameter: 0.3126 in; "
    "Contact angle: 0°; Number of balls: 9"
)

FAULT_LABELS = ["NO", "IRF", "ORF", "REF"]

def build_fft_prompt(encoded_fft: str, label: str, load: int, equip_info: str = EQUIP_INFO):
    speed = LOAD_TO_RPM[load]
    return {
        "instruction": FFT_INSTRUCTION_TEMPLATE.format(
            equip_info=equip_info, load=load, speed=speed),
        "input": encoded_fft,
        "output": label,
    }
 
 
def build_stat_prompt(stat_text: str, label: str, load: int, equip_info: str = EQUIP_INFO):
    speed = LOAD_TO_RPM[load]
    return {
        "instruction": STAT_INSTRUCTION_TEMPLATE.format(
            equip_info=equip_info, load=load, speed=speed),
        "input": stat_text,
        "output": label,
    }

def generate_all_data(cwru_files=CWRU_FILES, window_size=2048, overlap=0.2, fft_D=3):
    fft_prompts = []
    stat_prompts = []
    
    fs = 12000
    freq = np.fft.fftfreq(window_size, 1 / fs)[: window_size // 2]
    

    for (fault_type, fault_size, load), file_path in cwru_files.items():
        if not os.path.exists(file_path):
            print(f"  [SKIP] File not found: {file_path}")
            continue
            
        label = fault_type
        
        segments = load_and_segment(file_path, window_size, overlap)
        
        for seg in segments:
            mag = compute_fft_magnitude(seg)
            
            encoded_fft = encode_fft(mag, D=fft_D) 
            fft_prompts.append(build_fft_prompt(encoded_fft, label, load))
            
            tf = time_features(seg)
            ff = frequency_features(mag, freq)
            stat_summary = ", ".join([f"{n}: {v:.4f}" for n, v in zip(TIME_FEATURE_NAMES + FREQ_FEATURE_NAMES, tf + ff)])
            stat_prompts.append(build_stat_prompt(stat_summary, label, load))
            
    return fft_prompts, stat_prompts

if __name__ == "__main__":
    fft_data, stat_data = generate_all_data()
    with open("cwru_fft_dataset.json", "w") as f:
        json.dump(fft_data, f, indent=2)
    
    with open("cwru_stat_dataset.json", "w") as f:
        json.dump(stat_data, f, indent=2)
        
    print(f"Done! Saved {len(fft_data)} FFT samples and {len(stat_data)} statistical samples.")
