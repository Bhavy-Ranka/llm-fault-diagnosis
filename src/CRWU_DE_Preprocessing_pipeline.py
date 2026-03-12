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
    # ── Normal baseline ───────────────────────────────────────────
    ("NO",  "none", 0): "archive/cwru/Normal_0.mat",
    ("NO",  "none", 1): "archive/cwru/Normal_1.mat",
    ("NO",  "none", 2): "archive/cwru/Normal_2.mat",
    ("NO",  "none", 3): "archive/cwru/Normal_3.mat",
 
    # ── Inner Race Fault ──────────────────────────────────────────
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
 
    # ── Outer Race Fault (@6 o'clock position) ────────────────────
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
 
    # ── Rolling Element / Ball Fault ──────────────────────────────
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
LOAD_TO_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
EQUIP_INFO = (
    "CWRU Bearing Dataset; "
    "Device: Reliance Electric 2 hp motor; "
    "Bearing model: SKF deep-groove ball bearing 6205-2RS JEM; "
    "Pitch diameter: 1.748 in; Ball diameter: 0.3126 in; "
    "Contact angle: 0°; Number of balls: 9"
)

FAULT_LABELS = ["NO", "IRF", "ORF", "REF"]

def load_and_segment(file_path: str, window_size: int = 2048,
                     overlap: float = 0.2, channel: str = "DE_time"):
    mat = scipy.io.loadmat(file_path)
    key = next((k for k in mat if channel in k), None)    
    signal = mat[key].flatten()
    
    signal = (signal - np.mean(signal)) / np.std(signal)   
    step = int(window_size * (1 - overlap))
    
    segments = [signal[i: i + window_size]
                for i in range(0, len(signal) - window_size, step)]
    return np.array(segments)

def compute_fft_magnitude(segment:np.ndarray):
    L = len(segment)
    X = fft(segment)
    magnitude = np.abs(X)[:L // 2]      
    return magnitude / L  

               
TIME_FEATURE_NAMES = [
    "mean", "RMS", "standard_deviation", "crest_factor",
    "skewness", "shape_factor", "kurtosis",
    "peak_to_peak", "energy_factor", "impulse_factor"
]
 
FREQ_FEATURE_NAMES = [
    "peak_frequency", "peak_to_peak_frequency",
    "spectral_kurtosis", "spectral_bandwidth", "spectral_skewness"
]