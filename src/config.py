"""
Configuration file for Gym Movement Sensor Analysis Dashboard
"""

# File handling
MAX_FILE_SIZE_MB = 500
CHUNK_SIZE = 10000  # For processing large files in chunks

# Required columns
REQUIRED_SENSOR_COLS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
METADATA_COLS = ['athlete_id', 'exercise_type', 'weight_kg', 'set_number', 'rep_number']
OPTIONAL_COLS = ['timestamp']

# Sensor data parameters
SAMPLING_RATE_HZ = 100  # Expected sampling rate
SAMPLING_INTERVAL_MS = 10  # Expected interval between samples

# Data validation thresholds
OUTLIER_Z_SCORE_THRESHOLD = 3
MISSING_DATA_WARNING_THRESHOLD = 0.05  # 5%
MISSING_DATA_ERROR_THRESHOLD = 0.20   # 20%

# Timing analysis
IRREGULAR_SAMPLING_THRESHOLD_MS = 15
MAX_REP_DURATION_MS = 10000
SAMPLING_CONSISTENCY_THRESHOLD_MS = 50

# Feature extraction
WAVELET_TYPE = 'db4'
WAVELET_LEVEL = 3
FFT_MIN_SAMPLES = 10
SPECTROGRAM_NPERSEG = 256

# Machine Learning
CLUSTERING_MIN_SAMPLES = 5
KMEANS_MAX_K = 6
KMEANS_N_INIT = 10
DBSCAN_EPS_PERCENTILE_DEFAULT = 80
DBSCAN_MIN_SAMPLES_DEFAULT = 5
PCA_N_COMPONENTS = 3

# Visualization
PLOT_STYLE = 'seaborn-v0_8'
FIGURE_DPI = 100
MAX_PLOT_SAMPLES = 1000
COLOR_PALETTE = 'viridis'

# Report generation
REPORT_TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'
REPORT_FILENAME_FORMAT = 'gym_sensor_analysis_report_%Y%m%d_%H%M%S'

# UI Settings
PAGE_TITLE = "🏋️ Gym Movement Sensor Analysis Dashboard"
PAGE_ICON = "🏋️"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Tab names
TABS = [
    "📊 Data Overview",
    "🧹 Outlier Detection",
    "⏱️ Timing Analysis",
    "🔬 Feature Extraction",
    "🤖 ML Analysis",
    "📈 Signal Plots",
    "📋 Summary Report",
    "🔍 Anomaly Investigation"
]

# CSS styling
CUSTOM_CSS = """
<style>
    .main {padding-top: 0rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {padding: 8px 16px;}
</style>
"""