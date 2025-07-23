# Gym Movement Sensor Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing gym movement sensor data from accelerometers and gyroscopes. This application provides data validation, statistical analysis, signal processing, machine learning clustering, and anomaly detection capabilities.

## 🚀 Features

- **Data Validation & Cleaning**: Automatic outlier detection and data quality assessment
- **Statistical Analysis**: Comprehensive descriptive statistics and correlation analysis
- **Signal Processing**: Time-domain, frequency-domain, and wavelet feature extraction
- **Machine Learning**: K-Means and DBSCAN clustering with PCA visualization
- **Interactive Visualizations**: Real-time signal plots, spectrograms, and phase space analysis
- **Anomaly Detection**: Automated investigation of irregular patterns and noise points
- **Report Generation**: Downloadable analysis reports and cleaned datasets

## 📋 Requirements

```bash
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.9.0
scikit-learn>=1.2.0
pywavelets>=1.4.0
```

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/gym-sensor-analysis.git
cd gym-sensor-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Application

```bash
streamlit run src/app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## 📊 Data Format

The application expects CSV files with the following columns:

### Required Sensor Columns:
- `ax`, `ay`, `az`: Accelerometer data (g-force)
- `gx`, `gy`, `gz`: Gyroscope data (degrees/second)

### Required Metadata Columns:
- `athlete_id`: Unique identifier for each athlete
- `exercise_type`: Type of exercise being performed
- `weight_kg`: Weight used in the exercise
- `set_number`: Set number within the workout
- `rep_number`: Repetition number within the set

### Optional Columns:
- `timestamp`: Timestamp for each sensor reading (recommended for timing analysis)

### Expected Format:
- Sampling rate: 100 Hz (10ms intervals)
- File size limit: 500 MB

## 📁 Project Structure

```
gym-sensor-analysis/
├── src/
│   ├── app.py                 # Main Streamlit application
│   ├── config.py              # Configuration and constants
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Data loading and validation
│   │   └── cleaner.py         # Data cleaning and outlier detection
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── statistics.py      # Statistical analysis functions
│   │   ├── timing.py          # Timing and sampling analysis
│   │   └── features.py        # Feature extraction
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── clustering.py      # Clustering algorithms
│   │   └── pca.py            # PCA analysis
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plots.py          # Basic plotting functions
│   │   ├── signals.py        # Signal visualization
│   │   └── ml_plots.py       # ML visualization
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py        # Helper functions
│   │   └── report.py         # Report generation
│   └── ui/
│       ├── __init__.py
│       ├── sidebar.py        # Sidebar components
│       └── tabs.py           # Tab components
├── tests/
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_clustering.py
├── examples/
│   └── sample_data.csv       # Example data file
├── requirements.txt
├── README.md
└── .gitignore
```

## 🎯 Usage Guide

### 1. Upload Data
- Click on "Choose a CSV file" in the sidebar
- Select your gym sensor data CSV file
- Wait for the file to load and validate

### 2. Data Overview
- Review data statistics and quality metrics
- Check for missing values and data types
- Examine the distribution of exercises and athletes

### 3. Outlier Detection
- Automatic z-score based outlier detection
- Option to download cleaned data
- Visual representation of outliers by sensor

### 4. Feature Extraction
- Automatic extraction of time-domain, frequency-domain, and wavelet features
- Feature statistics and correlation analysis
- Download extracted features for external analysis

### 5. Machine Learning Analysis
- Adjust clustering parameters using the sliders
- Run K-Means and DBSCAN clustering
- Visualize results with PCA plots
- Download clustered data

### 6. Signal Visualization
- Select specific athlete, exercise, and repetition
- View time series plots, spectrograms, and phase space
- Analyze rolling statistics and patterns

### 7. Generate Reports
- Create comprehensive analysis reports
- Download in text format
- Include all key findings and recommendations

## 🔧 Configuration

Edit `src/config.py` to modify:
- File size limits
- Outlier detection thresholds
- Sampling rate expectations
- Feature extraction parameters
- Clustering default values

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

For issues, questions, or contributions, please:
- Open an issue on GitHub
- Contact the maintainers
- Check the documentation

## 🔍 Troubleshooting

### Common Issues:

1. **Memory Error with Large Files**
   - Reduce file size or increase system memory
   - Use data sampling features in the configuration

2. **Missing Columns Error**
   - Ensure your CSV has all required columns
   - Check column names are exactly as specified

3. **Clustering Fails**
   - Ensure sufficient data points (minimum 5)
   - Check for valid numeric data
   - Adjust clustering parameters

4. **Visualization Errors**
   - Update matplotlib backend settings
   - Check for sufficient data points
   - Verify timestamp column format