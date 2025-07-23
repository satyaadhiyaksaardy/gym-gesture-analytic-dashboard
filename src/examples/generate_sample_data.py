"""
Generate sample gym sensor data for testing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_sample_data(n_athletes=5, n_exercises=3, n_sets=3, n_reps=10, 
                        samples_per_rep=100, output_file='sample_data.csv'):
    """
    Generate synthetic gym sensor data
    
    Args:
        n_athletes: Number of athletes
        n_exercises: Number of exercise types
        n_sets: Number of sets per exercise
        n_reps: Number of reps per set
        samples_per_rep: Number of samples per repetition
        output_file: Output CSV filename
    """
    
    # Exercise types
    exercise_types = ['squat', 'deadlift', 'bench_press', 'overhead_press', 'barbell_row'][:n_exercises]
    
    # Initialize data list
    data = []
    
    # Starting timestamp
    base_time = datetime.now()
    current_time = base_time
    
    # Generate data
    for athlete_id in range(1, n_athletes + 1):
        for exercise in exercise_types:
            # Weight varies by exercise and athlete
            base_weight = {
                'squat': 80 + athlete_id * 10,
                'deadlift': 100 + athlete_id * 10,
                'bench_press': 60 + athlete_id * 5,
                'overhead_press': 40 + athlete_id * 5,
                'barbell_row': 50 + athlete_id * 5
            }.get(exercise, 50)
            
            for set_num in range(1, n_sets + 1):
                # Weight might increase with sets
                weight = base_weight + (set_num - 1) * 5
                
                for rep_num in range(1, n_reps + 1):
                    # Fatigue factor increases with reps
                    fatigue_factor = 1 + (rep_num - 1) * 0.05
                    
                    # Generate sensor data for this rep
                    for sample in range(samples_per_rep):
                        # Time progression (10ms intervals with some jitter)
                        time_increment = 10 + np.random.normal(0, 1)
                        current_time += timedelta(milliseconds=time_increment)
                        
                        # Generate accelerometer data (g-force)
                        # Base pattern depends on exercise
                        t = sample / samples_per_rep * 2 * np.pi
                        
                        if exercise == 'squat':
                            ax = 0.1 * np.sin(t) + np.random.normal(0, 0.05)
                            ay = 0.2 * np.cos(t) + np.random.normal(0, 0.05)
                            az = 1.0 + 0.3 * np.sin(2*t) + np.random.normal(0, 0.05)
                        elif exercise == 'deadlift':
                            ax = 0.15 * np.sin(t) + np.random.normal(0, 0.05)
                            ay = 0.1 * np.cos(t) + np.random.normal(0, 0.05)
                            az = 1.0 + 0.4 * np.sin(t) + np.random.normal(0, 0.05)
                        else:
                            ax = 0.2 * np.sin(t) + np.random.normal(0, 0.05)
                            ay = 0.15 * np.cos(t) + np.random.normal(0, 0.05)
                            az = 1.0 + 0.2 * np.sin(2*t) + np.random.normal(0, 0.05)
                        
                        # Apply fatigue
                        ax *= fatigue_factor
                        ay *= fatigue_factor
                        az = az * 0.9 + 0.1 * fatigue_factor
                        
                        # Generate gyroscope data (deg/s)
                        gx = 50 * np.sin(t + np.pi/4) * fatigue_factor + np.random.normal(0, 5)
                        gy = 40 * np.cos(t + np.pi/3) * fatigue_factor + np.random.normal(0, 5)
                        gz = 30 * np.sin(2*t) * fatigue_factor + np.random.normal(0, 5)
                        
                        # Add occasional outliers (1% chance)
                        if np.random.random() < 0.01:
                            if np.random.random() < 0.5:
                                ax *= 3
                            else:
                                gx *= 3
                        
                        # Append data
                        data.append({
                            'timestamp': int(current_time.timestamp() * 1000),  # milliseconds
                            'athlete_id': athlete_id,
                            'exercise_type': exercise,
                            'weight_kg': weight,
                            'set_number': set_num,
                            'rep_number': rep_num,
                            'ax': ax,
                            'ay': ay,
                            'az': az,
                            'gx': gx,
                            'gy': gy,
                            'gz': gz
                        })
                    
                    # Add inter-rep pause (1-2 seconds)
                    current_time += timedelta(seconds=np.random.uniform(1, 2))
                
                # Add inter-set pause (60-120 seconds)
                current_time += timedelta(seconds=np.random.uniform(60, 120))
            
            # Add inter-exercise pause (3-5 minutes)
            current_time += timedelta(minutes=np.random.uniform(3, 5))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values (0.5% of sensor data)
    sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    for col in sensor_cols:
        mask = np.random.random(len(df)) < 0.005
        df.loc[mask, col] = np.nan
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df):,} samples")
    print(f"Saved to: {output_file}")
    
    # Print summary
    print("\nData Summary:")
    print(f"- Athletes: {df['athlete_id'].nunique()}")
    print(f"- Exercises: {df['exercise_type'].nunique()}")
    print(f"- Total reps: {len(df.groupby(['athlete_id', 'exercise_type', 'set_number', 'rep_number']))}")
    print(f"- Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"- Missing values: {df[sensor_cols].isna().sum().sum()}")
    
    return df


if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_data(
        n_athletes=5,
        n_exercises=3,
        n_sets=3,
        n_reps=10,
        samples_per_rep=100,
        output_file='sample_gym_sensor_data.csv'
    )
    
    # Show first few rows
    print("\nFirst 5 rows:")
    print(df.head())