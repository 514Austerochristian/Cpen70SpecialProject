import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def find_csv_files(max_depth=3):
    """
    Recursively find all CSV files in the repository
    """
    csv_files = []
    root_path = Path.cwd()
    
    # Method 1: Using rglob for recursive search
    try:
        csv_files = list(root_path.rglob("*.csv"))
    except Exception as e:
        print(f"⚠️ rglob search failed: {e}")
        
        # Method 2: Walk through directories manually (backup method)
        try:
            for root, dirs, files in os.walk(root_path):
                # Limit depth
                level = root.replace(str(root_path), '').count(os.sep)
                if level < max_depth:
                    for file in files:
                        if file.lower().endswith('.csv'):
                            csv_files.append(Path(root) / file)
        except Exception as e:
            print(f"⚠️ Manual walk failed: {e}")
    
    # Remove duplicates and sort
    csv_files = list(set(csv_files))
    csv_files.sort()
    
    return csv_files

def show_available_csv_files():
    """
    Show all CSV files found in the repository with improved search
    """
    print("\n🔍 Searching for CSV files in the repository...")
    csv_files = find_csv_files()
    
    if not csv_files:
        print("❌ No CSV files found in the repository")
        
        # Show current directory contents for debugging
        print(f"\n📂 Current directory: {Path.cwd()}")
        print("📂 Contents of current directory:")
        try:
            for item in Path.cwd().iterdir():
                if item.is_file():
                    print(f"   📄 {item.name}")
                elif item.is_dir():
                    print(f"   📁 {item.name}/")
        except Exception as e:
            print(f"   ⚠️ Could not list directory contents: {e}")
        
        return []
    
    print(f"\n📊 Found {len(csv_files)} CSV files:")
    
    # Group files by directory for better organization
    files_by_dir = {}
    for file_path in csv_files:
        dir_path = file_path.parent
        if dir_path not in files_by_dir:
            files_by_dir[dir_path] = []
        files_by_dir[dir_path].append(file_path)
    
    # Display organized by directory
    for dir_path, files in sorted(files_by_dir.items()):
        relative_dir = dir_path.relative_to(Path.cwd()) if dir_path != Path.cwd() else Path(".")
        print(f"\n📁 {relative_dir}/")
        
        for i, file_path in enumerate(sorted(files), 1):
            file_size = get_file_size(file_path)
            file_modified = get_file_modified(file_path)
            relative_path = file_path.relative_to(Path.cwd())
            print(f"   {i:2d}. {file_path.name:<30} ({file_size:<8}) {file_modified} -> {relative_path}")
    
    print(f"\n💡 Tip: Copy the relative path (e.g., 'data/processed_data.csv') to use as input")
    return csv_files

def get_file_size(file_path):
    """Get human-readable file size"""
    try:
        size_bytes = file_path.stat().st_size
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f}KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f}MB"
        else:
            return f"{size_bytes/(1024**3):.1f}GB"
    except:
        return "N/A"

def get_file_modified(file_path):
    """Get file modification time"""
    try:
        import datetime
        mtime = file_path.stat().st_mtime
        return datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    except:
        return "N/A"

def get_input_file_path():
    """
    Get input CSV file path from user with improved file detection
    """
    while True:
        print("\n📁 Please provide the path to your processed CSV file:")
        print("   Examples:")
        print("   - 'data/processed/processed_data.csv'")
        print("   - './output/processed_data.csv'")
        print("   - 'my_data.csv'")
        print("   - Or type 'list' to see all available CSV files")
        
        file_path = input("Enter data file path (or 'list'): ").strip()
        file_path = file_path.strip('"').strip("'")
        
        # Show available files if requested
        if file_path.lower() in ['list', 'l', 'show', 'files']:
            show_available_csv_files()
            continue
        
        path = Path(file_path)
        
        if path.exists() and path.is_file():
            print(f"✅ File found: {path.absolute()}")
            return str(path)
        else:
            print(f"❌ File not found: {path.absolute()}")
            
            # Automatically show available files
            print("\n🔍 Searching for CSV files in the repository...")
            available_files = show_available_csv_files()
            
            if available_files:
                print("\nTry using one of the paths shown above.")

def calculate_wqi(df, weights):
    """
    Calculate Water Quality Index using weighted parameters
    """
    weighted_values = df[weights.keys()].apply(lambda x: x * weights[x.name], axis=0)
    wqi = weighted_values.sum(axis=1)
    return wqi

def create_sequences(data, target_col, look_back=12):
    """
    Create sequences for time series models (e.g., LSTM)
    """
    
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data.iloc[i:(i + look_back)].values)
        y.append(data.iloc[i + look_back][target_col])
    
    return np.array(X), np.array(y)

def main():
    """
    Main function for water quality data preparation
    """
    print("🌊 WATER QUALITY DATA PREPARATION")
    print("=" * 50)
    
    try:
        # Get input file path with improved detection
        data_path = get_input_file_path()
        
        # Load the processed data
        print(f"\n📂 Loading data from: {data_path}")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Define weights for WQI calculation
        weights = {
            'pH Level': 0.15,
            'Dissolved Oxygen (mg/L)': 0.25,
            'Nitrate-N/Nitrite-N (mg/L)': 0.10,
            'Ammonia (mg/L)': 0.15,
            'Phosphate (mg/L)': 0.10,
            'Surface Water Temp (°C)': 0.05,
            'Middle Water Temp (°C)': 0.05,
            'Bottom Water Temp (°C)': 0.05,
        }
        
        # Calculate WQI
        print("\n🧮 Calculating Water Quality Index (WQI)...")
        
        # Ensure all columns used in WQI are numeric
        print('DataFrame shape before imputation:', df.shape)
        print('NaN counts before imputation:')
        print(df.isna().sum())
        
        for col in weights.keys():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"⚠️ Warning: Column '{col}' not found in data")
        
        # Apply forward fill, then backward fill to handle missing values in WQI columns
        wqi_cols = [col for col in weights.keys() if col in df.columns]
        df[wqi_cols] = df[wqi_cols].ffill().bfill()
        
        print('DataFrame shape after imputation:', df.shape)
        print('NaN counts after imputation:')
        print(df.isna().sum())
        
        # Impute missing values in non-WQI columns
        non_wqi_cols = [col for col in df.columns if col not in wqi_cols]
        df[non_wqi_cols] = df[non_wqi_cols].fillna(method='ffill').fillna(method='bfill')
        
        print('DataFrame shape after imputing non-WQI columns:', df.shape)
        print('NaN counts after imputing non-WQI columns:')
        print(df.isna().sum())
        
        # Calculate WQI using available columns
        available_weights = {col: weight for col, weight in weights.items() if col in df.columns}
        df['WQI'] = calculate_wqi(df, available_weights)
        
        print(f"✅ WQI calculated - Range: {df['WQI'].min():.2f} to {df['WQI'].max():.2f}")
        
        # Handle missing values (drop remaining NaN)
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        if initial_rows != final_rows:
            print(f"🔄 Dropped {initial_rows - final_rows} rows with missing values")
        
        # Convert non-numeric columns to numeric using one-hot encoding
        non_numeric_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(non_numeric_cols) > 0:
            print(f"\n🔄 Converting non-numeric columns to numeric: {list(non_numeric_cols)}")
            df = pd.get_dummies(df, columns=non_numeric_cols)
            print("✅ Non-numeric columns converted using one-hot encoding")

        # Normalize features
        print("\n📊 Normalizing features...")
        scaler = MinMaxScaler()
        numeric_cols = df.select_dtypes(include=['float64', 'int64', 'uint8']).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        print("✅ Features normalized using MinMaxScaler")

        # Create output directory
        output_dir = Path('data/processed')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the processed DataFrame to CSV
        output_csv = output_dir / 'model_processed_data.csv'
        df.to_csv(output_csv, index=True)
        print(f"✅ Processed data saved to: {output_csv}")
        
        # Create sequences for time series models (e.g., LSTM)
        print(f"\n🔄 Creating sequences for time series prediction...")
        X, y = create_sequences(df, 'WQI', look_back=12)
        print(f"✅ Created {len(X)} sequences with look-back window of 12")
        
        # Create sequences for water quality index (WQI)
        if 'WQI' not in df.columns:
            raise KeyError("WQI column not found in the DataFrame. Ensure WQI is calculated before creating sequences.")
        print(f"\n📊 Creating sequences for WQI...")
        if 'WQI' in df.columns:
            X, y = create_sequences(df, 'WQI', look_back=12)
            print(f"✅ Created {len(X)} sequences for WQI with look-back window of 12")

        # Split data into train/test sets (80/20 split)
        print(f"\n✂️ Splitting data into train/test sets (80/20 split)...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"✅ Training samples: {len(X_train)}")
        print(f"✅ Testing samples: {len(X_test)}")
        
        # Save arrays as float32
        print(f"\n💾 Saving training arrays...")
        np.save(output_dir / 'X_train.npy', X_train.astype(np.float32))
        np.save(output_dir / 'X_test.npy', X_test.astype(np.float32))
        np.save(output_dir / 'y_train.npy', y_train.astype(np.float32))
        np.save(output_dir / 'y_test.npy', y_test.astype(np.float32))
        
        # Save scaler for future use
        import joblib
        joblib.dump(scaler, output_dir / 'scaler.pkl')
        print("✅ Scaler saved for future use")
        
        # Print final statistics
        print(f"\n📋 Final Statistics:")
        print('Non-null counts for all columns after processing:')
        print(df.count())
        
        print(f"\n🎉 DATA PREPARATION COMPLETED!")
        print(f"✅ Processed data shape: {df.shape}")
        print(f"✅ Training sequences: {X_train.shape}")
        print(f"✅ Testing sequences: {X_test.shape}")
        print(f"✅ All files saved in: {output_dir.absolute()}")
        
        print(f"\n🚀 Ready for model training!")
        print(f"Load your data with:")
        print(f"   X_train = np.load('{output_dir}/X_train.npy')")
        print(f"   X_test = np.load('{output_dir}/X_test.npy')")
        print(f"   y_train = np.load('{output_dir}/y_train.npy')")
        print(f"   y_test = np.load('{output_dir}/y_test.npy')")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Please check the file path and try again.")
    except KeyError as e:
        print(f"❌ Missing column in data: {e}")
        print("Please check that your data contains the required columns.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # run the main data preparation pipeline:
    main()
    
# This script prepares water quality data for modeling by calculating the Water Quality Index (WQI),
# normalizing features, creating sequences for time series prediction, and saving the processed data.
# It includes improved file detection, error handling, and detailed logging for better user experience.
# The output is saved in a structured format for easy access during model training.
