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

def create_sequences(data, target_col, location_col, look_back=12):
    """
    Create sequences for time series models (e.g., LSTM) grouped by location
    """
    X, y, locations = [], [], []
    
    # Group by location if location column exists
    if location_col in data.columns:
        for location in data[location_col].unique():
            location_data = data[data[location_col] == location].copy()
            location_data = location_data.drop(columns=[location_col])  # Drop location for sequence creation
            
            # Create sequences for this location
            for i in range(len(location_data) - look_back):
                X.append(location_data.iloc[i:(i + look_back)].values)
                y.append(location_data.iloc[i + look_back][target_col])
                locations.append(location)
    else:
        # No location column, process all data together
        for i in range(len(data) - look_back):
            X.append(data.iloc[i:(i + look_back)].values)
            y.append(data.iloc[i + look_back][target_col])
            locations.append('unknown')
    
    return np.array(X), np.array(y), np.array(locations)

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
        df = pd.read_csv(data_path)
        
        # Check if index should be parsed as dates
        if df.columns[0] in ['timestamp', 'date', 'Date', 'datetime']:
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Detect location column (case-insensitive)
        location_col = None
        for col in df.columns:
            if col.lower() == 'location':
                location_col = col
                break
        
        if location_col:
            # Standardize column name to lowercase 'location'
            if location_col != 'location':
                df.rename(columns={location_col: 'location'}, inplace=True)
                location_col = 'location'
            
            unique_locations = df[location_col].unique()
            print(f"📍 Found location column with {len(unique_locations)} unique locations:")
            for loc in unique_locations:
                count = len(df[df[location_col] == loc])
                print(f"   - {loc}: {count} rows")
        else:
            print("⚠️ No location column found - processing as single dataset")
        
        # Define weights for WQI calculation
        weights = {
            'pH Level': 0.15,
            'Dissolved Oxygen (mg/L)': 0.25,
            'Nitrate-N/Nitrite-N (mg/L)': 0.10,
            'Nitrate-N/Nitrite-N  (mg/L)': 0.10,  # Handle spacing variation
            'Ammonia (mg/L)': 0.15,
            'Phosphate (mg/L)': 0.10,
            'Surface Water Temp (°C)': 0.05,
            'Middle Water Temp (°C)': 0.05,
            'Bottom Water Temp (°C)': 0.05,
        }
        
        # Calculate WQI
        print("\n🧮 Calculating Water Quality Index (WQI)...")
        
        print('DataFrame shape before imputation:', df.shape)
        print('NaN counts before imputation:')
        print(df.isna().sum())
        
        # Convert weight columns to numeric
        for col in weights.keys():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"⚠️ Warning: Column '{col}' not found in data")
        
        # Separate location column before imputation
        location_data = None
        if location_col and location_col in df.columns:
            location_data = df[location_col].copy()
            df_processing = df.drop(columns=[location_col])
        else:
            df_processing = df.copy()
        
        # Apply forward fill, then backward fill
        wqi_cols = [col for col in weights.keys() if col in df_processing.columns]
        df_processing[wqi_cols] = df_processing[wqi_cols].ffill().bfill()
        
        print('DataFrame shape after imputation:', df_processing.shape)
        print('NaN counts after imputation:')
        print(df_processing.isna().sum())
        
        # Impute missing values in non-WQI columns
        non_wqi_cols = [col for col in df_processing.columns if col not in wqi_cols]
        df_processing[non_wqi_cols] = df_processing[non_wqi_cols].ffill().bfill()
        
        print('DataFrame shape after imputing non-WQI columns:', df_processing.shape)
        print('NaN counts after imputing non-WQI columns:')
        print(df_processing.isna().sum())
        
        # Calculate WQI using available columns
        available_weights = {col: weight for col, weight in weights.items() if col in df_processing.columns}
        df_processing['WQI'] = calculate_wqi(df_processing, available_weights)
        
        print(f"✅ WQI calculated - Range: {df_processing['WQI'].min():.2f} to {df_processing['WQI'].max():.2f}")
        
        # Add location column back
        if location_data is not None:
            df_processing['location'] = location_data.values
            print("✅ Location column preserved in processed data")
        
        # Handle missing values (drop remaining NaN)
        initial_rows = len(df_processing)
        df_processing = df_processing.dropna()
        final_rows = len(df_processing)
        
        if initial_rows != final_rows:
            print(f"🔄 Dropped {initial_rows - final_rows} rows with missing values")
        
        # Convert non-numeric columns to numeric using one-hot encoding (except location)
        columns_to_exclude = ['location'] if 'location' in df_processing.columns else []
        non_numeric_cols = df_processing.select_dtypes(include=['object', 'category']).columns
        non_numeric_cols = [col for col in non_numeric_cols if col not in columns_to_exclude]
        
        if len(non_numeric_cols) > 0:
            print(f"\n🔄 Converting non-numeric columns to numeric: {list(non_numeric_cols)}")
            df_processing = pd.get_dummies(df_processing, columns=non_numeric_cols)
            print("✅ Non-numeric columns converted using one-hot encoding")

        # Normalize features (excluding location column)
        print("\n📊 Normalizing features...")
        scaler = MinMaxScaler()
        
        # Separate location before normalization
        if 'location' in df_processing.columns:
            location_preserved = df_processing['location'].copy()
            numeric_cols = df_processing.select_dtypes(include=['float64', 'int64', 'uint8']).columns
            df_processing[numeric_cols] = scaler.fit_transform(df_processing[numeric_cols])
            # Keep location as is (not normalized)
            df_processing['location'] = location_preserved
        else:
            numeric_cols = df_processing.select_dtypes(include=['float64', 'int64', 'uint8']).columns
            df_processing[numeric_cols] = scaler.fit_transform(df_processing[numeric_cols])
        
        print("✅ Features normalized using MinMaxScaler")

        # Create output directory
        output_dir = Path('data/processed')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the processed DataFrame to CSV with location column
        output_csv = output_dir / 'model_processed_data.csv'
        df_processing.to_csv(output_csv, index=True)
        print(f"✅ Processed data saved to: {output_csv}")
        
        if 'location' in df_processing.columns:
            print(f"✅ Location column included in output file")
            print(f"   Locations preserved: {sorted(df_processing['location'].unique())}")
        
        # Create sequences for time series models (e.g., LSTM)
        print(f"\n🔄 Creating sequences for time series prediction...")
        
        # Ensure WQI column exists
        if 'WQI' not in df_processing.columns:
            raise KeyError("WQI column not found in the DataFrame. Ensure WQI is calculated before creating sequences.")
        
        # Create sequences with location tracking
        X, y, locations = create_sequences(df_processing, 'WQI', 'location' if 'location' in df_processing.columns else None, look_back=12)
        print(f"✅ Created {len(X)} sequences with look-back window of 12")
        
        if 'location' in df_processing.columns:
            print(f"✅ Location information preserved for each sequence")
            unique_seq_locations = np.unique(locations)
            for loc in unique_seq_locations:
                loc_count = np.sum(locations == loc)
                print(f"   - {loc}: {loc_count} sequences")

        # Split data into train/test sets (80/20 split)
        print(f"\n✂️ Splitting data into train/test sets (80/20 split)...")
        X_train, X_test, y_train, y_test, loc_train, loc_test = train_test_split(
            X, y, locations, test_size=0.2, random_state=42
        )
        
        print(f"✅ Training samples: {len(X_train)}")
        print(f"✅ Testing samples: {len(X_test)}")
        
        # Save arrays as float32
        print(f"\n💾 Saving training arrays...")
        np.save(output_dir / 'X_train.npy', X_train.astype(np.float32))
        np.save(output_dir / 'X_test.npy', X_test.astype(np.float32))
        np.save(output_dir / 'y_train.npy', y_train.astype(np.float32))
        np.save(output_dir / 'y_test.npy', y_test.astype(np.float32))
        np.save(output_dir / 'locations_train.npy', loc_train)
        np.save(output_dir / 'locations_test.npy', loc_test)
        print("✅ Location arrays saved for train/test sets")
        
        # Save scaler for future use
        import joblib
        joblib.dump(scaler, output_dir / 'scaler.pkl')
        print("✅ Scaler saved for future use")
        
        # Print final statistics
        print(f"\n📋 Final Statistics:")
        print('Non-null counts for all columns after processing:')
        print(df_processing.count())
        
        print(f"\n🎉 DATA PREPARATION COMPLETED!")
        print(f"✅ Processed data shape: {df_processing.shape}")
        print(f"✅ Training sequences: {X_train.shape}")
        print(f"✅ Testing sequences: {X_test.shape}")
        
        if 'location' in df_processing.columns:
            print(f"✅ Location column: PRESERVED")
            print(f"✅ Unique locations in output: {len(df_processing['location'].unique())}")
        
        print(f"✅ All files saved in: {output_dir.absolute()}")
        
        print(f"\n🚀 Ready for model training!")
        print(f"Load your data with:")
        print(f"   X_train = np.load('{output_dir}/X_train.npy')")
        print(f"   X_test = np.load('{output_dir}/X_test.npy')")
        print(f"   y_train = np.load('{output_dir}/y_train.npy')")
        print(f"   y_test = np.load('{output_dir}/y_test.npy')")
        print(f"   loc_train = np.load('{output_dir}/locations_train.npy')")
        print(f"   loc_test = np.load('{output_dir}/locations_test.npy')")
        
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