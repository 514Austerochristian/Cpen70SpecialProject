import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import os
import sys
from pathlib import Path

def get_file_path(file_description, file_type="CSV"):
    """
    Get file path from user input with validation
    """
    while True:
        print(f"\n📁 Please provide the path to your {file_description} {file_type} file:")
        print("   Examples:")
        print("   - './data/water_parameters.csv'")
        print("   - '/full/path/to/your/file.csv'")
        print("   - 'data/climate_data.csv'")
        
        file_path = input(f"Enter {file_description} file path: ").strip()
        
        # Remove quotes if user added them
        file_path = file_path.strip('"').strip("'")
        
        # Convert to Path object for better handling
        path = Path(file_path)
        
        if path.exists() and path.is_file():
            print(f"✅ File found: {path.absolute()}")
            return str(path)
        else:
            print(f"❌ File not found: {path.absolute()}")
            print("Please check the path and try again.")
            
            # Option to list files in current directory
            response = input("Would you like to see files in the current directory? (y/n): ").lower()
            if response in ['y', 'yes']:
                current_dir = Path.cwd()
                print(f"\n📂 Files in {current_dir}:")
                try:
                    csv_files = list(current_dir.glob("*.csv"))
                    if csv_files:
                        for i, file in enumerate(csv_files, 1):
                            print(f"   {i}. {file.name}")
                    else:
                        print("   No CSV files found in current directory")
                        
                    # Check for data subdirectory
                    data_dir = current_dir / "data"
                    if data_dir.exists():
                        print(f"\n📂 Files in {data_dir}:")
                        data_csv_files = list(data_dir.glob("*.csv"))
                        for i, file in enumerate(data_csv_files, 1):
                            print(f"   {i}. {file.relative_to(current_dir)}")
                except Exception as e:
                    print(f"   Error listing files: {e}")

def clean_and_normalize(file_path):
    """
    Comprehensive data cleaning function that addresses all required tasks:
    1. Handle missing values (gaps in time series)
    2. Fill remaining missing values
    3. Detect and remove outliers
    4. Ensure consistent timestamp format
    5. Check and correct data types
    """
    print(f"\n📋 Processing file: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded file with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None, None, None
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    print(f"📊 Available columns: {list(df.columns)}")

    # Store original location information if it exists
    location_info = None
    has_location = False
    location_column_names = ["Location", "location", "LOCATION", "Site", "site", "SITE"]
    actual_location_column = None
    
    for loc_col in location_column_names:
        if loc_col in df.columns:
            actual_location_column = loc_col
            has_location = True
            # Create a mapping of index to location for later restoration
            location_info = df[loc_col].copy()
            print(f"📍 Found location column: '{loc_col}' with {df[loc_col].nunique()} unique locations")
            break
    
    if not has_location:
        print("📍 No location column found - will process as single dataset")

    # TASK 4: ENSURE CONSISTENT TIMESTAMP FORMAT
    print("📅 Task 4: Converting timestamps to standardized format...")

    # Handle climate data with YEAR and MONTH columns
    if "YEAR" in df.columns and "MONTH" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str),
            errors="coerce"
        )
        df.drop(columns=["YEAR", "MONTH"], inplace=True)
        print("   ✓ Converted YEAR/MONTH columns to standardized timestamp")

    # Handle water data with Date column
    elif "Date" in df.columns:
        df.rename(columns={"Date": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        print("   ✓ Converted Date column to standardized timestamp")
    
    # Handle other common timestamp column names
    elif "date" in df.columns:
        df.rename(columns={"date": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        print("   ✓ Converted date column to standardized timestamp")
    
    elif "datetime" in df.columns:
        df.rename(columns={"datetime": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        print("   ✓ Converted datetime column to standardized timestamp")
    
    elif "time" in df.columns:
        df.rename(columns={"time": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        print("   ✓ Converted time column to standardized timestamp")
    
    else:
        print("⚠️  No standard timestamp column found. Available columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col}")
        
        while True:
            try:
                choice = input("\nEnter the number of the column to use as timestamp (or 'skip' to continue without timestamp): ").strip()
                if choice.lower() == 'skip':
                    print("   ⚠️ Proceeding without timestamp conversion")
                    break
                
                col_index = int(choice) - 1
                if 0 <= col_index < len(df.columns):
                    timestamp_col = df.columns[col_index]
                    df.rename(columns={timestamp_col: "timestamp"}, inplace=True)
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                    print(f"   ✓ Converted {timestamp_col} column to standardized timestamp")
                    break
                else:
                    print("   ❌ Invalid selection. Please try again.")
            except ValueError:
                print("   ❌ Please enter a valid number or 'skip'")

    # Drop rows with missing timestamp (if timestamp exists)
    if "timestamp" in df.columns:
        initial_rows = len(df)
        df.dropna(subset=["timestamp"], inplace=True)
        dropped_timestamp = initial_rows - len(df)
        if dropped_timestamp > 0:
            print(f"   ⚠️ Removed {dropped_timestamp} rows with invalid timestamps")
            # Update location info after dropping rows
            if has_location:
                location_info = location_info[df.index]

    # TASK 5: CHECK AND CORRECT DATA TYPES
    print("🔢 Task 5: Checking and correcting data types...")

    # Process data by location if Location column exists
    if has_location:
        # Remove any rows where location is missing
        df.dropna(subset=[actual_location_column], inplace=True)
        location_info = location_info[df.index]
        
        cleaned_by_location = {}
        normalized_by_location = {}
        location_mapping = {}  # Store original location for each processed row

        for location in df[actual_location_column].unique():
            print(f"\n📍 Processing location: {location}")
            subset = df[df[actual_location_column] == location].copy()
            original_indices = subset.index.tolist()

            # Store location mapping
            location_mapping[location] = location

            # Convert to numeric data types (Task 5)
            numeric_conversions = 0
            for col in subset.columns:
                if col not in ["timestamp", actual_location_column]:
                    original_dtype = subset[col].dtype
                    subset[col] = pd.to_numeric(subset[col], errors="coerce")
                    if original_dtype != subset[col].dtype:
                        numeric_conversions += 1

            if numeric_conversions > 0:
                print(f"   ✓ Converted {numeric_conversions} columns to numeric format")

            # TASK 1: HANDLE MISSING VALUES (GAPS IN TIME SERIES)
            if "timestamp" in subset.columns:
                print("🕐 Task 1: Handling gaps in time series using time-based patterns...")

                # Preserve location before setting timestamp as index
                subset_location = subset[actual_location_column].iloc[0]
                
                # Set timestamp as index and resample to daily frequency
                subset_work = subset.drop(columns=[actual_location_column]).set_index("timestamp").resample("D").mean()

                # Count gaps that were filled
                gaps_filled = subset_work.isnull().sum().sum()
                if gaps_filled > 0:
                    print(f"   📈 Identified {gaps_filled} gaps in time series data")
                
                # Add location back
                subset_work['Location'] = subset_location
            else:
                print("   ⚠️ No timestamp column found, skipping time series gap handling")
                subset_work = subset.copy()
                if actual_location_column != 'Location':
                    subset_work.rename(columns={actual_location_column: 'Location'}, inplace=True)

            # TASK 2: FILL REMAINING MISSING VALUES
            print("🔄 Task 2: Filling remaining missing values using forward/backward fill...")

            # Separate location column before filling
            if 'Location' in subset_work.columns:
                location_col = subset_work['Location'].copy()
                subset_for_filling = subset_work.drop(columns=['Location'])
            else:
                location_col = pd.Series([location] * len(subset_work))
                subset_for_filling = subset_work.copy()

            # Forward fill first (use previous values)
            subset_filled = subset_for_filling.ffill()
            # Then backward fill for any remaining NaNs at the beginning
            subset_filled = subset_filled.bfill()

            # Add location back
            subset_filled['Location'] = location_col

            # Count how many values were filled
            filled_values = (subset_for_filling.isnull() & subset_filled.drop(columns=['Location']).notnull()).sum().sum()
            if filled_values > 0:
                print(f"   ✓ Filled {filled_values} missing values using forward/backward fill")

            # Drop columns that still have all NaN values (except Location)
            subset_clean = subset_filled.dropna(axis=1, how="all")
            # Ensure Location column is preserved even if it was dropped
            if 'Location' not in subset_clean.columns:
                subset_clean['Location'] = location_col
            
            dropped_cols = len(subset_filled.columns) - len(subset_clean.columns)
            if 'Location' not in subset_filled.columns:
                dropped_cols += 1  # Account for added Location column
            if dropped_cols > 0:
                print(f"   ⚠️ Dropped {dropped_cols} columns with insufficient data")

            # Skip if empty after cleaning (but preserve location info)
            if subset_clean.drop(columns=['Location']).empty:
                print(f"   ❌ Skipping location '{location}' — no valid numeric data after cleaning")
                # Still create a minimal entry to preserve location
                minimal_entry = pd.DataFrame({'Location': [location]})
                cleaned_by_location[location] = minimal_entry
                normalized_by_location[location] = minimal_entry.copy()
                continue

            # TASK 3: DETECT AND REMOVE OUTLIERS
            print("🎯 Task 3: Detecting and removing outliers (>3 standard deviations)...")

            # Only apply outlier detection to numeric columns (exclude Location)
            numeric_cols = subset_clean.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Calculate z-scores to identify outliers
                z_scores = np.abs(stats.zscore(subset_clean[numeric_cols]))
                outlier_mask = (z_scores < 3).all(axis=1)

                outliers_removed = len(subset_clean) - outlier_mask.sum()
                
                # Apply outlier filter but keep Location column
                subset_clean_filtered = subset_clean[outlier_mask].copy()

                if outliers_removed > 0:
                    print(f"   ✓ Removed {outliers_removed} outlier data points")
                
                subset_clean = subset_clean_filtered
            else:
                print("   ⚠️ No numeric columns found for outlier detection")

            # Ensure we still have some data (preserve location even if no numeric data)
            if len(subset_clean) == 0:
                print(f"   ⚠️ No data points remaining for '{location}' after outlier removal - creating minimal entry")
                subset_clean = pd.DataFrame({'Location': [location]})

            # Normalize data using MinMaxScaler (excluding Location)
            print("📊 Normalizing data to 0-1 scale...")
            scaler = MinMaxScaler()
            numeric_cols = subset_clean.select_dtypes(include=[np.number]).columns
            subset_normalized = subset_clean.copy()
            
            if len(numeric_cols) > 0:
                subset_normalized[numeric_cols] = scaler.fit_transform(subset_clean[numeric_cols])
                print(f"   ✓ Normalized {len(numeric_cols)} numeric columns")
            else:
                print("   ⚠️ No numeric columns found for normalization")

            cleaned_by_location[location] = subset_clean
            normalized_by_location[location] = subset_normalized

            print(f"   ✅ Successfully processed location '{location}' with {len(subset_clean)} data points")

        return cleaned_by_location, normalized_by_location, location_mapping

    else:  # Process entire dataframe if no Location column
        print("📊 Processing entire dataset (no location grouping)")

        # TASK 5: Convert to numeric data types
        numeric_conversions = 0
        for col in df.columns:
            if col != "timestamp":
                original_dtype = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if original_dtype != df[col].dtype:
                    numeric_conversions += 1

        if numeric_conversions > 0:
            print(f"   ✓ Converted {numeric_conversions} columns to numeric format")

        # TASK 1: Handle gaps in time series
        if "timestamp" in df.columns:
            print("🕐 Task 1: Handling gaps in time series using time-based patterns...")
            df = df.set_index("timestamp").resample("D").mean()

            gaps_filled = df.isnull().sum().sum()
            if gaps_filled > 0:
                print(f"   📈 Identified {gaps_filled} gaps in time series data")
        else:
            print("   ⚠️ No timestamp column found, skipping time series gap handling")

        # TASK 2: Fill remaining missing values
        print("🔄 Task 2: Filling remaining missing values using forward/backward fill...")
        df_filled = df.ffill().bfill()

        filled_values = (df.isnull() & df_filled.notnull()).sum().sum()
        if filled_values > 0:
            print(f"   ✓ Filled {filled_values} missing values using forward/backward fill")

        df_clean = df_filled.dropna(axis=1, how="all")
        dropped_cols = len(df_filled.columns) - len(df_clean.columns)
        if dropped_cols > 0:
            print(f"   ⚠️ Dropped {dropped_cols} columns with insufficient data")

        if df_clean.empty:
            print(f"   ❌ No valid data remaining after cleaning")
            return pd.DataFrame(), pd.DataFrame(), {}

        # TASK 3: Detect and remove outliers
        print("🎯 Task 3: Detecting and removing outliers (>3 standard deviations)...")
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            z_scores = np.abs(stats.zscore(df_clean[numeric_cols]))
            outlier_mask = (z_scores < 3).all(axis=1)

            outliers_removed = len(df_clean) - outlier_mask.sum()
            df_clean = df_clean[outlier_mask]

            if outliers_removed > 0:
                print(f"   ✓ Removed {outliers_removed} outlier data points")
        else:
            print("   ⚠️ No numeric columns found for outlier detection")

        if df_clean.empty:
            print(f"   ❌ No data remaining after outlier filtering")
            return pd.DataFrame(), pd.DataFrame(), {}

        # Normalize data
        print("📊 Normalizing data to 0-1 scale...")
        scaler = MinMaxScaler()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_normalized = df_clean.copy()
        if len(numeric_cols) > 0:
            df_normalized[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
            print(f"   ✓ Normalized {len(numeric_cols)} numeric columns")
        else:
            print("   ⚠️ No numeric columns found for normalization")

        print(f"   ✅ Successfully processed dataset with {len(df_clean)} data points")
        return df_clean, df_normalized, {}

def merge_datasets(water_data, climate_data, water_locations, climate_locations, data_type="cleaned"):
    """
    Merge water and climate datasets into a single dataframe while preserving location information
    """
    print(f"\n🔗 Merging {data_type} datasets...")

    # Handle dictionary (location-based) water data with dataframe climate data
    if isinstance(water_data, dict) and not isinstance(climate_data, dict):
        print("📍 Combining location-based water data with global climate data...")
        combined_dataframes = []

        for location, water_df in water_data.items():
            if not water_df.empty:
                # Reset index to make timestamp a column for merging
                water_df_reset = water_df.reset_index()
                
                # Ensure Location column exists
                if 'Location' not in water_df_reset.columns:
                    water_df_reset['Location'] = location
                
                if not climate_data.empty:
                    climate_df_reset = climate_data.reset_index()

                    # Merge on timestamp if both have it
                    if 'timestamp' in water_df_reset.columns and 'timestamp' in climate_df_reset.columns:
                        merged_df = pd.merge(water_df_reset, climate_df_reset,
                                           on='timestamp', how='left',  # Use left join to preserve all water data
                                           suffixes=('_water', '_climate'))
                    else:
                        # If no timestamp, add climate data as additional columns
                        print(f"   ⚠️ No timestamp columns found, adding climate data as additional columns for {location}")
                        # Add climate data as average values for this location
                        climate_means = climate_df_reset.select_dtypes(include=[np.number]).mean()
                        for col, value in climate_means.items():
                            merged_df = water_df_reset.copy()
                            merged_df[f'{col}_climate'] = value
                else:
                    # If no climate data, just use water data
                    merged_df = water_df_reset.copy()

                # Ensure Location column is preserved
                if 'Location' not in merged_df.columns:
                    merged_df['Location'] = location

                # Add data source columns
                merged_df['data_source'] = 'water_climate_combined'
                merged_df['processing_stage'] = data_type

                combined_dataframes.append(merged_df)
                print(f"   ✓ Merged data for location: {location}")

        if combined_dataframes:
            return pd.concat(combined_dataframes, ignore_index=True)
        else:
            return pd.DataFrame()

    # Handle both datasets as dataframes
    elif not isinstance(water_data, dict) and not isinstance(climate_data, dict):
        print("📊 Combining global water and climate datasets...")
        if not water_data.empty and not climate_data.empty:
            # Reset index to make timestamp a column for merging
            water_df_reset = water_data.reset_index()
            climate_df_reset = climate_data.reset_index()

            # Merge on timestamp if both have it
            if 'timestamp' in water_df_reset.columns and 'timestamp' in climate_df_reset.columns:
                merged_df = pd.merge(water_df_reset, climate_df_reset,
                                   on='timestamp', how='left',
                                   suffixes=('_water', '_climate'))
            else:
                # If no timestamp, just concatenate (less ideal)
                print("   ⚠️ No timestamp columns found, concatenating data")
                merged_df = pd.concat([water_df_reset, climate_df_reset], axis=1)

            # Add data source columns
            merged_df['data_source'] = 'water_climate_combined'
            merged_df['processing_stage'] = data_type

            print("   ✅ Successfully merged water and climate data")
            return merged_df
        elif not water_data.empty:
            # If only water data exists
            water_df_reset = water_data.reset_index()
            water_df_reset['data_source'] = 'water_only'
            water_df_reset['processing_stage'] = data_type
            return water_df_reset
        else:
            return pd.DataFrame()

    # Handle both datasets as dictionaries (location-based)
    elif isinstance(water_data, dict) and isinstance(climate_data, dict):
        print("📍 Combining location-based water and climate data...")
        combined_dataframes = []

        # Process all water locations, matching with climate where available
        for water_location, water_df in water_data.items():
            if not water_df.empty:
                # Reset index to make timestamp a column for merging
                water_df_reset = water_df.reset_index()
                
                # Ensure Location column exists
                if 'Location' not in water_df_reset.columns:
                    water_df_reset['Location'] = water_location

                # Try to find matching climate data
                climate_df = None
                for climate_location, climate_df_candidate in climate_data.items():
                    if climate_location == water_location and not climate_df_candidate.empty:
                        climate_df = climate_df_candidate
                        break

                if climate_df is not None:
                    # Reset index for climate data
                    climate_df_reset = climate_df.reset_index()

                    # Merge on timestamp if both have it
                    if 'timestamp' in water_df_reset.columns and 'timestamp' in climate_df_reset.columns:
                        merged_df = pd.merge(water_df_reset, climate_df_reset,
                                           on='timestamp', how='left',
                                           suffixes=('_water', '_climate'))
                    else:
                        # If no timestamp, add climate data as additional columns
                        print(f"   ⚠️ No timestamp columns found, adding climate data as additional columns for {water_location}")
                        merged_df = water_df_reset.copy()
                        climate_means = climate_df_reset.select_dtypes(include=[np.number]).mean()
                        for col, value in climate_means.items():
                            merged_df[f'{col}_climate'] = value
                else:
                    # No matching climate data found, just use water data
                    merged_df = water_df_reset.copy()
                    print(f"   ⚠️ No matching climate data found for location: {water_location}")

                # Ensure Location column is preserved
                if 'Location' not in merged_df.columns:
                    merged_df['Location'] = water_location

                # Add data source columns
                merged_df['data_source'] = 'water_climate_combined'
                merged_df['processing_stage'] = data_type

                combined_dataframes.append(merged_df)
                print(f"   ✓ Processed data for location: {water_location}")

        if combined_dataframes:
            return pd.concat(combined_dataframes, ignore_index=True)
        else:
            return pd.DataFrame()

    else:
        print("   ⚠️ Incompatible data structures, preserving water data with locations")
        # Fallback: just return water data with preserved locations
        if isinstance(water_data, dict):
            combined_dataframes = []
            for location, water_df in water_data.items():
                if not water_df.empty:
                    water_df_reset = water_df.reset_index()
                    if 'Location' not in water_df_reset.columns:
                        water_df_reset['Location'] = location
                    water_df_reset['data_source'] = 'water_only'
                    water_df_reset['processing_stage'] = data_type
                    combined_dataframes.append(water_df_reset)
            
            if combined_dataframes:
                return pd.concat(combined_dataframes, ignore_index=True)
        
        return pd.DataFrame()

def main():
    """
    Main function to run the data processing pipeline
    """
    print("🌊 WATER QUALITY DATA PROCESSING PIPELINE")
    print("=" * 60)
    print("This script will help you process and merge water quality and climate data.")
    print("Please ensure your CSV files are accessible from this location.")
    print("=" * 60)

    # Get file paths from user
    water_file = get_file_path("water parameters")
    climate_file = get_file_path("climatic parameters")

    # Clean and scale water + climate datasets
    print("\n" + "="*60)
    print("🌊 PROCESSING WATER PARAMETERS DATA")
    print("="*60)
    water_clean, water_scaled, water_locations = clean_and_normalize(water_file)

    if water_clean is None:
        print("❌ Failed to process water data. Exiting.")
        return

    print("\n" + "="*60)
    print("🌡️ PROCESSING CLIMATE PARAMETERS DATA")
    print("="*60)
    climate_clean, climate_scaled, climate_locations = clean_and_normalize(climate_file)

    if climate_clean is None:
        print("❌ Failed to process climate data. Exiting.")
        return

    # Merge cleaned and normalized datasets
    print("\n" + "="*60)
    print("🔗 MERGING ALL PROCESSED DATA")
    print("="*60)

    # Merge cleaned data
    cleaned_merged = merge_datasets(water_clean, climate_clean, water_locations, climate_locations, "cleaned")

    # Merge normalized data
    normalized_merged = merge_datasets(water_scaled, climate_scaled, water_locations, climate_locations, "normalized")

    # Combine cleaned and normalized data into final dataset
    print("\n📊 Creating final combined dataset...")
    final_combined_data = pd.DataFrame()

    if not cleaned_merged.empty and not normalized_merged.empty:
        # Add processing type identifier
        cleaned_merged['data_type'] = 'cleaned'
        normalized_merged['data_type'] = 'normalized'

        # Combine both datasets
        final_combined_data = pd.concat([cleaned_merged, normalized_merged],
                                       ignore_index=True)

        print(f"   ✅ Final combined dataset created with {len(final_combined_data)} rows")
        print(f"   📋 Includes both cleaned and normalized data")

    elif not cleaned_merged.empty:
        cleaned_merged['data_type'] = 'cleaned_only'
        final_combined_data = cleaned_merged
        print(f"   ✅ Final dataset created with cleaned data only: {len(final_combined_data)} rows")

    elif not normalized_merged.empty:
        normalized_merged['data_type'] = 'normalized_only'
        final_combined_data = normalized_merged
        print(f"   ✅ Final dataset created with normalized data only: {len(final_combined_data)} rows")

    else:
        print("   ❌ No data available for final combination")
        return

    # Ensure Location column is present and properly formatted
    if 'Location' in final_combined_data.columns:
        print(f"📍 Location column preserved with {final_combined_data['Location'].nunique()} unique locations")
        print(f"   Locations: {sorted(final_combined_data['Location'].unique())}")
    else:
        print("⚠️ No Location column found in final data - this may indicate an issue with data processing")

    # Create sliding windows metadata and add to final dataset
    print("\n" + "="*60)
    print("🪟 TASK 7: PREPARING TIME-SERIES METADATA")
    print("="*60)

    if not final_combined_data.empty:
        # Add sliding window preparation metadata
        print("🪟 Adding time-series preparation metadata...")

        # Sort by timestamp for proper sequence ordering
        if 'timestamp' in final_combined_data.columns:
            final_combined_data = final_combined_data.sort_values(['Location', 'timestamp'] if 'Location' in final_combined_data.columns else 'timestamp')

        # Add sequence information for sliding windows
        final_combined_data['sequence_id'] = range(len(final_combined_data))
        final_combined_data['sliding_window_ready'] = True
        final_combined_data['window_size_recommended'] = 30
        final_combined_data['total_sequences'] = len(final_combined_data)

        print(f"   ✅ Added time-series metadata for {len(final_combined_data)} data points")

    # TASK 6: Ensure all numerical scaling is documented
    print("\n" + "="*60)
    print("📏 TASK 6: SCALING DOCUMENTATION")
    print("="*60)

    if not final_combined_data.empty:
        # Add scaling information
        numeric_columns = final_combined_data.select_dtypes(include=[np.number]).columns
        final_combined_data['numeric_columns_count'] = len(numeric_columns)
        final_combined_data['scaling_method'] = 'MinMaxScaler_0_to_1'