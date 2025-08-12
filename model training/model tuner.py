import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json

# Load the data
X_train = np.load('data/processed/X_train.npy', allow_pickle=True)
X_test = np.load('data/processed/X_test.npy', allow_pickle=True)
y_train = np.load('data/processed/y_train.npy', allow_pickle=True)
y_test = np.load('data/processed/y_test.npy', allow_pickle=True)

# Ensure data is float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

print(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Data shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")

def check_existing_files(model_name):
    """Check if model files already exist and ask user what to do"""
    model_file = f'models/{model_name}_model_tuned.h5'
    params_file = f'models/{model_name}_best_params.json'
    
    model_exists = os.path.exists(model_file)
    params_exists = os.path.exists(params_file)
    
    if model_exists or params_exists:
        print(f"\n⚠️  EXISTING FILES DETECTED for {model_name.upper()} model:")
        if model_exists:
            print(f"   • {model_file}")
        if params_exists:
            print(f"   • {params_file}")
        
        while True:
            choice = input(f"\nWhat would you like to do?\n"
                         f"  [s] Skip {model_name} model (use existing files)\n"
                         f"  [r] Replace existing files (retrain model)\n"
                         f"  [q] Quit program\n"
                         f"Choice: ").lower().strip()
            
            if choice in ['s', 'skip']:
                return 'skip'
            elif choice in ['r', 'replace']:
                return 'replace'
            elif choice in ['q', 'quit']:
                return 'quit'
            else:
                print("Invalid choice. Please enter 's', 'r', or 'q'.")
    
    return 'new'

def load_existing_results(model_name):
    """Load existing model results if they exist"""
    try:
        with open(f'models/{model_name}_best_params.json', 'r') as f:
            data = json.load(f)
        
        # Load the model
        model = keras.models.load_model(f'models/{model_name}_model_tuned.h5')
        
        print(f"✅ Loaded existing {model_name.upper()} model results:")
        metrics = data['metrics']
        print(f"   MSE:  {metrics['mse']:.6f}")
        print(f"   RMSE: {metrics['rmse']:.6f}")
        print(f"   MAE:  {metrics['mae']:.6f}")
        print(f"   R²:   {metrics['r2']:.6f}")
        
        return {
            'model': model,
            'hyperparameters': data['hyperparameters'],
            'batch_size': data['best_batch_size'],
            'metrics': data['metrics']
        }
    except Exception as e:
        print(f"❌ Error loading existing {model_name} results: {str(e)}")
        return None

# CNN Model Builder
def build_cnn_model(hp):
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        
        # First Conv1D layer
        layers.Conv1D(
            filters=hp.Int('cnn_filters_1', min_value=32, max_value=128, step=32),
            kernel_size=hp.Int('kernel_size_1', min_value=2, max_value=5),
            activation='relu',
            padding='same'
        ),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(hp.Float('cnn_dropout_1', min_value=0.1, max_value=0.5, step=0.1)),
        
        # Second Conv1D layer (optional)
        layers.Conv1D(
            filters=hp.Int('cnn_filters_2', min_value=64, max_value=256, step=64),
            kernel_size=hp.Int('kernel_size_2', min_value=2, max_value=5),
            activation='relu',
            padding='same'
        ),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(hp.Float('cnn_dropout_2', min_value=0.1, max_value=0.5, step=0.1)),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(
            units=hp.Int('dense_units', min_value=32, max_value=128, step=32),
            activation='relu'
        ),
        layers.Dropout(hp.Float('dense_dropout', min_value=0.1, max_value=0.5, step=0.1)),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        ),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# LSTM Model Builder
def build_lstm_model(hp):
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        
        # First LSTM layer
        layers.LSTM(
            units=hp.Int('lstm_units_1', min_value=32, max_value=128, step=32),
            return_sequences=hp.Boolean('return_sequences_1'),
            dropout=hp.Float('lstm_dropout_1', min_value=0.1, max_value=0.5, step=0.1)
        ),
    ])
    
    # Add second LSTM layer if first returns sequences
    if hp.Boolean('add_second_lstm'):
        model.add(layers.LSTM(
            units=hp.Int('lstm_units_2', min_value=32, max_value=128, step=32),
            return_sequences=False,
            dropout=hp.Float('lstm_dropout_2', min_value=0.1, max_value=0.5, step=0.1)
        ))
    
    # Dense layers
    model.add(layers.Dense(
        units=hp.Int('dense_units', min_value=32, max_value=128, step=32),
        activation='relu'
    ))
    model.add(layers.Dropout(hp.Float('dense_dropout', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(layers.Dense(1))
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        ),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Hybrid Model Builder (CNN + LSTM + Attention)
def build_hybrid_model(hp):
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    
    # CNN branch
    cnn = layers.Conv1D(
        filters=hp.Int('cnn_filters', min_value=32, max_value=128, step=32),
        kernel_size=hp.Int('kernel_size', min_value=2, max_value=5),
        activation='relu',
        padding='same'
    )(inputs)
    cnn = layers.MaxPooling1D(pool_size=2)(cnn)
    cnn = layers.Dropout(hp.Float('cnn_dropout', min_value=0.1, max_value=0.5, step=0.1))(cnn)
    # Flatten CNN output to 1D
    cnn_flattened = layers.GlobalAveragePooling1D()(cnn)
    
    # LSTM branch
    lstm = layers.LSTM(
        units=hp.Int('lstm_units', min_value=32, max_value=128, step=32),
        return_sequences=True
    )(inputs)
    lstm = layers.Dropout(hp.Float('lstm_dropout', min_value=0.1, max_value=0.5, step=0.1))(lstm)
    
    # Attention mechanism (optional)
    if hp.Boolean('use_attention'):
        attention = layers.MultiHeadAttention(
            num_heads=hp.Int('num_heads', min_value=2, max_value=8, step=2),
            key_dim=hp.Int('key_dim', min_value=16, max_value=64, step=16)
        )(lstm, lstm)
        # Flatten attention output to 1D
        attention_flattened = layers.GlobalAveragePooling1D()(attention)
    else:
        # Flatten LSTM output to 1D
        attention_flattened = layers.GlobalAveragePooling1D()(lstm)
    
    # Combine branches (now both are 1D)
    combined = layers.Concatenate()([cnn_flattened, attention_flattened])
    
    # Dense layers
    x = layers.Dense(
        units=hp.Int('dense_units', min_value=32, max_value=128, step=32),
        activation='relu'
    )(combined)
    x = layers.Dropout(hp.Float('dense_dropout', min_value=0.1, max_value=0.5, step=0.1))(x)
    
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        ),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def tune_model(model_builder, model_name, max_trials=20):
    """Tune a specific model type"""
    print(f"\n{'='*50}")
    print(f"TUNING {model_name.upper()} MODEL")
    print(f"{'='*50}")
    
    # Create tuner
    tuner = kt.RandomSearch(
        model_builder,
        objective='val_loss',
        max_trials=max_trials,
        directory=f'{model_name}_tuning',
        project_name=f'{model_name}_model',
        overwrite=True
    )
    
    # Define callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Search for best hyperparameters
    batch_sizes = [16, 32, 64]
    best_val_loss = float('inf')
    best_batch_size = 32
    best_model = None
    best_hps = None
    
    for batch_size in batch_sizes:
        print(f"\nTrying batch size: {batch_size}")
        
        tuner.search(
            X_train, y_train,
            epochs=50,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            batch_size=batch_size,
            verbose=1
        )
        
        # Get best trial for this batch size
        current_best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        current_best_model = tuner.hypermodel.build(current_best_hps)
        
        # Quick validation to find best batch size
        history = current_best_model.fit(
            X_train, y_train,
            epochs=20,
            validation_split=0.2,
            batch_size=batch_size,
            verbose=0
        )
        
        val_loss = min(history.history['val_loss'])
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_batch_size = batch_size
            best_model = current_best_model
            best_hps = current_best_hps
    
    print(f"\nBest batch size for {model_name}: {best_batch_size}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Train the best model with more epochs
    print(f"\nTraining final {model_name} model...")
    final_model = tuner.hypermodel.build(best_hps)
    
    history = final_model.fit(
        X_train, y_train,
        epochs=100,
        validation_split=0.2,
        batch_size=best_batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate the model
    y_pred = final_model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name.upper()} MODEL RESULTS:")
    print(f"Test MSE: {mse:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Test R²: {r2:.6f}")
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    final_model.save(f'models/{model_name}_model_tuned.h5')
    
    # Save hyperparameters
    with open(f'models/{model_name}_best_params.json', 'w') as f:
        json.dump({
            'hyperparameters': best_hps.values,
            'best_batch_size': best_batch_size,
            'metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            }
        }, f, indent=2)
    
    print(f"✅ {model_name.upper()} model saved successfully!")
    
    return {
        'model': final_model,
        'hyperparameters': best_hps.values,
        'batch_size': best_batch_size,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    }

def handle_model_training(model_name, model_builder, max_trials):
    """Handle training or loading of a specific model"""
    file_status = check_existing_files(model_name)
    
    if file_status == 'quit':
        print("Program terminated by user.")
        return None
    elif file_status == 'skip':
        result = load_existing_results(model_name)
        if result is None:
            print(f"⚠️  Failed to load existing {model_name} results. Training new model...")
            return tune_model(model_builder, model_name, max_trials)
        return result
    else:  # 'new' or 'replace'
        if file_status == 'replace':
            print(f"🗑️  Replacing existing {model_name} model files...")
        return tune_model(model_builder, model_name, max_trials)

def check_final_comparison():
    """Check if final comparison already exists"""
    comparison_file = 'models/model_comparison.json'
    if os.path.exists(comparison_file):
        print(f"\n⚠️  EXISTING FILE DETECTED: {comparison_file}")
        while True:
            choice = input(f"What would you like to do?\n"
                         f"  [u] Update comparison with current results\n"
                         f"  [k] Keep existing comparison file\n"
                         f"Choice: ").lower().strip()
            
            if choice in ['u', 'update']:
                return 'update'
            elif choice in ['k', 'keep']:
                return 'keep'
            else:
                print("Invalid choice. Please enter 'u' or 'k'.")
    return 'new'

# Main execution
if __name__ == "__main__":
    print("🚀 Starting comprehensive model tuning...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Dictionary to store results
    results = {}
    
    # Model configurations
    model_configs = [
        ('cnn', build_cnn_model, 15),
        ('lstm', build_lstm_model, 15),
        ('hybrid', build_hybrid_model, 20)
    ]
    
    # Process each model
    for model_name, model_builder, max_trials in model_configs:
        print(f"\n{'🔄' if check_existing_files(model_name) != 'new' else '🆕'} Processing {model_name.upper()} model...")
        
        result = handle_model_training(model_name, model_builder, max_trials)
        
        if result is None:  # User chose to quit
            print("Exiting program...")
            exit(0)
        
        results[model_name] = result
        print(f"✅ {model_name.upper()} model processing complete!")
    
    # Handle final comparison
    comparison_status = check_final_comparison()
    
    if comparison_status == 'keep':
        print("📊 Keeping existing comparison file. Loading results...")
        try:
            with open('models/model_comparison.json', 'r') as f:
                existing_comparison = json.load(f)
            print(f"🏆 Previous winner: {existing_comparison['best_model'].upper()}")
            print("Program completed using existing comparison.")
        except Exception as e:
            print(f"Error loading existing comparison: {str(e)}")
            print("Generating new comparison instead...")
            comparison_status = 'update'
    
    if comparison_status in ['new', 'update']:
        # Compare results
        print(f"\n{'='*60}")
        print("FINAL COMPARISON OF ALL MODELS")
        print(f"{'='*60}")
        
        best_model_name = min(results.keys(), key=lambda x: results[x]['metrics']['mse'])
        
        for model_name, result in results.items():
            metrics = result['metrics']
            print(f"\n{model_name.upper()} MODEL:")
            print(f"  MSE:  {metrics['mse']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  MAE:  {metrics['mae']:.6f}")
            print(f"  R²:   {metrics['r2']:.6f}")
            if model_name == best_model_name:
                print(f"  ⭐ BEST MODEL (lowest MSE)")
        
        print(f"\n🏆 WINNER: {best_model_name.upper()} MODEL")
        print(f"Best MSE: {results[best_model_name]['metrics']['mse']:.6f}")
        
        # Save comparison results
        serializable_results = {}
        for name, result in results.items():
            serializable_results[name] = {
                'hyperparameters': result['hyperparameters'],
                'batch_size': result['batch_size'],
                'metrics': result['metrics']
            }
        serializable_results['best_model'] = best_model_name
        
        with open('models/model_comparison.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        action = "Updated" if comparison_status == 'update' else "Created"
        print(f"📊 {action} comparison file: models/model_comparison.json")
    
    print(f"\n🎉 ALL TASKS COMPLETED!")
    print(f"📁 All models and results saved in 'models/' directory")
    
    # Summary of files created/updated
    print(f"\n📋 FILES SUMMARY:")
    for model_name in results.keys():
        print(f"  ✅ {model_name}_model_tuned.h5")
        print(f"  ✅ {model_name}_best_params.json")
    print(f"  ✅ model_comparison.json")