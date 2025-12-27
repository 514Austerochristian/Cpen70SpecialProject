# dashboard/models/comparison.py
"""
Model comparison functions
"""

import json
from pathlib import Path
import streamlit as st
from utils.paths import MODEL_COMPARISON_PATHS

def load_model_comparison(file_paths=None):
    """Load model comparison data from JSON file"""
    if file_paths is None:
        file_paths = MODEL_COMPARISON_PATHS
    
    for file_path in file_paths:
        try:
            path = Path(file_path)
            
            if path.exists():
                with open(path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                if st.checkbox("🔍 Debug: Show JSON Structure", key="debug_json"):
                    st.write("**JSON Structure:**")
                    st.json(data)
                    st.write("**Root level keys:**", list(data.keys()))
                    
                    if 'models' in data:
                        st.write("**Models available:**", list(data['models'].keys()))
                        for model_name, model_data in data['models'].items():
                            if isinstance(model_data, dict):
                                st.write(f"**{model_name} metrics:**", list(model_data.keys()))
                
                return data
                
        except json.JSONDecodeError as e:
            st.error(f"❌ Error parsing JSON file at {file_path}: {e}")
            continue
        except Exception as e:
            if file_path == file_paths[0]:
                st.warning(f"⚠️ Could not load from primary path: {e}")
            continue
    
    st.warning(f"⚠️ Model comparison file not found in any expected locations:")
    for path in file_paths:
        st.write(f"   • {path}")
    return None