import os
import json
import pandas as pd
import numpy as np
import cv2
import pytesseract
from pdf2image import convert_from_path
import PyPDF2
import fitz  # PyMuPDF for advanced PDF processing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import re
from typing import Dict, List, Tuple, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConstructionCostPredictor:
    """
    Machine Learning model for construction cost estimation based on CAD drawings and historical price data.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
    def extract_features_from_cad(self, file_path: str) -> Dict[str, Any]:
        """
        Extract features from CAD drawing (PDF or image) for ML model.
        """
        features = {}
        
        try:
            if file_path.lower().endswith('.pdf'):
                # Handle PDF files
                text, images = self._extract_from_pdf(file_path)
                features.update(self._extract_numerical_features(text))
                features.update(self._extract_categorical_features(text))
                
                # Extract features from PDF images if available
                if images:
                    for img in images:
                        img_features = self._extract_image_features(img)
                        # Average image features if multiple pages
                        for key, value in img_features.items():
                            if key in features:
                                features[key] = (features[key] + value) / 2
                            else:
                                features[key] = value
            else:
                # Handle image files (PNG, JPG, etc.)
                img = cv2.imread(file_path)
                if img is None:
                    logger.warning(f"Could not load image: {file_path}")
                    return features
                    
                # Extract text using OCR
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray)
                
                # Extract numerical features
                features.update(self._extract_numerical_features(text))
                features.update(self._extract_categorical_features(text))
                features.update(self._extract_image_features(img))
            
        except Exception as e:
            logger.error(f"Error extracting features from {file_path}: {e}")
            
        return features
    
    def _extract_from_pdf(self, pdf_path: str) -> Tuple[str, List[np.ndarray]]:
        """
        Extract text and images from PDF file.
        Returns tuple of (combined_text, list_of_images)
        """
        combined_text = ""
        images = []
        
        try:
            # Method 1: Extract text directly from PDF
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    combined_text += text + "\n"
            
            # Method 2: Use PyMuPDF for better text extraction
            try:
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    combined_text += text + "\n"
                doc.close()
            except Exception as e:
                logger.warning(f"PyMuPDF extraction failed: {e}")
            
            # Method 3: Convert PDF to images and extract text via OCR
            try:
                pdf_images = convert_from_path(pdf_path, dpi=300, poppler_path="/usr/bin")
                for img in pdf_images:
                    # Convert PIL image to OpenCV format
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    images.append(img_cv)
                    
                    # Extract text from image
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    ocr_text = pytesseract.image_to_string(gray, config='--psm 6')
                    combined_text += ocr_text + "\n"
            except Exception as e:
                logger.warning(f"PDF to image conversion failed: {e}")
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
        
        return combined_text, images
    
    def _extract_numerical_features(self, text: str) -> Dict[str, float]:
        """Extract numerical features from CAD text."""
        features = {}
        
        # Enhanced dimension extraction patterns for CAD drawings
        dimension_patterns = {
            'length': [
                r'length[:\s]*(\d+(?:\.\d+)?)\s*(?:ft|feet|m|meter|\')',
                r'(\d+(?:\.\d+)?)\s*(?:ft|feet|m|meter|\')\s*(?:long|length)',
                r'l[:\s]*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*\'[-\s]*(\d+(?:\.\d+)?)\s*"'  # feet-inches format
            ],
            'width': [
                r'width[:\s]*(\d+(?:\.\d+)?)\s*(?:ft|feet|m|meter|\')',
                r'(\d+(?:\.\d+)?)\s*(?:ft|feet|m|meter|\')\s*(?:wide|width)',
                r'w[:\s]*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*\'[-\s]*(\d+(?:\.\d+)?)\s*"'  # feet-inches format
            ],
            'height': [
                r'height[:\s]*(\d+(?:\.\d+)?)\s*(?:ft|feet|m|meter|\')',
                r'(\d+(?:\.\d+)?)\s*(?:ft|feet|m|meter|\')\s*(?:high|height|tall)',
                r'h[:\s]*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*\'[-\s]*(\d+(?:\.\d+)?)\s*"'  # feet-inches format
            ],
            'thickness': [
                r'thickness[:\s]*(\d+(?:\.\d+)?)\s*(?:in|inch|cm|mm)',
                r'(\d+(?:\.\d+)?)\s*(?:in|inch|cm|mm)\s*(?:thick|thickness)',
                r't[:\s]*(\d+(?:\.\d+)?)'
            ],
            'diameter': [
                r'diameter[:\s]*(\d+(?:\.\d+)?)\s*(?:in|inch|cm|mm|ft|feet)',
                r'(\d+(?:\.\d+)?)\s*(?:in|inch|cm|mm|ft|feet)\s*(?:diameter|dia)',
                r'd[:\s]*(\d+(?:\.\d+)?)'
            ],
            'area': [
                r'area[:\s]*(\d+(?:\.\d+)?)\s*(?:sf|sq\.?ft|sq\.?feet|m2|sq\.?m)',
                r'(\d+(?:\.\d+)?)\s*(?:sf|sq\.?ft|sq\.?feet|m2|sq\.?m)',
                r'(\d+(?:\.\d+)?)\s*(?:square\s*feet|square\s*meters)'
            ],
            'volume': [
                r'volume[:\s]*(\d+(?:\.\d+)?)\s*(?:cf|cubic\s*feet|m3|cubic\s*meters)',
                r'(\d+(?:\.\d+)?)\s*(?:cf|cubic\s*feet|m3|cubic\s*meters)'
            ]
        }
        
        for feature, patterns in dimension_patterns.items():
            value = 0.0
            for pattern in patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    if isinstance(matches[0], tuple):
                        # Handle feet-inches format (e.g., "12'-6"")
                        feet = float(matches[0][0])
                        inches = float(matches[0][1]) if len(matches[0]) > 1 else 0
                        value = feet + (inches / 12.0)
                    else:
                        value = float(matches[0])
                    break
            features[feature] = value
        
        # Enhanced construction element counting
        element_counts = {
            'concrete_count': len(re.findall(r'concrete|cement|slab|foundation|footing|beam|column', text.lower())),
            'steel_count': len(re.findall(r'steel|rebar|rebar|structural\s*steel|beam|column|girder', text.lower())),
            'wall_count': len(re.findall(r'wall|partition|exterior\s*wall|interior\s*wall', text.lower())),
            'door_count': len(re.findall(r'door|entry|exit|opening', text.lower())),
            'window_count': len(re.findall(r'window|opening|glazing|fenestration', text.lower())),
            'floor_count': len(re.findall(r'floor|level|story|storey|deck', text.lower())),
            'roof_count': len(re.findall(r'roof|ceiling|roofing|truss', text.lower())),
            'electrical_count': len(re.findall(r'electrical|power|lighting|outlet|switch', text.lower())),
            'plumbing_count': len(re.findall(r'plumbing|water|sewer|drain|fixture', text.lower())),
            'hvac_count': len(re.findall(r'hvac|heating|cooling|ventilation|air\s*conditioning', text.lower()))
        }
        features.update(element_counts)
        
        # Extract scale information
        scale_patterns = [
            r'scale[:\s]*1[:\s]*(\d+)',
            r'(\d+)[:\s]*1\s*scale',
            r'1[:\s]*(\d+)\s*scale'
        ]
        scale = 0.0
        for pattern in scale_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                scale = float(matches[0])
                break
        features['drawing_scale'] = scale
        
        # Extract project area from text
        area_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:sf|sq\.?ft|sq\.?feet|square\s*feet)',
            r'area[:\s]*(\d+(?:\.\d+)?)',
            r'gross\s*area[:\s]*(\d+(?:\.\d+)?)',
            r'building\s*area[:\s]*(\d+(?:\.\d+)?)'
        ]
        project_area = 0.0
        for pattern in area_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                project_area = float(matches[0])
                break
        features['project_area'] = project_area
        
        # Calculate derived features
        if features['length'] > 0 and features['width'] > 0:
            features['area_calculated'] = features['length'] * features['width']
        else:
            features['area_calculated'] = 0.0
            
        if features['area_calculated'] > 0 and features['height'] > 0:
            features['volume_calculated'] = features['area_calculated'] * features['height']
        else:
            features['volume_calculated'] = 0.0
        
        # Use project area if available and larger than calculated area
        if features['project_area'] > features['area_calculated']:
            features['area_calculated'] = features['project_area']
        
        return features
    
    def _extract_categorical_features(self, text: str) -> Dict[str, str]:
        """Extract categorical features from CAD text."""
        features = {}
        
        # Building type detection
        building_types = ['residential', 'commercial', 'industrial', 'institutional']
        for btype in building_types:
            if btype in text.lower():
                features['building_type'] = btype
                break
        else:
            features['building_type'] = 'unknown'
        
        # Construction method
        if 'precast' in text.lower():
            features['construction_method'] = 'precast'
        elif 'cast_in_place' in text.lower() or 'cast in place' in text.lower():
            features['construction_method'] = 'cast_in_place'
        elif 'steel_frame' in text.lower():
            features['construction_method'] = 'steel_frame'
        else:
            features['construction_method'] = 'traditional'
        
        # Complexity indicators
        complexity_indicators = ['curved', 'sloped', 'multi_level', 'basement', 'attic']
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in text.lower())
        features['complexity_level'] = 'high' if complexity_score > 2 else 'medium' if complexity_score > 0 else 'low'
        
        return features
    
    def _extract_image_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extract features from image properties."""
        features = {}
        
        # Basic image properties
        features['image_width'] = float(img.shape[1])
        features['image_height'] = float(img.shape[0])
        features['image_area'] = float(img.shape[0] * img.shape[1])
        
        # Edge density (indicates drawing complexity)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = float(np.sum(edges > 0) / (img.shape[0] * img.shape[1]))
        
        # Color complexity
        features['color_variance'] = float(np.var(img))
        
        return features
    
    def prepare_training_data(self, cad_files_dir: str, price_quotes_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data from CAD files (PDF or images) and price quotes.
        
        Args:
            cad_files_dir: Directory containing CAD drawing files (PDF or images)
            price_quotes_file: JSON/CSV file with price quotes data
            
        Returns:
            X: Feature matrix
            y: Target variables (costs)
        """
        logger.info("Preparing training data from CAD files...")
        
        # Load price quotes
        if price_quotes_file.endswith('.json'):
            with open(price_quotes_file, 'r') as f:
                quotes_data = json.load(f)
        else:
            quotes_data = pd.read_csv(price_quotes_file).to_dict('records')
        
        # Extract features from CAD files
        all_features = []
        all_targets = []
        
        for quote in quotes_data:
            # Get corresponding CAD file (PDF or image)
            cad_filename = quote.get('image_filename', '')
            if not cad_filename:
                logger.warning(f"No filename specified for quote: {quote}")
                continue
                
            cad_file_path = os.path.join(cad_files_dir, cad_filename)
            if not os.path.exists(cad_file_path):
                logger.warning(f"CAD file not found: {cad_file_path}")
                continue
            
            logger.info(f"Processing CAD file: {cad_filename}")
            
            # Extract features
            features = self.extract_features_from_cad(cad_file_path)
            if not features:
                logger.warning(f"No features extracted from: {cad_filename}")
                continue
            
            # Add quote-specific features
            features.update({
                'project_size': quote.get('project_size', 0),
                'location_factor': quote.get('location_factor', 1.0),
                'year_built': quote.get('year_built', 2024),
                'quality_level': quote.get('quality_level', 'standard')
            })
            
            all_features.append(features)
            
            # Target variables
            targets = {
                'material_cost': quote.get('material_cost', 0),
                'labor_cost': quote.get('labor_cost', 0),
                'equipment_cost': quote.get('equipment_cost', 0),
                'total_cost': quote.get('total_cost', 0)
            }
            all_targets.append(targets)
            
            logger.info(f"Successfully processed {cad_filename}")
        
        # Convert to DataFrames
        X = pd.DataFrame(all_features)
        y = pd.DataFrame(all_targets)
        
        logger.info(f"Prepared {len(X)} training samples with {len(X.columns)} features")
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2):
        """
        Train multiple models for cost prediction.
        """
        logger.info("Training models...")
        
        # Handle categorical variables
        categorical_columns = ['building_type', 'construction_method', 'complexity_level', 'quality_level']
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le
        
        # Fill missing values
        X = X.fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Train models for each cost component
        cost_components = ['material_cost', 'labor_cost', 'equipment_cost', 'total_cost']
        
        for component in cost_components:
            if component in y.columns:
                logger.info(f"Training model for {component}...")
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                self.scalers[component] = scaler
                
                # Train multiple models
                models = {
                    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                    'linear_regression': LinearRegression()
                }
                
                best_model = None
                best_score = -np.inf
                
                for name, model in models.items():
                    model.fit(X_train_scaled, y_train[component])
                    y_pred = model.predict(X_test_scaled)
                    score = r2_score(y_test[component], y_pred)
                    
                    logger.info(f"{name} R² score for {component}: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                
                self.models[component] = best_model
                logger.info(f"Best model for {component}: {best_score:.4f}")
        
        self.is_trained = True
        logger.info("Training completed!")
    
    def predict_costs(self, cad_file_path: str, additional_features: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Predict construction costs for a CAD file (PDF or image).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self.extract_features_from_cad(cad_file_path)
        
        # Add additional features
        if additional_features:
            features.update(additional_features)
        
        # Create feature vector
        feature_vector = pd.DataFrame([features])
        
        # Handle categorical variables
        for col, encoder in self.encoders.items():
            if col in feature_vector.columns:
                feature_vector[col] = encoder.transform(feature_vector[col].astype(str))
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in feature_vector.columns:
                feature_vector[col] = 0
        
        feature_vector = feature_vector[self.feature_columns].fillna(0)
        
        # Make predictions
        predictions = {}
        for component, model in self.models.items():
            scaler = self.scalers[component]
            features_scaled = scaler.transform(feature_vector)
            predictions[component] = float(model.predict(features_scaled)[0])
        
        return predictions
    
    def save_models(self, model_dir: str):
        """Save trained models and preprocessors."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for component, model in self.models.items():
            joblib.dump(model, os.path.join(model_dir, f'{component}_model.pkl'))
        
        # Save scalers
        for component, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(model_dir, f'{component}_scaler.pkl'))
        
        # Save encoders
        for col, encoder in self.encoders.items():
            joblib.dump(encoder, os.path.join(model_dir, f'{col}_encoder.pkl'))
        
        # Save feature columns
        with open(os.path.join(model_dir, 'feature_columns.json'), 'w') as f:
            json.dump(self.feature_columns, f)
        
        logger.info(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir: str):
        """Load trained models and preprocessors."""
        # Load models
        for component in ['material_cost', 'labor_cost', 'equipment_cost', 'total_cost']:
            model_path = os.path.join(model_dir, f'{component}_model.pkl')
            if os.path.exists(model_path):
                self.models[component] = joblib.load(model_path)
        
        # Load scalers
        for component in ['material_cost', 'labor_cost', 'equipment_cost', 'total_cost']:
            scaler_path = os.path.join(model_dir, f'{component}_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scalers[component] = joblib.load(scaler_path)
        
        # Load encoders
        for col in ['building_type', 'construction_method', 'complexity_level', 'quality_level']:
            encoder_path = os.path.join(model_dir, f'{col}_encoder.pkl')
            if os.path.exists(encoder_path):
                self.encoders[col] = joblib.load(encoder_path)
        
        # Load feature columns
        feature_columns_path = os.path.join(model_dir, 'feature_columns.json')
        if os.path.exists(feature_columns_path):
            with open(feature_columns_path, 'r') as f:
                self.feature_columns = json.load(f)
        
        self.is_trained = True
        logger.info(f"Models loaded from {model_dir}")

def create_sample_dataset():
    """
    Create a sample dataset structure for training.
    This function shows the expected format for your CAD and price data.
    """
    sample_data = {
        "projects": [
            {
                "image_filename": "project_001.png",
                "project_size": 2500,  # square feet
                "location_factor": 1.2,  # regional cost multiplier
                "year_built": 2024,
                "quality_level": "standard",  # standard, premium, economy
                "material_cost": 45000,
                "labor_cost": 30000,
                "equipment_cost": 8000,
                "total_cost": 83000,
                "description": "Residential single-family home"
            },
            {
                "image_filename": "project_002.png",
                "project_size": 5000,
                "location_factor": 1.0,
                "year_built": 2024,
                "quality_level": "premium",
                "material_cost": 120000,
                "labor_cost": 80000,
                "equipment_cost": 20000,
                "total_cost": 220000,
                "description": "Commercial office building"
            }
        ]
    }
    
    # Save sample dataset
    with open('sample_price_quotes.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("Sample dataset created: sample_price_quotes.json")
    print("Please format your data according to this structure.")

if __name__ == "__main__":
    # Example usage
    predictor = ConstructionCostPredictor()
    
    # Create sample dataset
    create_sample_dataset()
    
    print("\nTo train the model with your data:")
    print("1. Organize your CAD images in a directory")
    print("2. Create a JSON file with price quotes (see sample_price_quotes.json)")
    print("3. Run the training script:")
    print("   predictor = ConstructionCostPredictor()")
    print("   X, y = predictor.prepare_training_data('path/to/cad/images', 'path/to/price_quotes.json')")
    print("   predictor.train_models(X, y)")
    print("   predictor.save_models('models/')")
