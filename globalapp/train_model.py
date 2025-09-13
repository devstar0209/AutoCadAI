#!/usr/bin/env python3
"""
Training script for Construction Cost Prediction Model
Usage: python train_model.py --cad_dir /path/to/cad/images --quotes_file /path/to/quotes.json --output_dir models/
"""

import argparse
import os
import sys
from ml_training import ConstructionCostPredictor
import logging

def main():
    parser = argparse.ArgumentParser(description='Train Construction Cost Prediction Model')
    parser.add_argument('--cad_dir', required=True, help='Directory containing CAD drawing files (PDF or images)')
    parser.add_argument('--quotes_file', required=True, help='JSON file containing price quotes')
    parser.add_argument('--output_dir', default='models', help='Directory to save trained models')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not os.path.exists(args.cad_dir):
        logger.error(f"CAD directory not found: {args.cad_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.quotes_file):
        logger.error(f"Quotes file not found: {args.quotes_file}")
        sys.exit(1)
    
    # Check for CAD files in directory
    cad_files = []
    for file in os.listdir(args.cad_dir):
        if file.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            cad_files.append(file)
    
    if not cad_files:
        logger.error(f"No CAD files found in {args.cad_dir}. Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP")
        sys.exit(1)
    
    logger.info(f"Found {len(cad_files)} CAD files in {args.cad_dir}")
    
    # Initialize predictor
    predictor = ConstructionCostPredictor()
    
    try:
        # Prepare training data
        logger.info("Preparing training data from CAD files...")
        X, y = predictor.prepare_training_data(args.cad_dir, args.quotes_file)
        
        if len(X) == 0:
            logger.error("No training data found. Please check your CAD files and quotes file.")
            sys.exit(1)
        
        # Train models
        logger.info("Training models...")
        predictor.train_models(X, y, test_size=args.test_size)
        
        # Save models
        logger.info(f"Saving models to {args.output_dir}...")
        predictor.save_models(args.output_dir)
        
        logger.info("Training completed successfully!")
        logger.info(f"Models saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
