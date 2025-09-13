#!/usr/bin/env python3
"""
Sample Dataset Creator for Construction Cost Prediction Training
This script helps you create a properly formatted dataset from your existing data.
"""

import json
import os
import random
from typing import List, Dict, Any

def create_sample_price_quotes(num_projects: int = 100) -> Dict[str, Any]:
    """
    Create a sample price quotes dataset for training.
    Replace this with your actual data.
    """
    
    # Sample project types and their characteristics
    project_types = [
        {
            "type": "residential_single",
            "size_range": (1200, 3500),
            "quality_levels": ["economy", "standard", "premium"],
            "cost_per_sqft": {"economy": 80, "standard": 120, "premium": 180},
            "material_ratio": 0.6,
            "labor_ratio": 0.3,
            "equipment_ratio": 0.1
        },
        {
            "type": "residential_multi",
            "size_range": (2000, 8000),
            "quality_levels": ["standard", "premium"],
            "cost_per_sqft": {"standard": 100, "premium": 150},
            "material_ratio": 0.55,
            "labor_ratio": 0.35,
            "equipment_ratio": 0.1
        },
        {
            "type": "commercial_office",
            "size_range": (5000, 50000),
            "quality_levels": ["standard", "premium"],
            "cost_per_sqft": {"standard": 140, "premium": 220},
            "material_ratio": 0.5,
            "labor_ratio": 0.4,
            "equipment_ratio": 0.1
        },
        {
            "type": "commercial_retail",
            "size_range": (3000, 20000),
            "quality_levels": ["economy", "standard", "premium"],
            "cost_per_sqft": {"economy": 90, "standard": 130, "premium": 200},
            "material_ratio": 0.6,
            "labor_ratio": 0.3,
            "equipment_ratio": 0.1
        },
        {
            "type": "industrial_warehouse",
            "size_range": (10000, 100000),
            "quality_levels": ["economy", "standard"],
            "cost_per_sqft": {"economy": 60, "standard": 90},
            "material_ratio": 0.7,
            "labor_ratio": 0.2,
            "equipment_ratio": 0.1
        }
    ]
    
    # Location factors (regional cost multipliers)
    location_factors = {
        "California": 1.4,
        "New York": 1.3,
        "Texas": 0.9,
        "Florida": 1.0,
        "Illinois": 1.1,
        "Pennsylvania": 1.0,
        "Ohio": 0.9,
        "Georgia": 0.95,
        "North Carolina": 0.9,
        "Michigan": 0.95
    }
    
    projects = []
    
    for i in range(num_projects):
        # Select random project type
        project_type = random.choice(project_types)
        
        # Generate project size
        min_size, max_size = project_type["size_range"]
        project_size = random.randint(min_size, max_size)
        
        # Select quality level
        quality_level = random.choice(project_type["quality_levels"])
        
        # Calculate base cost per sqft
        base_cost_per_sqft = project_type["cost_per_sqft"][quality_level]
        
        # Apply location factor
        location = random.choice(list(location_factors.keys()))
        location_factor = location_factors[location]
        
        # Calculate total cost
        total_cost = project_size * base_cost_per_sqft * location_factor
        
        # Add some randomness (±15%)
        cost_variation = random.uniform(0.85, 1.15)
        total_cost *= cost_variation
        
        # Calculate component costs
        material_cost = total_cost * project_type["material_ratio"]
        labor_cost = total_cost * project_type["labor_ratio"]
        equipment_cost = total_cost * project_type["equipment_ratio"]
        
        # Generate year (2020-2024)
        year_built = random.randint(2020, 2024)
        
        # Create project entry
        project = {
            "image_filename": f"project_{i+1:03d}.png",
            "project_size": project_size,
            "location_factor": location_factor,
            "year_built": year_built,
            "quality_level": quality_level,
            "material_cost": round(material_cost),
            "labor_cost": round(labor_cost),
            "equipment_cost": round(equipment_cost),
            "total_cost": round(total_cost),
            "description": f"{project_type['type'].replace('_', ' ').title()} - {location}",
            "project_type": project_type["type"],
            "location": location
        }
        
        projects.append(project)
    
    return {"projects": projects}

def create_cad_pdf_placeholders(cad_files_dir: str, num_files: int = 100):
    """
    Create placeholder CAD PDF files for testing.
    In practice, replace these with your actual CAD PDF drawings.
    """
    os.makedirs(cad_files_dir, exist_ok=True)
    
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter, A3
        from reportlab.lib import colors
        from reportlab.lib.units import inch
    except ImportError:
        print("ReportLab not available. Creating simple text files instead.")
        # Fallback: create simple text files
        for i in range(num_files):
            text_content = f"""CAD Drawing - Project {i+1:03d}
Scale: 1:100
Building Type: Residential
Dimensions: 30' x 40' x 10'
Materials: Concrete, Steel, Wood
Construction Method: Traditional
Quality Level: Standard
Area: 1200 SF
"""
            with open(os.path.join(cad_files_dir, f"project_{i+1:03d}.txt"), 'w') as f:
                f.write(text_content)
        print(f"Created {num_files} placeholder CAD text files in {cad_files_dir}")
        return
    
    for i in range(num_files):
        # Create a simple placeholder PDF
        pdf_path = os.path.join(cad_files_dir, f"project_{i+1:03d}.pdf")
        c = canvas.Canvas(pdf_path, pagesize=A3)
        
        # Add title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, f"CAD Drawing - Project {i+1:03d}")
        
        # Add project information
        c.setFont("Helvetica", 12)
        c.drawString(100, 700, "Scale: 1:100")
        c.drawString(100, 680, "Building Type: Residential")
        c.drawString(100, 660, "Dimensions: 30' x 40' x 10'")
        c.drawString(100, 640, "Materials: Concrete, Steel, Wood")
        c.drawString(100, 620, "Construction Method: Traditional")
        c.drawString(100, 600, "Quality Level: Standard")
        c.drawString(100, 580, "Area: 1200 SF")
        
        # Draw a simple building outline
        c.setStrokeColor(colors.black)
        c.setLineWidth(2)
        c.rect(200, 300, 300, 200)  # Main building
        c.rect(250, 350, 200, 100)  # Interior space
        
        # Add some dimensions
        c.setFont("Helvetica", 10)
        c.drawString(200, 280, "30'")
        c.drawString(350, 250, "40'")
        
        # Add construction notes
        c.setFont("Helvetica", 10)
        c.drawString(100, 200, "Foundation: Concrete slab")
        c.drawString(100, 180, "Walls: Wood frame construction")
        c.drawString(100, 160, "Roof: Asphalt shingles")
        c.drawString(100, 140, "Windows: Double pane")
        c.drawString(100, 120, "Doors: Standard residential")
        
        c.save()
    
    print(f"Created {num_files} placeholder CAD PDF files in {cad_files_dir}")

def validate_dataset(cad_files_dir: str, quotes_file: str) -> bool:
    """
    Validate that the dataset is properly formatted.
    """
    print("Validating dataset...")
    
    # Check if quotes file exists
    if not os.path.exists(quotes_file):
        print(f"Error: Quotes file not found: {quotes_file}")
        return False
    
    # Load quotes data
    try:
        with open(quotes_file, 'r') as f:
            quotes_data = json.load(f)
    except Exception as e:
        print(f"Error loading quotes file: {e}")
        return False
    
    # Check required fields
    required_fields = [
        "image_filename", "project_size", "location_factor", "year_built",
        "quality_level", "material_cost", "labor_cost", "equipment_cost", "total_cost"
    ]
    
    projects = quotes_data.get("projects", [])
    if not projects:
        print("Error: No projects found in quotes file")
        return False
    
    print(f"Found {len(projects)} projects in quotes file")
    
    # Validate each project
    valid_projects = 0
    for i, project in enumerate(projects):
        missing_fields = [field for field in required_fields if field not in project]
        if missing_fields:
            print(f"Project {i+1} missing fields: {missing_fields}")
            continue
        
        # Check if corresponding CAD file exists (PDF or image)
        cad_file_path = os.path.join(cad_files_dir, project["image_filename"])
        if not os.path.exists(cad_file_path):
            print(f"Project {i+1}: CAD file not found: {project['image_filename']}")
            continue
        
        # Validate cost relationships
        calculated_total = project["material_cost"] + project["labor_cost"] + project["equipment_cost"]
        if abs(calculated_total - project["total_cost"]) > 1:  # Allow $1 rounding difference
            print(f"Project {i+1}: Cost mismatch - calculated: {calculated_total}, stated: {project['total_cost']}")
            continue
        
        valid_projects += 1
    
    print(f"Valid projects: {valid_projects}/{len(projects)}")
    
    if valid_projects < 10:
        print("Warning: Less than 10 valid projects. Consider adding more data.")
        return False
    
    print("Dataset validation completed successfully!")
    return True

def main():
    """
    Main function to create and validate sample dataset.
    """
    print("Construction Cost Prediction Dataset Creator")
    print("=" * 50)
    
    # Configuration
    num_projects = 100
    cad_files_dir = "cad_files"
    quotes_file = "price_quotes.json"
    
    print(f"Creating dataset with {num_projects} projects...")
    
    # Create sample price quotes
    print("1. Creating price quotes data...")
    quotes_data = create_sample_price_quotes(num_projects)
    
    with open(quotes_file, 'w') as f:
        json.dump(quotes_data, f, indent=2)
    
    print(f"   Created {quotes_file} with {len(quotes_data['projects'])} projects")
    
    # Create placeholder CAD PDF files
    print("2. Creating placeholder CAD PDF files...")
    create_cad_pdf_placeholders(cad_files_dir, num_projects)
    
    # Validate dataset
    print("3. Validating dataset...")
    if validate_dataset(cad_files_dir, quotes_file):
        print("\n✅ Dataset created successfully!")
        print(f"   CAD files: {cad_files_dir}/")
        print(f"   Price quotes: {quotes_file}")
        print("\nNext steps:")
        print("1. Replace placeholder PDFs with your actual CAD drawings")
        print("2. Update price_quotes.json with your actual project data")
        print("3. Run: python train_model.py --cad_dir ./cad_files --quotes_file ./price_quotes.json")
    else:
        print("\n❌ Dataset validation failed. Please check the errors above.")

if __name__ == "__main__":
    main()
