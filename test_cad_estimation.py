#!/usr/bin/env python3
# source venv/bin/activate && python test_cad_estimation.py
"""
Test script for CAD cost estimation functionality
This script tests the real PDF processing workflow: PDF -> Image -> OCR -> AI Analysis
"""

import os
import sys
import django
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cadProcessor.settings')
django.setup()

from globalapp.task import start_pdf_processing

def test_complete_workflow():
    """Test the complete workflow: start_pdf_processing -> process_cad_drawing -> get_construction_jobs"""
    print("\n🔄 COMPLETE WORKFLOW TEST")
    print("=" * 60)
    
    pdf_path = "/media/logan/Work/AutoCadAI/globalapp/media/pdfs/test.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    print(f"📄 Testing complete workflow with: {pdf_path}")
    
    # Create output paths (same as in upload_pdf view)
    excel_path = pdf_path.replace(".pdf", ".xlsx")
    output_pdf_path = pdf_path.replace(".pdf", "_cost.pdf")
    
    print(f"📊 Excel output: {excel_path}")
    print(f"📄 PDF output: {output_pdf_path}")
    
    try:
        # Test the complete workflow starting from start_pdf_processing
        print("\n🚀 Starting complete PDF processing workflow...")
        print("Starting PDF processing...")
        
        start_pdf_processing(pdf_path, excel_path, output_pdf_path)
        
        print("✅ Complete workflow executed successfully")
        
        # Check if output files were created
        if os.path.exists(excel_path):
            print(f"✅ Excel file created: {excel_path}")
        else:
            print(f"⚠️  Excel file not created: {excel_path}")
            
        if os.path.exists(output_pdf_path):
            print(f"✅ Cost PDF created: {output_pdf_path}")
        else:
            print(f"⚠️  Cost PDF not created: {output_pdf_path}")
            
    except Exception as e:
        print(f"❌ Complete workflow failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("🚀 Starting CAD Cost Estimation Tests")
    print("=" * 60)
    
    # Test the complete workflow (start_pdf_processing -> process_cad_drawing -> get_construction_jobs)
    test_complete_workflow()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 60)
    print("\n📝 Notes:")
    print("- Complete workflow: upload_pdf -> start_pdf_processing -> process_pdf_page -> process_cad_drawing -> get_construction_jobs")
    print("- If text extraction returns 0 characters, check PDF format")
    print("- If AI analysis fails, check OpenAI API key")
    print("- Results are returned in JSON format for easy parsing")

if __name__ == "__main__":
    main()
