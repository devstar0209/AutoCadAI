# Set up
## Create a Virtual Environment
sudo apt install python3.10-venv -y
python3 -m venv venv

## Activate the Virtual Environment
source venv/bin/activate

## Install Dependencies
pip install -r requirements.txt

## Install tesseract
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev

## Test
python test_cad_estimation.py

-- or --

source venv/bin/activate && python test_cad_estimation.py