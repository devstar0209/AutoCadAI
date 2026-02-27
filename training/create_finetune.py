import os

import openai
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()


OUTPUT_DATASET = "./data/hvac_finetune.jsonl"
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def fine_tune_model(dataset_file):
    print("Uploading dataset...")
    prep_file = client.files.create(file=open(dataset_file, "rb"), purpose='fine-tune')
    dataset_id = prep_file.id
    print(f"✅ Dataset uploaded. File ID: {dataset_id}")

    print("Creating fine-tune job...")
    try:
        fine_tune = client.fine_tuning.jobs.create(training_file=dataset_id, model="ft:gpt-4o-2024-08-06:global-precisional-services-llc::DDjkPdOP")
        print(f"🚀 Fine-tune started. ID: {fine_tune.id}")
    except openai.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    except openai.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
    except openai.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)
    
    return fine_tune.id


fine_tune_model(OUTPUT_DATASET)