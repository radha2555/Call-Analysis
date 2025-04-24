import os
import re
import glob
import json
import logging
import pymongo
import certifi
import threading
from datetime import datetime
from filelock import FileLock
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pymongo.errors import ConnectionFailure
from sentence_transformers import SentenceTransformer
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 🔹 Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ GROQ_API_KEY is missing! Ensure it's set in the .env file.")

# 🔹 MongoDB Saving Configuration
MONGO_URI= os.getenv("MONGO_URI")
MONGO_DB = "CallAnalysis"
MONGO_COLLECTION = "phone_records"

# 🔹 Folder & Log Configurations
DOWNLOAD_FOLDER = os.path.join(os.getcwd(), "data")  # Folder to save files
LLM_PROCESSED_LOG = "processed_llm_files.txt"  # Track processed files

# Ensure folders and log files exist
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
if not os.path.exists(LLM_PROCESSED_LOG):
    with open(LLM_PROCESSED_LOG, "w") as f:
        pass  # Create an empty file
    
# 🔹 Setup Logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress specific library logs
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# 🔹 Initialize SentenceTransformer for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L12-v2")

# ---- Helper Functions ----

def fetch_transcriptions_from_mongodb():
    """Fetch all transcriptions from MongoDB."""
    try:
        client = pymongo.MongoClient(
            MONGO_URI,
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=5000
        )
        db = client[MONGO_DB]
        collection = db[MONGO_COLLECTION]

        # Only fetch documents that have a transcription field
        transcriptions = list(collection.find(
            {"transcription": {"$exists": True}}, 
            {"_id": 0, "filename": 1, "transcription": 1}
        ))
        return transcriptions

    except Exception as e:
        print(f"⚠️ Error fetching transcriptions from MongoDB: {str(e)}")
        return []
    finally:
        if 'client' in locals():
            client.close()

def save_processed_llm_file(filename):
    """Save only unique filenames to the LLM processed log file."""
    processed_files = load_processed_files(log_file=LLM_PROCESSED_LOG)
    
    if filename not in processed_files:  # Prevent duplicates
        with open(LLM_PROCESSED_LOG, "a") as f:
            f.write(filename + "\n")

def extract_text_from_folder(folder_path):
    texts = []
    files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.txt')]

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return "\n".join(texts), files

def extract_json_from_response(response_text):
    """
    Extracts JSON from a response string that may include explanations or formatting issues.
    """
    try:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            json_str = match.group(0)  # Extract JSON portion
            return json.loads(json_str)  # Convert to dictionary
        else:
            logger.error("❌ No valid JSON found in response.")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error decoding JSON: {e}")
        return None

def analyze_text_with_groq(text):
    """Analyze text using Groq's Llama-3 model for multiple tasks."""
    try:
        model = "llama3-70b-8192"

        groq_chat = ChatGroq(groq_api_key=api_key, model_name=model)

        prompt = f"""
        Perform the following analysis on the given transcription text:
        - Summarize the text content in a clear and concise manner.
        - Extract entities (name, location, phone number, age, and date of birth).
        - Extract any mention of a **reschedule time** (such as "tomorrow at 7", "next Tuesday", "after 9:30", etc.).
        - Analyze the sentiment (positive, neutral, or negative).
        - Identify the customer interest (Interested, Not sure, or Not Interested).

        Provide your response in **valid JSON format** without extra text.
        Example:
        {{
            "summary": "...",
            "entities": {{
                "name": "...",
                "location": "...",
                "phone_number": "...",
                "age": "...",
                "dob": "...",
                "call_reschedule_time": "..."  # Example: "tomorrow at 7"
            }},
            "sentiment": "...",
            "customer_interest": "..."
        }}

        Text: 
        {text}
        """

        response = groq_chat.invoke(prompt)

        if hasattr(response, "content"):
            response_text = response.content  
        else:
            logger.error(f"❌ Unexpected Groq API response: {response}")
            return {"error": "Invalid API response"}

        # Extract and parse JSON safely
        analysis_result = extract_json_from_response(response_text)

        if not analysis_result:
            return {"error": "Invalid JSON response from Groq"}

        return analysis_result

    except Exception as e:
        logger.error(f"❌ Error during LLM analysis: {str(e)}")
        return {"error": str(e)}

def store_results_in_mongodb(results):
    """Store LLM results in MongoDB."""
    try:
        with pymongo.MongoClient(
            MONGO_URI, tls=True, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000
        ) as client:

            db = client[MONGO_DB]
            collection = db[MONGO_COLLECTION]  # ✅ Using the same collection as transcriptions

            client.admin.command("ping")  # Check MongoDB connection
            logger.info("✅ Connected to MongoDB successfully!")

            for file_name, result in results.items():
                if "error" in result:
                    logger.warning(f"⚠️ Skipping {file_name} due to error: {result['error']}")
                    continue

                update_data = {
                    "sentiment": result.get("sentiment"),
                    "customer_interest": result.get("customer_interest"),
                    "summary": result.get("summary"),
                    "entities": result.get("entities"),
                    "date_processed": datetime.today().strftime("%d-%m-%Y")
                }

                update_result = collection.update_one(
                    {"filename": file_name},
                    {"$set": update_data},
                    upsert=True
                )

                if update_result.matched_count > 0:
                    logger.info(f"✅ LLM results updated for {file_name}.")
                else:
                    logger.warning(f"⚠️ No matching record found for {file_name}. New entry stored.")

    except Exception as e:
        logger.error(f"❌ Error storing LLM results in MongoDB: {str(e)}")

def load_processed_files(log_file):
    """Load the list of already processed files from a log file."""
    processed_files = set()
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            for line in f:
                processed_files.add(line.strip())  # Remove any extra whitespace
    return processed_files

def process_folder(): 
    """Fetch transcriptions from the 'data' folder and process them using LLM."""
    all_results = {}
    processed_files = load_processed_files("processed_llm_files.txt")

    transcriptions = fetch_transcriptions_from_mongodb()
    if not transcriptions:
        logger.info("⏭️ No transcriptions found in MongoDB.")
        return

    llm_success = True

    def process_file(transcription):
        nonlocal llm_success
        filename = transcription["filename"]
        
        if filename in processed_files or "transcription" not in transcription:
            if "transcription" not in transcription:
                logger.warning(f"⚠️ Skipping {filename} - no transcription found")
            return

        logger.info(f"🔄 Processing file: {filename}")  
        text = transcription["transcription"]
        analysis_result = analyze_text_with_groq(text)

        if analysis_result:
            all_results[filename] = analysis_result
            save_processed_llm_file(filename)
        else:
            llm_success = False

    # Correct threading implementation (outside process_file)
    threads = []
    for transcription in transcriptions:
        t = threading.Thread(target=process_file, args=(transcription,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if all_results:
        store_results_in_mongodb(all_results)

    if llm_success:
        logger.info("✅ All files processed successfully!")
    else:
        logger.warning("⚠️ Some files failed to process.")

    return all_results

if __name__ == "__main__":
    process_folder()