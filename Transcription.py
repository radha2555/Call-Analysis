import os
import subprocess
from datetime import datetime
import pymongo
import certifi
from sentence_transformers import SentenceTransformer
from pymongo.errors import ConnectionFailure, OperationFailure
import re
import time
from groq import Groq
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
import sys

# 🔹 Configure logging to handle Unicode characters in Windows
class UnicodeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # Fallback to ASCII-only output
            msg = record.getMessage().encode('ascii', 'ignore').decode('ascii')
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription.log', encoding='utf-8'),
        UnicodeStreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 🔹 Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client with the API key
client = Groq(api_key=api_key)

MONGO_URI= os.getenv("MONGO_URI")
# ---- CONFIG ----
DOWNLOAD_FOLDER = os.path.join(os.getcwd(), "data")  # Folder to save files
TRANSCRIPTION_LOG = "transcription_log.txt"  # File to track transcribed files
EMBEDDINGS_LOG = "embeddings_log.txt"  # Log file to track embeddings

# MongoDB Configuration
MONGO_DB = "CallAnalysis"
MONGO_COLLECTION = "phone_records"
EMBEDDING_COLLECTION = "Embeddings"

# Initialize models and connections
embedding_model = None
mongo_client = None

# ---- UTILITY FUNCTIONS ----
def load_models():
    """Load ML models only once."""
    global embedding_model
    if embedding_model is None:
        logger.info("Loading embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L12-v2')
        logger.info("Embedding model loaded")

def get_mongo_client():
    """Get MongoDB client with connection pooling."""
    global mongo_client
    if mongo_client is None:
        try:
            mongo_client = pymongo.MongoClient(
                MONGO_URI,
                tls=True,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=5000,
                maxPoolSize=50
            )
            mongo_client.admin.command('ping')
            logger.info("MongoDB connection established")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
    return mongo_client

def extract_phone_number(filename):
    """Extract the phone number from the filename."""
    match = re.search(r'_(\d{10})_', filename)
    return match.group(1) if match else None

def extract_time_from_filename(filename):
    """Extract the time from the filename."""
    match = re.search(r'_\d{4}-\d{1,2}-\d{1,2}-(\d{1,2})-(\d{1,2})-\d{1,2}_', filename)
    if match:
        hour = int(match.group(1))
        minute = match.group(2)
        return f"{hour}:{minute}"
    return None

def convert_time_to_mongo_format(time_str):
    """Convert time string to MongoDB format."""
    try:
        time_parts = time_str.strip().split()
        if len(time_parts) == 2:
            return f"{time_parts[0]} {time_parts[1]}"
        elif len(time_parts) == 1 and ":" in time_parts[0]:
            return time_parts[0]
        raise ValueError("Invalid time format")
    except Exception as e:
        logger.error(f"Time conversion error: {e}")
        return None

def is_already_processed(filename, log_file):
    """Check if file has already been processed (successfully or failed)."""
    if not os.path.exists(log_file):
        return False
    with open(log_file, "r", encoding='utf-8') as f:
        for line in f:
            if line.startswith(filename + ","):
                return True
    return False

# ---- CORE FUNCTIONS ----
def transcribe_audio(file_path):
    """Transcribe audio using Groq API."""
    try:
        with open(file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(file_path, file.read()),
                model="whisper-large-v3",
                temperature=1,
                language="en",
                response_format="verbose_json",
            )
        return transcription.text if transcription.text else None
    except Exception as e:
        logger.error(f"Transcription failed for {file_path}: {e}")
        return None

def store_transcription(filename, transcription_text):
    """Store transcription in MongoDB."""
    try:
        db = get_mongo_client()[MONGO_DB]
        collection = db[MONGO_COLLECTION]
        
        # Try to match by filename first
        result = collection.update_one(
            {"filename": filename},
            {"$set": {"transcription": transcription_text}},
            upsert=False
        )
        
        if result.matched_count > 0:
            logger.info(f"Transcription stored (matched by filename): {filename}")
            return True

        # Fallback: Try to match by phone number and time
        phone_number = extract_phone_number(filename)
        extracted_time = extract_time_from_filename(filename)
        mongo_time = convert_time_to_mongo_format(extracted_time) if extracted_time else None

        if phone_number and mongo_time:
            result = collection.update_one(
                {"phone_number": phone_number, "call_time": mongo_time},
                {"$set": {"transcription": transcription_text}},
                upsert=False
            )
            if result.matched_count > 0:
                logger.info(f"Transcription stored (matched by phone/time): {filename}")
                return True

        logger.warning(f"No matching record found for: {filename}")
        return False

    except Exception as e:
        logger.error(f"MongoDB storage failed: {e}")
        return False

def generate_embeddings(filename, text):
    """Generate and store embeddings."""
    try:
        db = get_mongo_client()[MONGO_DB]
        collection = db[EMBEDDING_COLLECTION]
        
        # Skip if already exists
        if collection.find_one({"filename": filename}):
            return True
            
        embedding = embedding_model.encode([text])[0]
        result = collection.insert_one({
            "filename": filename,
            "embedding": embedding.tolist(),
            "date_processed": datetime.now().strftime("%d-%m-%Y")
        })
        
        if result.inserted_id:
            with open(EMBEDDINGS_LOG, "a", encoding='utf-8') as f:
                f.write(f"{filename},{result.inserted_id},{datetime.now().isoformat()}\n")
            return True
        return False
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return False

def process_single_file(audio_file):
    """Process a single audio file with retry logic."""
    try:
        input_path = os.path.join(DOWNLOAD_FOLDER, audio_file)
        base_filename = os.path.splitext(audio_file)[0]
        
        # Skip if already processed (successfully or failed)
        if is_already_processed(base_filename, TRANSCRIPTION_LOG):
            logger.info(f"Already processed: {audio_file}")
            return False
            
        # Transcribe with retry
        max_attempts = 2
        transcription = None
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"Transcribing attempt {attempt + 1} for: {audio_file}")
                transcription = transcribe_audio(input_path)
                if transcription:
                    break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {audio_file}: {e}")
                if attempt == max_attempts - 1:  # Last attempt failed
                    # Log the failed attempt to prevent reprocessing
                    with open(TRANSCRIPTION_LOG, "a", encoding='utf-8') as f:
                        f.write(f"{base_filename},{datetime.now().isoformat()},failed\n")
                    return False
                time.sleep(1)  # Small delay between attempts

        if not transcription:
            # Log the failure
            with open(TRANSCRIPTION_LOG, "a", encoding='utf-8') as f:
                f.write(f"{base_filename},{datetime.now().isoformat()},failed\n")
            return False
            
        # Store transcription
        if not store_transcription(base_filename, transcription):
            # Log the failure
            with open(TRANSCRIPTION_LOG, "a", encoding='utf-8') as f:
                f.write(f"{base_filename},{datetime.now().isoformat()},failed\n")
            return False
            
        # Generate embeddings
        if not generate_embeddings(base_filename, transcription):
            # Log the failure
            with open(TRANSCRIPTION_LOG, "a", encoding='utf-8') as f:
                f.write(f"{base_filename},{datetime.now().isoformat()},failed\n")
            return False
            
        # Log success
        with open(TRANSCRIPTION_LOG, "a", encoding='utf-8') as f:
            f.write(f"{base_filename},{datetime.now().isoformat()},success\n")
            
        # Clean up
        os.remove(input_path)
        logger.info(f"Successfully processed: {audio_file}")
        return True
        
    except Exception as e:
        logger.error(f"Processing failed for {audio_file}: {e}")
        # Log the failure
        with open(TRANSCRIPTION_LOG, "a", encoding='utf-8') as f:
            f.write(f"{base_filename},{datetime.now().isoformat()},failed\n")
            os.remove(input_path)
        return False

def process_files_parallel(audio_files, max_workers=4):
    """Process files in parallel with progress tracking."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_file, audio_files),
            total=len(audio_files),
            desc="Processing files"
        ))
    return sum(results)  # Return count of successful processes

def run_llm(successful_files):
    """Run LLM processing."""
    if not successful_files:
        logger.warning("No files processed successfully - skipping LLM")
        return
        
    logger.info(f"Running LLM for {len(successful_files)} files")
    subprocess.run(["python", "llm.py"])

# ---- MAIN EXECUTION ----
def main():
    # Initialize
    load_models()
    get_mongo_client()

    # Get list of audio files
    audio_files = [
        f for f in os.listdir(DOWNLOAD_FOLDER)
        if f.endswith((".aac", ".wav", ".mp4", ".mp3"))
    ]

    if not audio_files:
        logger.info("No audio files found to process")
        return

    logger.info(f"Found {len(audio_files)} files to process")

    for audio_file in audio_files:
        # Step-by-step process each file
        success = process_single_file(audio_file)

        if success:
            # Run LLM immediately after successful transcription+embedding
            logger.info(f"Running LLM for: {audio_file}")
            subprocess.run(["python", "llm.py", audio_file])  # optionally pass filename
        else:
            logger.warning(f"Skipping LLM due to failed transcription or embedding for: {audio_file}")

    logger.info("All processing completed")

if __name__ == "__main__":
    # Set environment variable to disable oneDNN optimizations if needed
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    main()