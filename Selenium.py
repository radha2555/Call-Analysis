import time
import os
import subprocess
import logging
import schedule
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import pymongo
import certifi
import json
from datetime import datetime
import bson
import shutil

# Disable warnings
import warnings
warnings.filterwarnings("ignore")

# ---- CONFIG ----
# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

USERNAME = "44521492"
PASSWORD = "Tanmay@345"
URL = "https://ctv1.sarv.com/telephony/0/dashboard/"
CHECK_INTERVAL = 30  # 30 seconds
MAX_WAIT_TIME = 60  # Max seconds to wait for download completion

# ---- FILES & FOLDERS ----
DOWNLOAD_FOLDER = os.path.join(os.getcwd(), "data")  # Folder to save files
DOWNLOAD_LOG = "downloaded_files.txt"  # File to track downloaded files
TEMPORARY_LOG = "temp_downloaded_files.txt"  # Temporary log for current session
PHONE_RECORDS_LOG = "phone_records.log"  # File to track phone records
LLM_PROCESSED_LOG = "processed_llm_files.txt"  # Track processed files
TRANSCRIPTION_LOG = "transcription_log.txt"  # File to track transcribed files
EMBEDDINGS_LOG = "embeddings_log.txt"  # Log file to track embeddings

# Ensure folders and log files exist
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
open(TEMPORARY_LOG, "a").close()  # Create temporary log file if it doesn't exist
open(PHONE_RECORDS_LOG, "a").close()  # Create phone records log file if it doesn't exist

# MongoDB Configuration
MONGO_URI= os.getenv("MONGO_URI")
MONGO_DB = "CallAnalysis"
MONGO_COLLECTION = "phone_records"

# Configure Chrome options
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": DOWNLOAD_FOLDER,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True,
    "profile.default_content_setting_values.automatic_downloads": 1,
})
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--no-sandbox")

def extract_filenames_from_log():
    """Extracts filenames appearing after the second '*' and removes the file extension."""
    extracted_filenames = []
    try:
        with open(TEMPORARY_LOG, "r") as file:
            for line in file:
                parts = line.strip().split("*", 2)
                if len(parts) == 3:
                    filename_with_ext = parts[2]
                    filename = filename_with_ext.split(".", 1)[0]
                    extracted_filenames.append(filename)
    except Exception as e:
        print(f"❌ Error reading download log: {e}")
    return extracted_filenames

def save_downloaded_file(log_entry):
    """Save the log entry to the permanent log file."""
    with open(DOWNLOAD_LOG, "a") as f:
        f.write(log_entry + "\n")

def save_downloaded(log_entry):
    try:
        with open(TEMPORARY_LOG, "a") as f:
            f.write(log_entry + "\n")
        print(f"✅ Saved to logs: {log_entry}")
    except Exception as e:
        print(f"❌ Error saving to log files: {e}")

def clear_temp_log():
    """Clear the temporary log file."""
    try:
        open(TEMPORARY_LOG, "w").close()
        print("✅ Temporary log file cleared.")
    except Exception as e:
        print(f"❌ Error clearing temporary log file: {e}")

def extract_phone_numbers(driver):
    """Extracts phone numbers and times from the webpage."""
    phone_data = []
    try:
        phone_elements = driver.find_elements(By.XPATH, '//td[1]/span[1]/span/a')
        time_elements = driver.find_elements(By.XPATH, '//td[2]/span[1]')

        for phone_elem, time_elem in zip(phone_elements, time_elements):
            phone_number = phone_elem.get_attribute("href").split("/")[-1] if phone_elem else None
            call_time = time_elem.text.strip()
            
            if call_time:
                call_time = f"{call_time[:5]} {call_time[-2:]}"  # Extract HH:MM + AM/PM
            
            if phone_number and call_time:
                phone_data.append({
                    "phone_number": phone_number, 
                    "call_time": call_time,
                    "timestamp": datetime.now().isoformat()  # Add current timestamp
                })

    except Exception as e:
        print("❌ Error extracting phone numbers:", e)
    return phone_data

def load_phone_records_from_log():
    """Load phone records from the local log file."""
    records = []
    try:
        with open(PHONE_RECORDS_LOG, "r") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"❌ Error reading phone records log: {e}")
    return records

def save_phone_record_to_log(record):
    """Save a phone record to the local log file."""
    try:
        with open(PHONE_RECORDS_LOG, "a") as f:
            f.write(bson.json_util.dumps(record) + "\n")
        print(f"✅ Saved phone record to log: {record['phone_number']}")
    except Exception as e:
        print(f"❌ Error saving phone record to log: {e}")

def is_record_exists(record):
    """Check if a record exists either in log file or MongoDB."""
    # First check the local log file
    existing_records = load_phone_records_from_log()
    for existing in existing_records:
        if (existing["phone_number"] == record["phone_number"] and 
            existing["call_time"] == record["call_time"]):
            return True
    return False

def store_phone_records(phone_data):
    """Stores phone records in both MongoDB and local log file."""
    inserted_count = 0
    filenames = extract_filenames_from_log()

    if not filenames:
        print("⏭️ No new files downloaded - skipping database update")
        return

    for i, record in enumerate(phone_data):
        if i >= len(filenames):
            break
            
        # Add filename to record if available
        record["filename"] = filenames[i]
        
        # Skip if record already exists
        if is_record_exists(record):
            print(f"⏭️ Skipping duplicate record: {record['phone_number']}")
            continue
        
        # Store in MongoDB
        try:
            client = pymongo.MongoClient(
                MONGO_URI, tls=True, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000
            )
            db = client[MONGO_DB]
            collection = db[MONGO_COLLECTION]
            
            collection.insert_one(record)
            inserted_count += 1
            print(f"✅ Inserted new record to MongoDB: {record}")
        except Exception as e:
            print(f"❌ Error storing in MongoDB: {e}")
        finally:
            if "client" in locals():
                client.close()
        
        # Store in local log file
        save_phone_record_to_log(record)

    print(f"📞✅ Processed {len(phone_data)} records. Inserted {inserted_count} new records.")

def load_downloaded_files():
    """Load the list of previously downloaded files."""
    if os.path.exists(DOWNLOAD_LOG):
        with open(DOWNLOAD_LOG, "r") as f:
            return {line.strip() for line in f.readlines()}  
    return set()

def wait_for_downloads():
    """Wait until all downloads in progress are finished."""
    start_time = time.time()
    while time.time() - start_time < MAX_WAIT_TIME:
        if not any(filename.endswith(".crdownload") for filename in os.listdir(DOWNLOAD_FOLDER)):
            print("✅ All downloads completed!")
            return
        time.sleep(2)
    print("⚠️ Timeout! Some files may not have fully downloaded.")

def download_all_files(driver):
    """Download all unique files and wait for each to complete."""
    downloaded_files = load_downloaded_files()
    new_downloads = set()

    try:
        download_links = driver.find_elements(By.XPATH, "//a[contains(@href, '.mp3') or contains(@href, '.wav')]")

        if not download_links:
            print("⏳ No files available for download.")
            return
        
        unique_files = {}
        for download_link in download_links:
            file_url = download_link.get_attribute("href")
            filename = file_url.split("/")[-1].strip()
            if filename not in unique_files:
                unique_files[filename] = download_link

        for filename, download_link in unique_files.items():
            if filename in downloaded_files or filename in new_downloads:
                continue

            driver.execute_script("arguments[0].scrollIntoView(true);", download_link)
            time.sleep(1)
            driver.execute_script("arguments[0].click();", download_link)

            print(f"✅ Download triggered: {filename}")
            save_downloaded_file(filename) 
            save_downloaded(filename)
            new_downloads.add(filename)

            wait_for_downloads()
            time.sleep(2)

    except Exception as e:
        print(f"⚠️ Error clicking download links: {e}")

def initialize_session(driver, wait):
    """Initialize the browser session and navigate to report tab with filters"""
    try:
        driver.get(URL)

        # Perform login
        username_field = wait.until(EC.presence_of_element_located((By.NAME, "username")))
        password_field = driver.find_element(By.NAME, "password")
        username_field.send_keys(USERNAME)
        password_field.send_keys(PASSWORD)
        password_field.send_keys(Keys.RETURN)
        time.sleep(5)
        print("✅ Logged in successfully")

        # Navigate to Report tab
        report_tab = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "Report")))
        report_tab.click()
        print("✅ Navigated to Report tab")
        time.sleep(3)

        # Apply filters
        today_filter_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "todayFilter")))
        today_filter_button.click()
        print("✅ Applied 'Today' filter")
        
        search_by_button = wait.until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[6]/div[1]/div[2]/button")))
        search_by_button.click()
        print("✅ Clicked on 'Search By' button")

        search_status = wait.until(EC.element_to_be_clickable((By.ID, "search_status")))
        driver.execute_script("arguments[0].scrollIntoView(true);", search_status)
        time.sleep(1)
        driver.execute_script("arguments[0].click();", search_status)
        print("✅ Clicked on 'Status' dropdown")

        select = Select(search_status)
        select.select_by_visible_text("Both Answered")
        print("✅ Selected 'Both Answered'")

        search_button = wait.until(EC.element_to_be_clickable((By.ID, "searchButton")))
        search_button.click()
        print("✅ Clicked on 'Search' button")
        time.sleep(5)

    except Exception as e:
        print(f"❌ Error initializing session: {e}")
        raise

def clear_all_logs():
    """Clear all log files at once"""
    log_files = [
        TEMPORARY_LOG,
        PHONE_RECORDS_LOG,
        DOWNLOAD_LOG,
        "transcription.log",
        EMBEDDINGS_LOG,
        LLM_PROCESSED_LOG,
        TRANSCRIPTION_LOG
    ]

    try:
        for log_file in log_files:
            if os.path.exists(log_file):
                open(log_file, "w").close()
                print(f"✅ Cleared log file: {log_file}")
            else:
                print(f"⚠️ Log file not found: {log_file}")
    except Exception as e:
        print(f"❌ Error clearing log files: {e}")

def scheduled_log_clearance():
    """Wrapper function for scheduled log clearance"""
    print("\n⏰ Running scheduled log clearance at 3:00 AM")
    clear_all_logs()
    print("✅ Log clearance completed. Continuing normal operations...\n")

def setup_scheduler():
    """Set up the scheduled tasks"""
    # Clear logs every day at 3:00 AM
    schedule.every().day.at("03:00").do(scheduled_log_clearance)
    print("⏰ Scheduled log clearance set for 3:00 AM daily")

def main():
    setup_scheduler()  # Set up the scheduled tasks
    
    while True:
        try:
            # Initialize driver and wait for each iteration
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            wait = WebDriverWait(driver, 20)
            
            # Run pending scheduled jobs
            schedule.run_pending()
            
            # Initialize session (login and apply filters)
            initialize_session(driver, wait)
            
            # Download new files
            download_all_files(driver)
        
            # Only process if new files were downloaded
            if os.path.getsize(TEMPORARY_LOG) > 0:
                phone_data = extract_phone_numbers(driver)
                if phone_data:
                    store_phone_records(phone_data)
                    print("📞✅ Phone numbers stored in MongoDB and log file.")
                    # Run transcription and LLM processing
                    try:
                        subprocess.run(["python", "new.py"], check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Error running transcription: {e}")

            clear_temp_log()
            print(f"🔄 Waiting {CHECK_INTERVAL} seconds before next check...")
            
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            
        finally:
            try:
                if driver:
                    driver.quit()
            except:
                pass
            
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()