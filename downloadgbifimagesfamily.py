import os
import json
import asyncio
import aiohttp
from aiohttp_retry import RetryClient, ExponentialRetry
import requests
import pandas as pd
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Define the local folder paths
source_folder_path = './metadata'  # Path of the folder containing gbifmultimedia files
target_folder_path = './imagesTrochilidae'  # Path of the target folder for images and metadata

# Define headers with a common User-Agent
HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/90.0.4430.93 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    # Add other headers if necessary
}

# Define a semaphore to limit the number of concurrent requests
CONCURRENT_REQUESTS = 10  # Adjust this number based on the server's capacity and your network
SEMAPHORE = asyncio.Semaphore(CONCURRENT_REQUESTS)

# Function to check if a file exists locally
def file_exists_locally(file_name, folder_path):
    try:
        file_path = os.path.join(folder_path, file_name)
        return os.path.exists(file_path)
    except Exception as e:
        logging.error(f"Error checking if file exists {file_name}: {e}")
        raise

# Function to create a subfolder locally if it doesn't exist
def get_or_create_local_folder(folder_name, parent_folder_path):
    try:
        folder_path = os.path.join(parent_folder_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            logging.info(f"Created folder: {folder_path}")
        return folder_path
    except Exception as e:
        logging.error(f"Error creating folder {folder_name}: {e}")
        raise

# Function to save a file locally
def save_file_locally(file_name, file_content, folder_path):
    try:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'wb') as f:
            f.write(file_content)
        logging.info(f"Saved file: {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error saving file {file_name}: {e}")
        raise

# Function to download a metadata file locally
def download_metadata_file(file_path):
    try:
        return pd.read_csv(file_path, delimiter='\t', low_memory=False)
    except Exception as e:
        logging.error(f"Failed to download metadata from {file_path}: {e}")
        raise

# List gbifmultimedia files in the source local folder
def list_local_files(folder_path):
    try:
        files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        logging.info(f"Found {len(files)} metadata files in {folder_path}")
        return files
    except Exception as e:
        logging.error(f"Failed to list files in folder {folder_path}: {e}")
        raise

# Function to check if the URL is reachable (using a HEAD request)
def is_url_reachable(url, timeout=10):
    try:
        response = requests.head(url, headers=HEADERS, timeout=timeout)
        if response.status_code == 200:
            return True
        else:
            logging.warning(f"URL {url} returned status code {response.status_code}")
            return False
    except requests.RequestException as e:
        logging.warning(f"URL {url} is not reachable: {e}")
        return False

# Asynchronous function to download an image with retry and semaphore
async def download_image_async(session, url, timeout=20):
    retry_options = ExponentialRetry(
        attempts=3,
        start_timeout=1.0,
        max_timeout=10.0,
        statuses={500, 502, 503, 504},
        exceptions={aiohttp.ClientError, asyncio.TimeoutError},
    )
    async with RetryClient(session, retry_options=retry_options) as retry_client:
        async with SEMAPHORE:
            try:
                async with retry_client.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        logging.info(f"Successfully downloaded {url}")
                        return await response.read()
                    else:
                        logging.error(f"Error downloading image {url}: {response.status}")
                        return None
            except Exception as e:
                logging.error(f"Error downloading image {url}: {e}")
                return None

# Asynchronous function to download images and create metadata
async def download_and_create_metadata_async(row, subfolder_path, download_counts, session):
    try:
        media = json.loads(row['media'].replace("'", '"')) if isinstance(row['media'], str) else []
        gbif_id = row['gbifID']

        image_urls = [item['identifier'] for item in media if item.get('type') == 'StillImage']

        tasks = []
        image_name_mapping = {}  # Map task index to image name

        for idx, image_url in enumerate(image_urls, start=1):
            # **New Addition:** Skip URLs that contain 'observation.org'
            if 'observation.org' in image_url.lower():
                logging.info(f"Ignoring URL from observation.org: {image_url}")
                continue

            image_name = os.path.basename(image_url)
            image_name_with_id = f"{os.path.splitext(image_name)[0]}_{gbif_id}_{idx}{os.path.splitext(image_name)[1]}"

            if not file_exists_locally(image_name_with_id, subfolder_path):
                if is_url_reachable(image_url):
                    tasks.append(download_image_async(session, image_url))
                    image_name_mapping[len(tasks) - 1] = image_name_with_id
                else:
                    download_counts['fail'] += 1
                    logging.error(f"URL not reachable: {image_url}")
            else:
                logging.info(f"File already exists locally: {image_name_with_id}")

        if not tasks:
            logging.info(f"No new images to download for gbifID {gbif_id}")
            return

        results = await asyncio.gather(*tasks)

        for idx, image_content in enumerate(results):
            if image_content:
                image_name_with_id = image_name_mapping[idx]
                save_file_locally(image_name_with_id, image_content, subfolder_path)
                download_counts['success'] += 1

                # Create metadata content
                metadata_content = '\n'.join([f"{key}: {value}" for key, value in row.items()])
                metadata_file_name = f"{os.path.splitext(image_name_with_id)[0]}.txt"
                save_file_locally(metadata_file_name, metadata_content.encode('utf-8'), subfolder_path)
            else:
                download_counts['fail'] += 1
                logging.error(f"Failed to download image for gbifID {gbif_id}")

    except Exception as e:
        download_counts['fail'] += 1
        logging.error(f"Failed to process record with gbifID {row.get('gbifID')}: {e}")

# Main asynchronous processing function
async def process_metadata_files_async():
    local_files = list_local_files(source_folder_path)
    download_counts = {'success': 0, 'fail': 0}

    timeout = aiohttp.ClientTimeout(total=60)  # Adjust as necessary

    async with aiohttp.ClientSession(headers=HEADERS, timeout=timeout) as session:
        tasks = []

        for local_file in local_files:
            file_path = os.path.join(source_folder_path, local_file)

            # Extract the species name from the file name
            species_name = '_'.join(os.path.splitext(local_file)[0].split('_')[1:])

            # Get or create a subfolder in the target local folder named after the species
            subfolder_path = get_or_create_local_folder(species_name, target_folder_path)

            try:
                # Download metadata from local folder
                df = download_metadata_file(file_path)
                logging.info(f"Processing file: {file_path} with {len(df)} records")
            except Exception as e:
                logging.error(f"Skipping file {local_file} due to download error: {e}")
                continue

            # Check if the 'media' column exists
            if 'media' not in df.columns:
                logging.error(f"'media' column not found in {local_file}, skipping.")
                continue

            for _, row in df.iterrows():
                tasks.append(
                    download_and_create_metadata_async(row, subfolder_path, download_counts, session)
                )

        # Execute all tasks with controlled concurrency
        await asyncio.gather(*tasks)

    # Log the counts at the end
    logging.info(f"Total successful downloads: {download_counts['success']}")
    logging.info(f"Total failed downloads: {download_counts['fail']}")

# Wrapper function to run the asynchronous main function
def process_metadata_files():
    start_time = time.time()
    try:
        asyncio.run(process_metadata_files_async())
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
    end_time = time.time()
    elapsed = end_time - start_time
    logging.info(f"Data downloaded and organized by species in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    process_metadata_files()
