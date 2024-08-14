from pygbif import species
import os
import logging
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
import mimetypes

# List of species names
species_names = [
    "Calypte anna",
    "Selasphorus sasin",
    "Selasphorus rufus",
    "Archilochus alexandri",
    "Calypte costae",
    "Selasphorus calliope",
    "Selasphorus platycercus",
    "Archilochus colubris",
    "Amazilia violiceps",
    "Cynanthus latirostris",
    "Eugenes fulgens"
]

# Define the local folder paths
source_folder_path = './metadata'  # Replace with the path of the folder containing gbifmultimedia files
target_folder_path = './images'  # Replace with the path of the target folder for images and metadata

# Function to get taxon key
def get_taxon_key(name):
    result = species.name_backbone(name=name)
    return result.get('usageKey')

# Get taxon keys for all species
taxon_keys = {get_taxon_key(name): name for name in species_names}
print(taxon_keys)

# Configure logging
def configure_logging(species_folder):
    log_file = os.path.join(species_folder, 'failures.log')
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )
    return log_file

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_image(url, timeout=60):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        return response.content
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading image {url}: {e}")
        raise

# Function to save a file locally
def save_file_locally(file_name, file_content, folder_path):
    try:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'wb') as f:
            f.write(file_content)
        return file_path
    except Exception as e:
        logging.error(f"Error saving file {file_name}: {e}")
        raise

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
        return folder_path
    except Exception as e:
        logging.error(f"Error creating folder {folder_name}: {e}")
        raise

# Function to download an image and create metadata
def download_and_create_metadata(row, subfolder_path, download_counts):
    try:
        media = json.loads(row['media'].replace("'", '"')) if isinstance(row['media'], str) else []
        gbif_id = row['gbifID']

        image_urls = [item['identifier'] for item in media if item.get('type') == 'StillImage']

        for idx, image_url in enumerate(image_urls, start=1):
            image_name = image_url.split('/')[-1]
            image_name_with_id = f"{os.path.splitext(image_name)[0]}_{gbif_id}_{idx}{os.path.splitext(image_name)[1]}"

            if not file_exists_locally(image_name_with_id, subfolder_path):
                try:
                    # Download the image with retry logic
                    image_content = download_image(image_url)
                    download_counts['success'] += 1

                    # Save the image locally
                    mime_type, _ = mimetypes.guess_type(image_name_with_id)
                    if mime_type is None:
                        mime_type = 'application/octet-stream'  # Fallback MIME type
                    save_file_locally(image_name_with_id, image_content, subfolder_path)

                    # Create metadata content
                    metadata_content = '\n'.join([f"{key}: {value}" for key, value in row.items()])
                    metadata_file_name = f"{os.path.splitext(image_name_with_id)[0]}.txt"
                    save_file_locally(metadata_file_name, metadata_content.encode('utf-8'), subfolder_path)
                except Exception as e:
                    download_counts['fail'] += 1
                    logging.error(f"Failed to download image {image_url} for gbifID {gbif_id}: {e}")
            else:
                print(f"File already exists: {image_name_with_id}")
    except Exception as e:
        download_counts['fail'] += 1
        logging.error(f"Failed to process record with gbifID {row.get('gbifID')}: {e}")

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
        print(f"Files found in folder {folder_path}: {files}")
        return files
    except Exception as e:
        logging.error(f"Failed to list files in folder {folder_path}: {e}")
        raise

local_files = list_local_files(source_folder_path)
print(f"Local files: {local_files}")  # Debug statement

# Process each gbifmultimedia file
for local_file in local_files:
    file_path = os.path.join(source_folder_path, local_file)

    # Extract the suffix from the file name
    suffix = os.path.splitext(local_file)[0].split('_')[-1]

    # Get or create a subfolder in the target local folder
    subfolder_name = f"subfolder_{suffix}"
    subfolder_path = get_or_create_local_folder(subfolder_name, target_folder_path)

    # Determine the species folder path for logging
    species_folder = os.path.join('./logs', suffix)

    # Ensure the species folder exists
    if not os.path.exists(species_folder):
        os.makedirs(species_folder)

    # Configure logging for this species
    configure_logging(species_folder)

    try:
        # Download metadata from local folder
        df = download_metadata_file(file_path)
    except Exception as e:
        print(f"Skipping file {local_file} due to download error: {e}")
        continue

    # Print the column names for debugging
    print(f"Columns in {local_file}: {df.columns.tolist()}")

    # Check if the 'media' column exists
    if 'media' not in df.columns:
        print(f"'media' column not found in {local_file}, skipping.")
        continue

    # Initialize counters for downloads
    download_counts = {'success': 0, 'fail': 0}

    try:
        # Use ThreadPoolExecutor to download images and create metadata files concurrently
        with ThreadPoolExecutor(max_workers=1) as executor:  # Reduce number of threads to avoid concurrency issues
            futures = [executor.submit(download_and_create_metadata, row, subfolder_path, download_counts) for _, row in df.iterrows()]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error occurred during processing: {e}")

    except Exception as e:
        print(f"Skipping file {local_file} due to processing error: {e}")
        continue

    # Log the counts at the end
    log_file = os.path.join(species_folder, 'failures.log')
    with open(log_file, 'a') as log:
        log.write(f"\nTotal successful downloads: {download_counts['success']}\n")
        log.write(f"Total failed downloads: {download_counts['fail']}\n")

print("Data downloaded and organized by species.")
