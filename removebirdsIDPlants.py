import os
import json
import logging
from logging.handlers import RotatingFileHandler
from PIL import Image, UnidentifiedImageError
import pandas as pd
from pygbif import species
from datetime import datetime
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
import requests
import cv2
import ast  # Ensure ast is imported for metadata parsing

# Configuration Constants
API_KEY = None
PROJECT_TYPE = "kt"  # Assuming 'kt' is a valid project type; adjust as needed
LANG = "en"  # Language parameter for project API
API_ENDPOINT_IDENTIFY_TEMPLATE = "https://my-api.plantnet.org/v2/identify/{project_id}?api-key={api_key}"
API_ENDPOINT_PROJECTS = "https://my-api.plantnet.org/v2/projects"

CONFIG_FILE = '/home/exouser/pri/config.json'
RATE_LIMIT = 20000  # Maximum identifications per day
COMPLETED_FOLDERS_FILE = '/home/exouser/pri/completed_folders.txt'

# Metadata Configuration Constants
METADATA_BASE_DIR = '/home/exouser/pri/imagesTrochilidae/'  # Base directory for metadata files
CSV_DIRECTORY = '/home/exouser/pri/'  # Directory containing the CSV files
OUTPUT_SUFFIX = '_updated'  # Suffix for the updated CSV files

# Log Configuration Constants
LOG_DIR = '/home/exouser/pri/logs/'  # Directory to store all log files
os.makedirs(LOG_DIR, exist_ok=True)  # Ensure the log directory exists

PLANT_LOG_FILE = os.path.join(LOG_DIR, 'plant_identifier.log')
METADATA_LOG_FILE = os.path.join(LOG_DIR, 'update_csv_with_metadata.log')
SUMMARY_LOG_FILE = '/home/exouser/pri/processing_summary.log'
INVALID_DATE_LOG_FILE = '/home/exouser/pri/invalid_date_metadata_images.log'

# Set up logging with rotation for plant_identifier.log
plant_handler = RotatingFileHandler(PLANT_LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)  # 5 MB per file, keep 5 backups
logging.basicConfig(
    handlers=[plant_handler],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Additional Metadata Logging Handler
metadata_handler = RotatingFileHandler(METADATA_LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)  # 5 MB per file, keep 5 backups
metadata_logger = logging.getLogger('metadata_logger')
metadata_logger.setLevel(logging.INFO)
metadata_logger.addHandler(metadata_handler)

# Log the absolute paths for verification
logging.info(f"Plant Identifier Log File: {PLANT_LOG_FILE}")
metadata_logger.info(f"Metadata Log File: {METADATA_LOG_FILE}")

# Initialize a list to track images with invalid/missing date fields
invalid_date_metadata_images = []

# Counters for summary report
total_images_processed = 0
images_with_valid_dates = 0
images_with_invalid_dates = 0
images_skipped_due_to_critical_errors = 0

# Device Configuration for GPU Utilization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    logging.info(f"CUDA is available. Using GPU: {gpu_name}")
else:
    logging.info("CUDA is not available. Using CPU.")

# Load the COCO class labels
COCO_CLASSES = [
    '__background__', 'person', 'bird', 'tree', 'bush', 'grass', 'flower',
    'rock', 'stream', 'river', 'lake', 'mountain', 'sky', 'cloud',
    'butterfly', 'insect', 'squirrel', 'deer', 'rabbit', 'frog',
    'snake', 'fish', 'mushroom', 'leaf', 'log', 'path',
    'binoculars', 'camera', 'backpack', 'hat', 'walking stick',
    'tent', 'campfire', 'kayak', 'canoe', 'birdhouse', 'hummingbird',
    'feeder', 'nest', 'birdbath', 'birdwatching platform',
    'bird identification book', 'scope', 'bird call', 'waterfall',
    'pond', 'branch', 'twig', 'fence', 'trail sign', 'sand', 'shell'
]

# Load a pre-trained segmentation model and move it to the selected device
try:
    # Using ResNet50 for reduced memory footprint
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights).to(device).eval()
    logging.info("Segmentation model loaded and moved to device successfully.")
except Exception as e:
    logging.error(f"Error loading segmentation model: {e}")
    raise

# Define a transformation to preprocess the input image for segmentation
transform_seg = T.Compose([
    T.ToTensor(),  # Convert image to tensor
    weights.transforms()  # Use the transforms associated with the weights
])

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            # Validate config structure
            if not isinstance(config, dict):
                raise ValueError("Config file is not a JSON object.")
            required_fields = ['date', 'count_today', 'species_progress', 'completed_species']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing '{field}' in config file.")
            logging.info("Configuration loaded successfully.")
            return config
        except Exception as e:
            logging.error(f"Error loading config file: {e}")
            return {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'count_today': 0,
                'species_progress': {},
                'completed_species': []
            }
    else:
        # Initialize config
        config = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'count_today': 0,
            'species_progress': {},
            'completed_species': []
        }
        save_config(config)
        logging.info("Configuration file initialized.")
        return config

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        logging.info("Configuration saved successfully.")
    except Exception as e:
        logging.error(f"Error saving config file: {e}")

# Function to validate metadata fields and return processed_metadata
def validate_metadata(metadata, metadata_file_path):
    required_fields = {
        'scientificName': str,
        'speciesKey': str,
        'decimalLatitude': float,
        'decimalLongitude': float,
        'coordinateUncertaintyInMeters': float,
        'continent': str,
        'stateProvince': str,
        'year': int,
        'month': int,
        'day': int,
        'eventDate': str
    }

    errors = []
    metadata_valid = True
    # Initialize fields to default values
    processed_metadata = {
        'year': np.nan,
        'month': np.nan,
        'day': np.nan
    }

    for field, field_type in required_fields.items():
        value = metadata.get(field)
        if value is None:
            if field in ['year', 'month', 'day']:
                # These fields can be ignored if invalid
                continue
            errors.append(f"Missing required metadata field: {field}")
            metadata_valid = False
            continue
        if value == 'Unknown':
            if field in ['year', 'month', 'day']:
                # These fields can be ignored if unknown
                continue
            errors.append(f"Metadata field '{field}' is 'Unknown'")
            metadata_valid = False
            continue
        # Type checking and conversion
        try:
            if field_type == float:
                processed_metadata[field] = float(value)
            elif field_type == int:
                processed_metadata[field] = int(value)
            elif field_type == str:
                processed_metadata[field] = str(value)
        except ValueError:
            if field in ['year', 'month', 'day']:
                # Ignore invalid year, month, day and rely on eventDate
                logging.warning(
                    f"Metadata field '{field}' has invalid type. Expected {field_type.__name__}. Ignoring this field and using 'eventDate'.")
                # Append to tracking list
                invalid_date_metadata_images.append(os.path.abspath(metadata_file_path))
                continue
            errors.append(f"Metadata field '{field}' has invalid type. Expected {field_type.__name__}.")
            metadata_valid = False

    # Additional validation for latitude and longitude ranges
    lat = processed_metadata.get('decimalLatitude')
    lon = processed_metadata.get('decimalLongitude')
    if isinstance(lat, float):
        if not (-90.0 <= lat <= 90.0):
            errors.append(f"Invalid latitude value: {lat}. Must be between -90 and 90.")
            metadata_valid = False
    if isinstance(lon, float):
        if not (-180.0 <= lon <= 180.0):
            errors.append(f"Invalid longitude value: {lon}. Must be between -180 and 180.")
            metadata_valid = False

    if errors:
        for error in errors:
            logging.error(error)
        # Log the full path to the metadata file if validation fails for reasons other than year/month/day
        logging.error(f"Metadata validation failed for file: {os.path.abspath(metadata_file_path)}")
        return False
    return processed_metadata

# Function to validate image
def validate_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify that it is, in fact, an image
        # Re-open for processing since verify() can close the file
        with Image.open(image_path) as img:
            img = img.convert('RGB')  # Ensure RGB
            # Optional: Check image dimensions
            width, height = img.size
            if width < 100 or height < 100:
                logging.warning(f"Image {image_path} has unusually small dimensions: {width}x{height}.")
        return True
    except (UnidentifiedImageError, IOError) as e:
        logging.error(f"Image validation failed for {image_path}: {e}")
        return False

# Function to perform segmentation
def segment_image(image):
    # Ensure the image is in RGB mode (i.e., it has 3 channels)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    input_tensor = transform_seg(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    logging.info(f"Input tensor is on device: {input_tensor.device}")
    with torch.no_grad():
        try:
            output = model(input_tensor)['out'][0]
            logging.info(f"Model output is on device: {output.device}")
            output_predictions = output.argmax(0).byte().cpu().numpy()
            logging.info("Segmentation completed successfully.")
            return output_predictions
        except Exception as e:
            logging.error(f"Error during image segmentation: {e}")
            return None

# Function to find the "best guess" for a bird if not detected
def find_best_guess(segmentation_mask, unique_classes):
    if len(unique_classes) > 1:
        largest_object_class = None
        largest_object_size = 0

        for class_id in unique_classes:
            if class_id == 0:  # Skip background
                continue
            # Calculate the size of the current class
            object_size = np.sum(segmentation_mask == class_id)
            if object_size > largest_object_size:
                largest_object_size = object_size
                largest_object_class = class_id

        return largest_object_class
    return None

# Function to parse the custom metadata file format
def parse_metadata_file(metadata_file):
    metadata = {}
    try:
        with open(metadata_file, 'r') as file:
            for line in file:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    metadata[key.strip()] = value.strip()
    except Exception as e:
        logging.error(f"Error parsing metadata file {metadata_file}: {e}")
    return metadata

# Function to get the scientific name from GBIF using the gbifID
def get_scientific_name_from_gbif(gbif_id):
    if gbif_id != 'Unknown':
        try:
            result = species.name_usage(key=gbif_id)
            return result.get('scientificName', 'Unknown')
        except Exception as e:
            logging.error(f"Error retrieving scientific name for GBIF ID {gbif_id}: {str(e)}")
            return 'Unknown'
    return 'Unknown'

# Function to validate API response
def validate_api_response(json_result):
    if not isinstance(json_result, dict):
        logging.error("API response is not a JSON object.")
        return None
    if 'results' not in json_result:
        logging.error("API response missing 'results' field.")
        return None
    if not isinstance(json_result['results'], list):
        logging.error("'results' field is not a list.")
        return None
    if not json_result['results']:
        logging.warning("API response 'results' list is empty.")
        return None
    return json_result['results'][0]

# Function to extract additional metadata fields
def extract_additional_metadata(metadata_file_path):
    """
    Extracts additional metadata fields: creator, publisher, license, rightsHolder, identifier.

    Parameters:
        metadata_file_path (str): Path to the metadata .txt file.

    Returns:
        dict: Dictionary containing the extracted metadata fields.
    """
    extracted_fields = {
        'creator': set(),
        'publisher': set(),
        'license': set(),
        'rightsHolder': set(),
        'identifier': set()
    }

    if not os.path.exists(metadata_file_path):
        logging.warning(f"Metadata file not found: {metadata_file_path}")
        return {k: 'Unknown' for k in extracted_fields}  # Return 'Unknown' for all fields

    try:
        with open(metadata_file_path, 'r') as file:
            lines = file.readlines()

        metadata_dict = {}
        for line in lines:
            if ':' not in line:
                continue  # Skip lines without key-value pairs
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            metadata_dict[key] = value

        # Extract the 'extensions' field which contains multimedia information
        extensions_str = metadata_dict.get('extensions', '{}')
        try:
            extensions = ast.literal_eval(extensions_str)
        except Exception as e:
            logging.error(f"Error parsing 'extensions' in {metadata_file_path}: {e}")
            extensions = {}

        # Extract 'extensions' > 'Multimedia' if present
        multimedia_extensions = extensions.get('http://rs.gbif.org/terms/1.0/Multimedia', [])

        # Extract 'media' field if present
        media_str = metadata_dict.get('media', '[]')
        try:
            media = ast.literal_eval(media_str)
        except Exception as e:
            logging.error(f"Error parsing 'media' in {metadata_file_path}: {e}")
            media = []

        # Combine both multimedia lists
        combined_multimedia = multimedia_extensions + media

        for media_entry in combined_multimedia:
            creator = media_entry.get('http://purl.org/dc/terms/creator') or media_entry.get('creator')
            publisher = media_entry.get('http://purl.org/dc/terms/publisher') or media_entry.get('publisher')
            license_url = media_entry.get('http://purl.org/dc/terms/license') or media_entry.get('license')
            rights_holder = media_entry.get('http://purl.org/dc/terms/rightsHolder') or media_entry.get('rightsHolder')
            identifier = media_entry.get('http://purl.org/dc/terms/identifier') or media_entry.get('identifier')

            if creator:
                # Split by ';' in case multiple creators are present in a single entry
                creators = [c.strip() for c in creator.split(';')]
                extracted_fields['creator'].update(creators)
            if publisher:
                publishers = [p.strip() for p in publisher.split(';')]
                extracted_fields['publisher'].update(publishers)
            if license_url:
                licenses = [l.strip() for l in license_url.split(';')]
                extracted_fields['license'].update(licenses)
            if rights_holder:
                rights_holders = [r.strip() for r in rights_holder.split(';')]
                extracted_fields['rightsHolder'].update(rights_holders)
            if identifier:
                identifiers = [i.strip() for i in identifier.split(';')]
                extracted_fields['identifier'].update(identifiers)

    except Exception as e:
        logging.error(f"Error reading/parsing metadata file {metadata_file_path}: {e}")

    # Convert sets to '; ' separated strings or 'Unknown' if empty
    for key in extracted_fields:
        if extracted_fields[key]:
            extracted_fields[key] = '; '.join(sorted(extracted_fields[key]))
        else:
            extracted_fields[key] = 'Unknown'

    return extracted_fields

# Function to get project ID based on latitude and longitude
def get_project_id(lat, lon):
    params = {
        'lang': LANG,
        'lat': lat,
        'lon': lon,
        'type': PROJECT_TYPE,
        'api-key': API_KEY
    }
    try:
        response = requests.get(API_ENDPOINT_PROJECTS, params=params)
        if response.status_code == 200:
            projects = response.json()
            if projects:
                project_id = projects[0].get('id', 'all')  # Default to 'all' if 'id' is missing
                logging.info(f"Selected project ID: {project_id} for coordinates ({lat}, {lon})")
                return project_id
            else:
                logging.warning(f"No projects found for coordinates ({lat}, {lon}). Using 'all'.")
                return 'all'
        else:
            logging.error(f"Failed to retrieve projects for coordinates ({lat}, {lon}): {response.status_code} - {response.text}")
            return 'all'
    except Exception as e:
        logging.error(f"Exception occurred while fetching project ID: {e}")
        return 'all'

# Function to validate API response
def validate_api_response(json_result):
    if not isinstance(json_result, dict):
        logging.error("API response is not a JSON object.")
        return None
    if 'results' not in json_result:
        logging.error("API response missing 'results' field.")
        return None
    if not isinstance(json_result['results'], list):
        logging.error("'results' field is not a list.")
        return None
    if not json_result['results']:
        logging.warning("API response 'results' list is empty.")
        return None
    return json_result['results'][0]

# Function to process images, remove birds, and identify plants
def process_and_identify_images(input_dir, output_dir, results_csv, last_index, processed_images_set, config):
    global total_images_processed, images_with_valid_dates, images_with_invalid_dates, images_skipped_due_to_critical_errors
    # Ensure the output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory verified/created: {output_dir}")
    except Exception as e:
        logging.error(f"Error creating output directory {output_dir}: {e}")
        return last_index

    # Get a sorted list of image files
    try:
        all_images = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        total_images = len(all_images)
        logging.info(f"Found {total_images} image(s) in {input_dir}.")
    except Exception as e:
        logging.error(f"Error listing files in input directory {input_dir}: {e}")
        return last_index

    # Initialize or get the starting index
    start_index = last_index + 1

    for idx in range(start_index, total_images):
        if config['count_today'] >= RATE_LIMIT:
            logging.warning("Daily rate limit reached. Stopping processing for today.")
            break

        filename = all_images[idx]
        image_path = os.path.join(input_dir, filename)
        output_image_path = os.path.join(output_dir, filename)
        metadata_file = os.path.splitext(image_path)[0] + '.txt'

        # Check if the image has already been processed
        if filename in processed_images_set:
            logging.info(f"Image '{filename}' has already been processed. Skipping.")
            continue

        # Check if the inpainted image already exists
        if os.path.exists(output_image_path):
            logging.info(f"Inpainted image '{filename}' already exists. Using existing file for submission.")
            # Optionally, you can skip adding this image to processed_images_set
            # to allow re-submission by ensuring it's not in the CSV
            # However, based on your current tracking via CSV, it will be skipped unless removed from CSV
            # Proceed to submission
            # Extract metadata and proceed as usual
            # To allow re-submission, ensure it's not in the CSV
            # (i.e., remove from processed_images_set if needed)
            # For this script, we'll assume that if it's in the CSV, it's already been submitted
            # To resubmit, remove it from the CSV as per earlier instructions
            # Hence, here we skip processing as it's already in the CSV
            continue

        # Validate image
        if not validate_image(image_path):
            logging.warning(f"Image validation failed for {filename}, skipping.")
            images_skipped_due_to_critical_errors += 1
            continue

        if not os.path.exists(metadata_file):
            logging.warning(f"Metadata file not found for {filename}, skipping.")
            images_skipped_due_to_critical_errors += 1
            continue

        # Parse the custom metadata file
        metadata = parse_metadata_file(metadata_file)

        # Validate metadata and get processed_metadata
        processed_metadata = validate_metadata(metadata, metadata_file)
        if processed_metadata is False:
            logging.warning(f"Metadata validation failed for {filename}, skipping.")
            images_skipped_due_to_critical_errors += 1
            continue

        # Extract necessary fields from processed_metadata
        scientific_name = processed_metadata.get('scientificName', 'Unknown')
        species_key = processed_metadata.get('speciesKey', 'Unknown')
        decimal_latitude = processed_metadata.get('decimalLatitude', 'Unknown')
        decimal_longitude = processed_metadata.get('decimalLongitude', 'Unknown')
        coordinate_uncertainty = processed_metadata.get('coordinateUncertaintyInMeters', 'Unknown')
        continent = processed_metadata.get('continent', 'Unknown')
        state_province = processed_metadata.get('stateProvince', 'Unknown')
        event_date = processed_metadata.get('eventDate', 'Unknown')
        year = processed_metadata.get('year', np.nan)
        month = processed_metadata.get('month', np.nan)
        day = processed_metadata.get('day', np.nan)

        # Determine if date fields are valid
        try:
            date_fields_valid = not (np.isnan(year) or np.isnan(month) or np.isnan(day))
        except TypeError as e:
            logging.error(f"Type error when checking NaN for {filename}: {e}")
            date_fields_valid = False
            images_with_invalid_dates += 1
            invalid_date_metadata_images.append(os.path.abspath(metadata_file))

        if date_fields_valid:
            images_with_valid_dates += 1
        else:
            images_with_invalid_dates += 1

        # Load the input image
        try:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
        except (UnidentifiedImageError, IOError) as e:
            logging.error(f"Error opening image {filename}: {e}")
            images_skipped_due_to_critical_errors += 1
            continue

        # Perform segmentation
        segmentation_mask = segment_image(image)
        if segmentation_mask is None:
            logging.error(f"Segmentation failed for {filename}, skipping.")
            images_skipped_due_to_critical_errors += 1
            continue

        # Identify bird in the image
        unique_classes = np.unique(segmentation_mask)
        try:
            bird_class_index = COCO_CLASSES.index('bird')
        except ValueError:
            bird_class_index = None

        if bird_class_index is not None and bird_class_index in unique_classes:
            bird_mask = (segmentation_mask == bird_class_index).astype(np.uint8) * 255
            logging.info(f"Bird detected in {filename}.")
        else:
            logging.info(f"No bird detected in {filename}. Making a best guess.")
            best_guess_class = find_best_guess(segmentation_mask, unique_classes)
            if best_guess_class is not None:
                bird_mask = (segmentation_mask == best_guess_class).astype(np.uint8) * 255
                guessed_class = COCO_CLASSES[best_guess_class] if best_guess_class < len(
                    COCO_CLASSES) else f"Class {best_guess_class}"
                logging.info(f"Best guess for bird class in {filename}: {guessed_class}")
            else:
                logging.warning(f"Could not make a guess for {filename}, skipping.")
                images_skipped_due_to_critical_errors += 1
                continue

        # Ensure the mask has the same dimensions as the image
        if bird_mask.shape != image_np.shape[:2]:
            logging.warning(
                f"Mask size {bird_mask.shape} does not match image size {image_np.shape[:2]} for {filename}. Resizing mask.")
            bird_mask = cv2.resize(bird_mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            logging.info(f"Resized mask to {bird_mask.shape} for {filename}.")

        # Check if the mask has any non-zero pixels
        if not np.any(bird_mask):
            logging.info(f"No regions to inpaint in {filename}. Skipping inpainting.")
            # Optionally, decide how to handle images with no inpaint regions
            # For this guide, we'll proceed to send to Pl@ntNet API
            # Copy the original image to output directory
            try:
                image.save(output_image_path)
                logging.info(f"Original image copied to {output_image_path}.")
            except Exception as e:
                logging.error(f"Error copying image {filename} to output directory: {e}")
                images_skipped_due_to_critical_errors += 1
                continue
        else:
            # Inpaint the image to remove the bird (or best guess)
            try:
                inpainted_image = cv2.inpaint(image_np, bird_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                cv2.imwrite(output_image_path, cv2.cvtColor(inpainted_image, cv2.COLOR_RGB2BGR))
                logging.info(f"Inpainted image saved to {output_image_path}.")
            except Exception as e:
                logging.error(f"Error inpainting image {filename}: {e}")
                images_skipped_due_to_critical_errors += 1
                continue

        # Extract additional metadata fields
        additional_metadata = extract_additional_metadata(metadata_file)

        # Determine project ID based on latitude and longitude
        try:
            lat = float(decimal_latitude)
            lon = float(decimal_longitude)
            project_id = get_project_id(lat, lon)
        except (ValueError, TypeError):
            logging.error(f"Invalid latitude or longitude for {filename}. Using default project 'all'.")
            project_id = 'all'

        # Prepare the identification API endpoint with the obtained project ID
        identification_api_endpoint = API_ENDPOINT_IDENTIFY_TEMPLATE.format(project_id=project_id, api_key=API_KEY)

        # Send the (inpainted) image to Pl@ntNet API for plant identification
        try:
            with open(output_image_path, 'rb') as image_data:
                files = [('images', (filename, image_data))]
                # data = {'organs': ['flower']}  # Adjust organs as necessary
                data = {
                    'organs': ['flower'],
                   #  'lat': decimal_latitude,
                    # 'lon': decimal_longitude,
                    # 'no-reject': 'true',
                }
                response = requests.post(identification_api_endpoint, files=files, data=data)
        except Exception as e:
            logging.error(f"Error sending image {filename} to Pl@ntNet API: {e}")
            images_skipped_due_to_critical_errors += 1
            continue

        # Validate API response
        if response.status_code == 200:
            try:
                json_result = response.json()
            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding failed for {filename}: {e}")
                images_skipped_due_to_critical_errors += 1
                continue

            best_match = validate_api_response(json_result)
            if best_match is None:
                logging.warning(f"No valid results from Pl@ntNet for {filename}.")
                species_name = 'Unknown'
                gbif_id = 'Unknown'
                score = 0
            else:
                species_info = best_match.get('species', {})
                gbif_info = best_match.get('gbif') or {}  # Handle NoneType
                if not isinstance(gbif_info, dict):
                    logging.warning(f"'gbif' field is not a dict for {filename}.")
                    gbif_info = {}
                gbif_id = gbif_info.get('id', 'Unknown')
                species_name = species_info.get('scientificNameWithoutAuthor', 'Unknown')
                score = best_match.get('score', 0) * 100

                # Retrieve the scientific name from GBIF if needed
                if species_name == 'Unknown' and gbif_id != 'Unknown':
                    species_name = get_scientific_name_from_gbif(gbif_id)

            # Capture the current datetime for identification
            identification_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Add result to dictionary including metadata and image file name
            result = {
                'imageFileName': filename,
                'gbifIDPlant': gbif_id,
                'speciesPlant': species_name,
                'scorePlant': score,
                'scientificNameBird': scientific_name,
                'gbifIDBird': species_key,
                'decimalLatitude': decimal_latitude,
                'decimalLongitude': decimal_longitude,
                'coordinateUncertaintyInMeters': coordinate_uncertainty,
                'continent': continent,
                'stateProvince': state_province,
                'year': year,
                'month': month,
                'day': day,
                'eventDate': event_date,
                'dateFieldsValid': date_fields_valid,
                # Additional Metadata Fields
                'creator': additional_metadata.get('creator', 'Unknown'),
                'publisher': additional_metadata.get('publisher', 'Unknown'),
                'license': additional_metadata.get('license', 'Unknown'),
                'rightsHolder': additional_metadata.get('rightsHolder', 'Unknown'),
                'identifier': additional_metadata.get('identifier', 'Unknown'),
                # New Fields
                'projectID': project_id,
                'identificationDateTime': identification_datetime
            }

            # Append the result to the CSV immediately using to_csv with mode='a'
            try:
                # Check if CSV exists to determine if header is needed
                write_header = not os.path.exists(results_csv)
                # Ensure all required columns are present
                required_columns = [
                    'imageFileName', 'gbifIDPlant', 'speciesPlant', 'scorePlant',
                    'scientificNameBird', 'gbifIDBird', 'decimalLatitude',
                    'decimalLongitude', 'coordinateUncertaintyInMeters',
                    'continent', 'stateProvince', 'year', 'month', 'day',
                    'eventDate', 'dateFieldsValid', 'creator', 'publisher',
                    'license', 'rightsHolder', 'identifier',
                    'projectID', 'identificationDateTime'  # New Columns
                ]
                df_result = pd.DataFrame([result], columns=required_columns)
                df_result.to_csv(results_csv, mode='a', header=write_header, index=False)
                logging.info(f"Result for '{filename}' appended to {results_csv}.")
                # Add to processed_images_set to avoid reprocessing
                processed_images_set.add(filename)
            except Exception as e:
                logging.error(f"Error writing result for {filename} to CSV: {e}")
                images_skipped_due_to_critical_errors += 1
                continue

            # Increment the daily count
            config['count_today'] += 1
            total_images_processed += 1
            logging.info(f"Processed {filename}: {species_name} ({score:.2f}%)")

        else:
            logging.error(f"Pl@ntNet API error for {filename}: {response.status_code} - {response.text}")
            if response.status_code == 429:
                logging.error(f"Received 429 Too Many Requests. Stopping further processing.")
                break  # Stop processing further images
            images_skipped_due_to_critical_errors += 1
            continue

        # Update last_index after successful processing
        last_index = idx
        config['species_progress'][os.path.basename(output_dir)] = last_index
        save_config(config)

        # Clear CUDA cache periodically to free memory
        if idx % 10 == 0:  # Adjust the frequency as needed
            torch.cuda.empty_cache()
            logging.info("CUDA cache cleared.")

    return last_index

def load_processed_images(results_csv):
    processed_images_set = set()
    if os.path.exists(results_csv):
        try:
            existing_df = pd.read_csv(results_csv)
            if 'imageFileName' in existing_df.columns:
                processed_images_set = set(existing_df['imageFileName'].tolist())
                logging.info(f"Loaded {len(processed_images_set)} processed image(s) from {results_csv}.")
            else:
                logging.warning(
                    f"'imageFileName' column not found in {results_csv}. Assuming no images have been processed.")
        except Exception as e:
            logging.error(f"Error reading {results_csv}: {e}")
            # If reading fails, assume no images have been processed
            processed_images_set = set()
    return processed_images_set


if __name__ == "__main__":
    # === Addition Starts Here ===
    # Path to the separate API key config file
    api_config_file = '/home/exouser/pri/plantnet_api_config.json'

    # Load API key from the config file
    try:
        with open(api_config_file, 'r') as f:
            api_config = json.load(f)
        API_KEY = api_config.get('PLANTNET_API_KEY')
        if not API_KEY:
            logging.error("PLANTNET_API_KEY not found in the API config file.")
            exit(1)
        else:
            logging.info("PLANTNET_API_KEY loaded successfully from the config file.")
    except FileNotFoundError:
        logging.error(f"API config file not found at {api_config_file}. Please create the file with your API key.")
        exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from the API config file: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"Unexpected error loading API key: {e}")
        exit(1)
    # === Addition Ends Here ===

    # Optional: Set this variable to the species you want to process exclusively
    SPECIES_TO_PROCESS = None  # Set to None to process all species

    # Define priority lists
    first_priority_species = [
        'Selasphorus_calliope',
        'Calypte_anna',
        'Selasphorus_sasin',
        'Selasphorus_rufus',
        'Archilochus_alexandri',
        'Calypte_costae',
        'Selasphorus_platycercus',
        'Archilochus_colubris',
        'Amazilia_violiceps',
        'Cynanthus_latirostris',
        'Eugenes_fulgens'
    ]

    second_priority_species = [
        'Orthorhyncus_cristatus',
        'Chlorostilbon_maugaeus',
        'Anthracothorax_viridis',
        'Anthracothorax_dominicus',
        'Eulampis_holosericeusCan'
    ]

    # Base directories
    input_base_dir = '/home/exouser/pri/imagesTrochilidae/'
    output_base_dir = '/home/exouser/pri/imagesNoBirds/'
    # input_base_dir = './imagesRejected/'
    # output_base_dir = './imagesNoBirds2/'

    # Load or initialize config
    config = load_config()

    # Reset count_today if the date has changed
    today_str = datetime.now().strftime('%Y-%m-%d')
    if config.get('date') != today_str:
        config['date'] = today_str
        config['count_today'] = 0
        # Optionally, reset species_progress or keep it
        # Here, we assume species_progress is persistent across days
        logging.info("Config reset for a new day.")
    else:
        logging.info("Config loaded for today.")

    # Get all species subfolders
    try:
        all_subfolders = [
            name for name in os.listdir(input_base_dir)
            if os.path.isdir(os.path.join(input_base_dir, name))
        ]
        logging.info(f"Found {len(all_subfolders)} species folder(s) in {input_base_dir}.")
    except Exception as e:
        logging.error(f"Error listing subfolders in {input_base_dir}: {e}")
        all_subfolders = []

    # === Modification to Processing Order ===
    if SPECIES_TO_PROCESS:
        if SPECIES_TO_PROCESS in all_subfolders:
            processing_order = [SPECIES_TO_PROCESS]
            logging.info(f"Processing only the specified species: {SPECIES_TO_PROCESS}")
        else:
            logging.error(f"Specified species '{SPECIES_TO_PROCESS}' not found in {input_base_dir}. Exiting.")
            exit(1)
    else:
        # Define processing order
        processing_order = first_priority_species + second_priority_species
        # Add remaining species not in priority lists
        remaining_species = [s for s in all_subfolders if s not in processing_order]
        processing_order += remaining_species
        logging.info("Processing order defined for all species.")

    # Initialize completed_species list
    if os.path.exists(COMPLETED_FOLDERS_FILE):
        try:
            with open(COMPLETED_FOLDERS_FILE, 'r') as f:
                completed_species = [line.strip() for line in f if line.strip()]
            logging.info(f"Loaded {len(completed_species)} completed species from {COMPLETED_FOLDERS_FILE}.")
        except Exception as e:
            logging.error(f"Error reading {COMPLETED_FOLDERS_FILE}: {e}")
            completed_species = []
    else:
        completed_species = []

    # Process each species in order
    for species_name in processing_order:
        if species_name in completed_species:
            logging.info(f"Species '{species_name}' already completed. Skipping.")
            continue

        input_directory = os.path.join(input_base_dir, species_name)
        output_directory = os.path.join(output_base_dir, species_name)
        results_csv = f'/home/exouser/pri/identified_plants_{species_name}.csv'  # Added underscore for readability

        # Get last_index from config
        last_index = config['species_progress'].get(species_name, -1)

        # Load existing processed images to prevent duplicates
        processed_images_set = load_processed_images(results_csv)

        logging.info(f"Starting processing for species '{species_name}'.")

        # Process images and update processed_images_set
        last_index = process_and_identify_images(
            input_directory,
            output_directory,
            results_csv,
            last_index,
            processed_images_set,
            config
        )

        # Get total images in species folder
        try:
            total_images = len([
                f for f in os.listdir(input_directory)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
        except Exception as e:
            logging.error(f"Error counting images in {input_directory}: {e}")
            total_images = 0

        # Check if species is completed
        if os.path.exists(results_csv):
            try:
                results_df = pd.read_csv(results_csv)
                num_processed = len(results_df)
                if num_processed >= total_images:
                    logging.info(f"All images processed for species '{species_name}'. Marking as completed.")
                    completed_species.append(species_name)
                    # Remove species from species_progress
                    if species_name in config['species_progress']:
                        del config['species_progress'][species_name]
                    # Append to completed_folders.txt
                    try:
                        with open(COMPLETED_FOLDERS_FILE, 'a') as f:
                            f.write(f"{species_name}\n")
                        logging.info(f"Species '{species_name}' added to {COMPLETED_FOLDERS_FILE}.")
                    except Exception as e:
                        logging.error(f"Error writing to {COMPLETED_FOLDERS_FILE}: {e}")
            except Exception as e:
                logging.error(f"Error reading {results_csv} for completion check: {e}")

        # Save config after each species
        save_config(config)

        # Check if rate limit has been reached
        if config['count_today'] >= RATE_LIMIT:
            logging.warning("Daily rate limit reached. Stopping further processing for today.")
            break

    # After all processing is complete, generate summary report
    logging.info("=== Summary Report ===")
    logging.info(f"Total images processed: {total_images_processed}")
    logging.info(f"Images with valid date fields: {images_with_valid_dates}")
    logging.info(f"Images with invalid/missing date fields: {images_with_invalid_dates}")
    logging.info(f"Images skipped due to critical metadata issues: {images_skipped_due_to_critical_errors}")
    logging.info("======================")

    # Save the summary to a separate file
    try:
        with open(SUMMARY_LOG_FILE, 'w') as f:
            f.write("=== Summary Report ===\n")
            f.write(f"Total images processed: {total_images_processed}\n")
            f.write(f"Images with valid date fields: {images_with_valid_dates}\n")
            f.write(f"Images with invalid/missing date fields: {images_with_invalid_dates}\n")
            f.write(f"Images skipped due to critical metadata issues: {images_skipped_due_to_critical_errors}\n")
            f.write("======================\n")
        logging.info(f"Summary report saved to '{SUMMARY_LOG_FILE}'.")
    except Exception as e:
        logging.error(f"Error writing summary report to '{SUMMARY_LOG_FILE}': {e}")

    # Save list of images with invalid/missing date fields
    if invalid_date_metadata_images:
        try:
            with open(INVALID_DATE_LOG_FILE, 'w') as f:
                for image_path in invalid_date_metadata_images:
                    f.write(f"{image_path}\n")
            logging.info(
                f"List of images with invalid/missing 'year', 'month', or 'day' saved to '{INVALID_DATE_LOG_FILE}'.")
        except Exception as e:
            logging.error(f"Error writing invalid date metadata images to '{INVALID_DATE_LOG_FILE}': {e}")
    else:
        logging.info("No images with invalid/missing 'year', 'month', or 'day' were found.")
    # === Addition Ends Here ===
