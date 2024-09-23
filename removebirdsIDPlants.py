import os
import requests
import json
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet101_Weights
from PIL import Image, UnidentifiedImageError
import pandas as pd
from pygbif import species
from datetime import datetime
import logging

# Configuration Constants
API_KEY = os.getenv('PLANTNET_API_KEY',
                    "2b10SRlISJG0KaNZCAHV6k8hz")  # Replace with your actual API key or set the environment variable
PROJECT = "all"
API_ENDPOINT = f"https://my-api.plantnet.org/v2/identify/{PROJECT}?api-key={API_KEY}"
CONFIG_FILE = 'config.json'
RATE_LIMIT = 5000  # Maximum identifications per day
COMPLETED_FOLDERS_FILE = 'completed_folders.txt'

# Set up logging with rotation
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler('plant_identifier.log', maxBytes=5 * 1024 * 1024,
                              backupCount=5)  # 5 MB per file, keep 5 backups
logging.basicConfig(
    handlers=[handler],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
    # Updated to use 'weights' parameter instead of 'pretrained'
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights).to(device).eval()
    logging.info("Segmentation model loaded and moved to device successfully.")
except Exception as e:
    logging.error(f"Error loading segmentation model: {e}")
    raise

# Define a transformation to preprocess the input image
transform = T.Compose([
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
        return config


def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        logging.info("Configuration saved successfully.")
    except Exception as e:
        logging.error(f"Error saving config file: {e}")


# Function to validate metadata fields
def validate_metadata(metadata):
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

    for field, field_type in required_fields.items():
        value = metadata.get(field)
        if value is None:
            errors.append(f"Missing required metadata field: {field}")
            continue
        if value == 'Unknown':
            errors.append(f"Metadata field '{field}' is 'Unknown'")
            continue
        # Type checking and conversion
        try:
            if field_type == float:
                metadata[field] = float(value)
            elif field_type == int:
                metadata[field] = int(value)
            elif field_type == str:
                metadata[field] = str(value)
        except ValueError:
            errors.append(f"Metadata field '{field}' has invalid type. Expected {field_type.__name__}.")

    # Additional validation for latitude and longitude ranges
    lat = metadata.get('decimalLatitude')
    lon = metadata.get('decimalLongitude')
    if isinstance(lat, float):
        if not (-90.0 <= lat <= 90.0):
            errors.append(f"Invalid latitude value: {lat}. Must be between -90 and 90.")
    if isinstance(lon, float):
        if not (-180.0 <= lon <= 180.0):
            errors.append(f"Invalid longitude value: {lon}. Must be between -180 and 180.")

    if errors:
        for error in errors:
            logging.error(error)
        return False
    return True


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
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
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


# Function to process images, remove birds, and identify plants
def process_and_identify_images(input_dir, output_dir, results_csv, last_index, processed_images_set):
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

        # Validate image
        if not validate_image(image_path):
            logging.warning(f"Image validation failed for {filename}, skipping.")
            continue

        if not os.path.exists(metadata_file):
            logging.warning(f"Metadata file not found for {filename}, skipping.")
            continue

        # Parse the custom metadata file
        metadata = parse_metadata_file(metadata_file)

        # Validate metadata
        if not validate_metadata(metadata):
            logging.warning(f"Metadata validation failed for {filename}, skipping.")
            continue

        # Extract necessary fields from metadata
        scientific_name = metadata.get('scientificName', 'Unknown')
        species_key = metadata.get('speciesKey', 'Unknown')
        decimal_latitude = metadata.get('decimalLatitude', 'Unknown')
        decimal_longitude = metadata.get('decimalLongitude', 'Unknown')
        coordinate_uncertainty = metadata.get('coordinateUncertaintyInMeters', 'Unknown')
        continent = metadata.get('continent', 'Unknown')
        state_province = metadata.get('stateProvince', 'Unknown')
        year = metadata.get('year', 'Unknown')
        month = metadata.get('month', 'Unknown')
        day = metadata.get('day', 'Unknown')
        event_date = metadata.get('eventDate', 'Unknown')

        # Load the input image
        try:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)
        except (UnidentifiedImageError, IOError) as e:
            logging.error(f"Error opening image {filename}: {e}")
            continue

        # Perform segmentation
        segmentation_mask = segment_image(image)
        if segmentation_mask is None:
            logging.error(f"Segmentation failed for {filename}, skipping.")
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
                continue

        # Inpaint the image to remove the bird (or best guess)
        try:
            inpainted_image = cv2.inpaint(image_np, bird_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            cv2.imwrite(output_image_path, cv2.cvtColor(inpainted_image, cv2.COLOR_RGB2BGR))
            logging.info(f"Inpainted image saved to {output_image_path}.")
        except Exception as e:
            logging.error(f"Error inpainting image {filename}: {e}")
            continue

        # Send the inpainted image to Pl@ntNet API for plant identification
        try:
            with open(output_image_path, 'rb') as image_data:
                files = [('images', (filename, image_data))]
                data = {'organs': ['flower']}  # Adjust organs as necessary
                response = requests.post(API_ENDPOINT, files=files, data=data)
        except Exception as e:
            logging.error(f"Error sending image {filename} to Pl@ntNet API: {e}")
            continue

        # Validate API response
        if response.status_code == 200:
            try:
                json_result = response.json()
            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding failed for {filename}: {e}")
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
                'eventDate': event_date
            }

            # Append the result to the CSV immediately
            try:
                if os.path.exists(results_csv):
                    results_df = pd.read_csv(results_csv)
                    results_df = results_df.append(result, ignore_index=True)
                else:
                    results_df = pd.DataFrame([result])
                results_df.to_csv(results_csv, index=False)
                logging.info(f"Result for '{filename}' appended to {results_csv}.")
                # Add to processed_images_set to avoid reprocessing
                processed_images_set.add(filename)
            except Exception as e:
                logging.error(f"Error writing result for {filename} to CSV: {e}")
                continue

            # Increment the daily count
            config['count_today'] += 1
            logging.info(f"Processed {filename}: {species_name} ({score:.2f}%)")

        else:
            logging.error(f"Pl@ntNet API error for {filename}: {response.status_code} - {response.text}")
            continue

        # Update last_index after successful processing
        last_index = idx
        config['species_progress'][os.path.basename(output_dir)] = last_index
        save_config(config)


def load_processed_images(results_csv):
    processed_images_set = set()
    if os.path.exists(results_csv):
        try:
            existing_df = pd.read_csv(results_csv)
            processed_images_set = set(existing_df['imageFileName'].tolist())
            logging.info(f"Loaded {len(processed_images_set)} processed image(s) from {results_csv}.")
        except Exception as e:
            logging.error(f"Error reading {results_csv}: {e}")
            # If reading fails, assume no images have been processed
            processed_images_set = set()
    return processed_images_set


if __name__ == "__main__":
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
    input_base_dir = './imagesTrochilidae/'
    output_base_dir = './imagesNoBirds/'

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

    # Define processing order
    processing_order = first_priority_species + second_priority_species
    # Add remaining species not in priority lists
    remaining_species = [s for s in all_subfolders if s not in processing_order]
    processing_order += remaining_species

    logging.info("Processing order defined.")

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
        results_csv = f'./identified_plants{species_name}.csv'

        # Get last_index from config
        last_index = config['species_progress'].get(species_name, -1)

        # Load existing processed images to prevent duplicates
        processed_images_set = load_processed_images(results_csv)

        logging.info(f"Starting processing for species '{species_name}'.")

        # Process images and update processed_images_set
        process_and_identify_images(
            input_directory,
            output_directory,
            results_csv,
            last_index,
            processed_images_set
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

    logging.info("Image processing and plant identification completed.")
