import os
import requests
import json
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision
from PIL import Image
import pandas as pd
from pygbif import species

# Pl@ntNet API details
API_KEY = "2b10SRlISJG0KaNZCAHV6k8hz"  # Replace with your actual API key
PROJECT = "all"
api_endpoint = f"https://my-api.plantnet.org/v2/identify/{PROJECT}?api-key={API_KEY}"

# Load the COCO class labels
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Load a pre-trained segmentation model
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# Define a transformation to preprocess the input image
transform = T.Compose([
    T.ToTensor(),  # Convert image to tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Function to perform segmentation
def segment_image(image):
    # Ensure the image is in RGB mode (i.e., it has 3 channels)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    return output_predictions

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
    with open(metadata_file, 'r') as file:
        for line in file:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                metadata[key.strip()] = value.strip()
    return metadata

# Function to get the scientific name from GBIF using the gbifID
def get_scientific_name_from_gbif(gbif_id):
    if gbif_id != 'Unknown':
        try:
            result = species.name_usage(key=gbif_id)
            return result.get('scientificName', 'Unknown')
        except Exception as e:
            print(f"Error retrieving scientific name for GBIF ID {gbif_id}: {str(e)}")
            return 'Unknown'
    return 'Unknown'

# Function to process images, remove birds, and identify plants
def process_and_identify_images(input_dir, output_dir, results_csv):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_list = []

    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            try:
                image_path = os.path.join(input_dir, filename)
                output_image_path = os.path.join(output_dir, filename)
                metadata_file = image_path.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')

                if not os.path.exists(metadata_file):
                    print(f"Metadata file not found for {filename}, skipping.")
                    continue

                # Parse the custom metadata file
                metadata = parse_metadata_file(metadata_file)

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
                image = Image.open(image_path)
                image_np = np.array(image)

                # Perform segmentation
                segmentation_mask = segment_image(image)

                # Identify bird in the image
                unique_classes = np.unique(segmentation_mask)
                bird_class_index = COCO_CLASSES.index('bird') if 'bird' in COCO_CLASSES else None

                if bird_class_index is not None and bird_class_index in unique_classes:
                    bird_mask = (segmentation_mask == bird_class_index).astype(np.uint8) * 255
                    print(f"Bird detected in {filename}.")
                else:
                    print(f"No bird detected in {filename}. Making a best guess.")
                    best_guess_class = find_best_guess(segmentation_mask, unique_classes)
                    if best_guess_class is not None:
                        bird_mask = (segmentation_mask == best_guess_class).astype(np.uint8) * 255
                        print(f"Best guess for bird class in {filename}: {COCO_CLASSES[best_guess_class]}")
                    else:
                        print(f"Could not make a guess for {filename}, skipping.")
                        continue

                # Inpaint the image to remove the bird (or best guess)
                inpainted_image = cv2.inpaint(image_np, bird_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                cv2.imwrite(output_image_path, cv2.cvtColor(inpainted_image, cv2.COLOR_RGB2BGR))

                # Send the inpainted image to Pl@ntNet API for plant identification
                with open(output_image_path, 'rb') as image_data:
                    files = [('images', (filename, image_data))]
                    data = {'organs': ['flower']}  # Adjust organs as necessary
                    response = requests.post(api_endpoint, files=files, data=data)

                    if response.status_code == 200:
                        json_result = response.json()
                        best_match = json_result.get('results', [{}])[0]
                        species_name = best_match.get('species', {}).get('scientificNameWithoutAuthor', 'Unknown')
                        gbif_id = best_match.get('gbif', {}).get('id', 'Unknown')
                        score = best_match.get('score', 0) * 100

                        # Retrieve the scientific name from GBIF if needed
                        if species_name == 'Unknown' and gbif_id != 'Unknown':
                            species_name = get_scientific_name_from_gbif(gbif_id)
                        # Add result to list including metadata and image file name
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
                        results_list.append(result)
                        print(f"Processed {filename}: {species_name} ({score:.2f}%)")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    # Convert results to a DataFrame and save as CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(results_csv, index=False)
    print(f"Processing completed. Results saved to {results_csv}")


# Directory paths
input_directory = './images/test2/'  # Replace with your input directory containing images
output_directory = './imagesNoBirds/CalpteAnna/'  # Replace with your output directory to save processed images
results_csv = './identified_plantsCalpteAnna2.csv'  # Path to save the CSV results

# Process all images in the directory and save results
process_and_identify_images(input_directory, output_directory, results_csv)
