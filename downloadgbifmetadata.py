import os
import json
import pandas as pd
from pygbif import occurrences as occ
import time

# Load credentials from the JSON file
with open('credentials.json', 'r') as file:
    credentials = json.load(file)

USERNAME = credentials['USERNAME']
PASSWORD = credentials['PASSWORD']
EMAIL = credentials['EMAIL']

from pygbif import species

# List of species names
species_names = [
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
    "Anthracothorax dominicus",
    "Orthorhyncus cristatus",
    "Anthracothorax viridis",
    "Eulampis holosericeus",
    "Riccordia maugaeus"
]

# Function to get taxon key
def get_taxon_key(name):
    result = species.name_backbone(name=name)
    return result.get('usageKey')

# Get taxon keys for all species
taxon_keys = {get_taxon_key(name): name for name in species_names}
print(taxon_keys)
taxon_keys = {
    2476675: "Selasphorus sasin",
    2476676: "Selasphorus rufus",
    5228513: "Archilochus alexandri",
    2476675: "Calypte costae",
    7597244: "Selasphorus calliope",
    2476680: "Selasphorus platycercus",
    5228514: "Archilochus colubris",
    2476462: "Amazilia violiceps",
    5228542: "Cynanthus latirostris",
    2476108: "Eugenes fulgens",
    2476715: "Anthracothorax dominicus",
    2476284: "Orthorhyncus cristatus",
    2476728: "Anthracothorax viridis",
    2476399: "Eulampis holosericeus",
    11091395: "Riccordia maugaeus"
}

# Function to save dataframe to CSV with retries
def save_to_csv_with_retries(df, filename, max_retries=5):
    for attempt in range(max_retries):
        try:
            df.to_csv(filename, index=False, sep='\t')
            return
        except OSError as e:
            print(f"Error saving {filename}: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to save {filename} after {max_retries} attempts")

# Function to download GBIF metadata
def download_gbif_metadata():
    for taxon_key, species_name in taxon_keys.items():
        all_records = []
        offset = 0
        limit = 300
        while True:
            response = occ.search(
                taxonKey=taxon_key,
                country='US',
                hasCoordinate=True,
                hasGeospatialIssue=False,
                mediatype='StillImage',
                limit=limit,
                offset=offset
            )

            results = response.get('results', [])
            all_records.extend(results)

            if len(results) < limit:
                break

            offset += limit

        df = pd.DataFrame(all_records)
        filename = f"./metadata/gbifmultimedia_{taxon_key}.txt"
        save_to_csv_with_retries(df, filename)
        print(f"Downloaded metadata for {species_name} with taxon_key {taxon_key}")

# Download the metadata and save to files
download_gbif_metadata()
print("Downloaded metadata for all specified species.")
