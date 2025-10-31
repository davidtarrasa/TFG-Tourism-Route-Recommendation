import requests
import json
import os
from dotenv import load_dotenv
from tqdm import tqdm

city = "osaka"
# Text file containing the fsq_ids (one per line)
input_file = f"data/raw/POIS_{city}_ids.txt"  # Replace with your file name containing one Foursquare ID per line

# File to save all results
output_file = f"data/processed/foursquare/POIS_{city}_api.json"

# File to log errors (Foursquare ids with no information)
error_file = f"data/errors/Errors_POIS_{city}.txt"


load_dotenv()  # Esto lee .env y exporta las variables al entorno

API_KEY = os.getenv("FOURSQUARE_API_KEY")
if not API_KEY:
    raise RuntimeError("FOURSQUARE_API_KEY no está definida. Añádela a .env")

# Request headers (NEW VERSION)
headers = {
    "accept": "application/json",
    "X-Places-Api-Version": "2025-06-17",
    "authorization": f"Bearer {API_KEY}"
}

def _count_nonempty_lines(path):
    try:
        with open(path, 'r') as _f:
            return sum(1 for _line in _f if _line.strip())
    except FileNotFoundError:
        return None

if __name__ == "__main__":
    total_ids = _count_nonempty_lines(input_file)
    results = []
    #max_pois = 1
    # Read the text file line by line
    with open(error_file, 'w') as error_log:
        with open(input_file, 'r') as f:
            for idx, line in enumerate(f):
                # --- PROGRESS BAR (cambio mínimo) ---
                iterable = f
                if tqdm is not None and total_ids:
                    iterable = tqdm(f, total=total_ids, desc="Descargando POIs", unit="id")
                # --- FIN PROGRESS BAR ---
                #if idx >= max_pois:
                #    break
                fsq_id = line.strip()
                if not fsq_id:
                    continue

                if fsq_id:  # Ensure it's not an empty line
                
                    # New endpoint Pro places (not premium!)
                    url = f"https://places-api.foursquare.com/places/{fsq_id}?fields=fsq_place_id%2Cname%2Clatitude%2Clongitude%2Ccategories%2Cchains%2Cdate_closed%2Clocation"


                    # Make the request to the Foursquare API
                    response = requests.get(url, headers=headers)

                    # Check if the request was successful
                    if response.status_code == 200:
                        # Convert the response to JSON and add it to the results list
                        data = response.json()
                        results.append(data)
                    else:
                        # If the request fails, log the fsq_id in the error file
                        print(fsq_id)
                        print(response)
                        error_log.write(fsq_id + '\n')

    # Save all results to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f"All results have been saved to {output_file}")
