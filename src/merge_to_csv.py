import os
import pandas as pd
from tqdm import tqdm
import sys

SOURCE_DIR = '../results_als_general3/'       # file with all .txt articles
OUTPUT_CSV = '../data/corpus_als_general_pmc3.csv'
TEXT_COLUMN = 'text'
YEAR_COLUMN = 'year'


def find_all_txt_files(directory):

    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_paths.append(os.path.join(root, file))
    return file_paths


def extract_year_from_filename(filename):
 
    basename = os.path.basename(filename)
    year_match = basename[:4]  # the year is the first 4 characters
    try:
        return int(year_match)
    except ValueError:
        return None


def main():
    print(f"Starting merging txt files from: {SOURCE_DIR}")

    if not os.path.exists(SOURCE_DIR):
        print(f"ERROR: Source folder not found: {SOURCE_DIR}")
        sys.exit()


    file_paths = find_all_txt_files(SOURCE_DIR)
    
    if not file_paths:
        print(f"ERROR: No txt file found in {SOURCE_DIR}")
        sys.exit()
        
    print(f"Found {len(file_paths)} .txt files to process.")

    # reads each file, extracts year and store the abstract
    all_data = []
    for path in tqdm(file_paths, desc="Reading files"):
        year = extract_year_from_filename(path)
        if year is None:
            continue  
        try:
            with open(path, 'r', encoding='utf-8') as infile:
                content = infile.read()
                content = ' '.join(content.replace('\n', ' ').split()) 
                if content:
                    all_data.append({TEXT_COLUMN: content, YEAR_COLUMN: year})
        except Exception as e:
            print(f"Error when reading file {path}: {e}")

    print(f"Processed {len(all_data)} valid abstracts.")


    if not all_data:
        print("No text was read successfully. Exiting.")
        sys.exit()

    df = pd.DataFrame(all_data)

    output_dir = os.path.dirname(OUTPUT_CSV)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving CSV file in: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False, escapechar='\\')

    print("\nFinished.")
    print(f"CSV file is ready in: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
