# This file is used to create sample data files. It removes duplicates: multiple edited endings
import json
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_unique_stories(input_file, output_file):
    """
    Extracts unique stories from a JSON lines file based on 'story_id' and writes them to another file.

    Parameters:
    - input_file: Path to the input file containing stories in JSON lines format.
    - output_file: Path to the output file where unique stories will be written.
    """
    seen_story_ids = set()
    unique_stories = []

    try:
        with open(input_file, 'r') as infile:
            for line in infile:
                try:
                    story = json.loads(line)
                    if story['story_id'] not in seen_story_ids:
                        seen_story_ids.add(story['story_id'])
                        unique_stories.append(story)
                except json.JSONDecodeError:
                    logging.error(f"Error decoding JSON from line in file {input_file}")
    except FileNotFoundError:
        logging.error(f"File not found: {input_file}")
        return

    with open(output_file, 'w') as outfile:
        for story in unique_stories:
            json_line = json.dumps(story)
            outfile.write(json_line + '\n')
    
    logging.info(f"Extracted {len(unique_stories)} unique stories from {input_file} to {output_file}.")

# Example usage
file_pairs = [
    ('dev_data.json', 'dev_data_sample.json'),
    ('test_data.json', 'test_data_sample.json'),
    ('train_supervised_small.json', 'train_supervised_small_sample.json')
]

for input_file, output_file in file_pairs:
    extract_unique_stories(input_file, output_file)

logging.info("Done extracting unique stories based on story_id.")
