import json
import spacy
import nltk
import csv
from nltk.corpus import wordnet as wn

# Load spaCy's large English Language model
nlp = spacy.load("en_core_web_lg")

# Ensure NLTK data is available
nltk.download('wordnet')
nltk.download('omw-1.4')

def are_synonyms_spacy(word1, word2):
    """Check for semantic similarity using spaCy."""
    token1 = nlp(word1)
    token2 = nlp(word2)
    return token1.similarity(token2) > 0.75  # Adjust threshold as needed

def are_synonyms_nltk(word1, word2):
    """Check if two words are synonyms using WordNet."""
    word1_synsets = wn.synsets(word1)
    word2_synsets = wn.synsets(word2)
    return any(s1 == s2 for s1 in word1_synsets for s2 in word2_synsets)

def mark_differences(original_ending, edited_endings):
    """Marks words in original_ending not in edited_endings, considering synonyms and semantic similarity."""
    edited_words = set(word for sentence in edited_endings for word in sentence.split())
    marked_original = []
    for word in original_ending.split():
        if word not in edited_words and not any(are_synonyms_nltk(word, edited_word) or are_synonyms_spacy(word, edited_word) for edited_word in edited_words):
            marked_original.append(f"<s>{word}</s>")  # Mark words not appearing in edited_endings and not considered similar
        else:
            marked_original.append(word)
    return ' '.join(marked_original)

def process_file(input_file_path, output_file_path, output_csv_path):
    """Reads, processes, and writes the JSON data for a given file and generates a corresponding CSV file."""
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile, open(output_csv_path, 'w', newline='') as csv_outfile:
        fieldnames = ['story_id', 'premise', 'initial', 'counterfactual', 'original_ending', 'edited_ending_marked']
        writer = csv.DictWriter(csv_outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for line in infile:
            obj = json.loads(line)
            obj['original_ending'] = mark_differences(obj['original_ending'], obj['edited_ending'])
            json.dump(obj, outfile)
            outfile.write('\n')
            
            # Prepare data for CSV
            csv_row = {
                'story_id': obj['story_id'],
                'premise': obj['premise'],
                'initial': obj['initial'],
                'counterfactual': obj['counterfactual'],
                'original_ending': obj['original_ending'],
                'edited_ending_marked': ' '.join(obj['edited_ending'])  # Assuming edited_ending is a list of sentences
            }
            writer.writerow(csv_row)
    print(f"Processed files saved as {output_file_path} and {output_csv_path}")

# TODO: adjust the files path
file_paths = [
    ("dev_data_sample.json", "dev_data_sample_marked_original.json", "dev_data_sample_marked_original.csv"),
    ("test_data_sample.json", "test_data_sample_marked_original.json", "test_data_sample_marked_original.csv"),
    ("train_supervised_small_sample.json", "train_supervised_small_sample_marked_original.json", "train_supervised_small_sample_marked_original.csv")
]

for input_path, output_json_path, output_csv_path in file_paths:
    process_file(input_path, output_json_path, output_csv_path)
