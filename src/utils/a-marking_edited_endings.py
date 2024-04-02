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

def mark_differences_in_edited(original_ending, edited_endings):
    """Marks words in edited_endings not in original_ending, considering synonyms and semantic similarity."""
    original_words = set(original_ending.split())
    marked_edited_endings = []
    for sentence in edited_endings:
        marked_sentence = []
        for word in sentence.split():
            if word not in original_words and not any(are_synonyms_nltk(word, orig_word) or are_synonyms_spacy(word, orig_word) for orig_word in original_words):
                marked_sentence.append(f"<s>{word}</s>")  # Mark words not appearing in original_ending and not considered similar
            else:
                marked_sentence.append(word)
        marked_edited_endings.append(' '.join(marked_sentence))
    return marked_edited_endings

def process_file(input_file_path, output_file_path, output_csv_path):
    """Reads, processes, and writes the JSON data for a given file and generates a corresponding CSV file."""
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile, open(output_csv_path, 'w', newline='') as csv_outfile:
        fieldnames = ['story_id', 'premise', 'initial', 'counterfactual', 'original_ending', 'edited_ending_marked']
        writer = csv.DictWriter(csv_outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for line in infile:
            obj = json.loads(line)
            obj['edited_ending'] = mark_differences_in_edited(obj['original_ending'], obj['edited_ending'])
            json.dump(obj, outfile)
            outfile.write('\n')
            
            # Prepare data for CSV with edited_endings as the last column and original_ending just before it
            csv_row = {
                'story_id': obj['story_id'],
                'premise': obj['premise'],
                'initial': obj['initial'],
                'counterfactual': obj['counterfactual'],
                'original_ending': obj['original_ending'],
                'edited_ending_marked': ' '.join(obj['edited_ending']),  # Now marked edited_endings is a list of marked sentences
            }
            writer.writerow(csv_row)
    print(f"Processed files saved as {output_file_path} and {output_csv_path}")


# TODO: adjust the files path
file_paths = [
    ("dev_data.json", "dev_data_marked_edited.json", "dev_data_marked_edited.csv"),
    ("test_data.json", "test_data_marked_edited.json", "test_data_marked_edited.csv"),
    ("train_supervised_small.json", "train_supervised_small_marked_edited.json", "train_supervised_small_marked_edited.csv")
]

# Sample usage
#file_paths = [
#    ("dev_data_sample.json", "dev_data_sample_marked_edited.json", "dev_data_sample_marked_edited.csv"),
#    ("test_data_sample.json", "test_data_sample_marked_edited.json", "test_data_sample_marked_edited.csv"),
#    ("train_supervised_small_sample.json", "train_supervised_small_sample_marked_edited.json", "train_supervised_small_sample_marked_edited.csv")
#]

for input_path, output_json_path, output_csv_path in file_paths:
    process_file(input_path, output_json_path, output_csv_path)
