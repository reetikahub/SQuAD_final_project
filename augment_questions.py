import copy
import json
from tqdm import tqdm
import spacy
import random
import nltk
from nltk.corpus import wordnet


def detect_named_entities(original_text, back_translated_text, detector):
    original_detection = detector(original_text)
    back_translated_detection = detector(back_translated_text)

    original_ner = set([x.text for x in original_detection.ents])
    back_translated_ner = set([x.text for x in back_translated_detection.ents])

    return original_ner == back_translated_ner


def get_translations(language):
    translations_file = './data/{0}-translations.json'.format(language)
    with open(translations_file, "r") as fh:
        translations = json.load(fh)
    return translations


def shuffle_text(original_text):
    text_split = original_text.split(' ')
    random.shuffle(text_split)
    return ' '.join(text_split)


# Return True with Probability
def check_probability(probability):
    return random.random() < probability/100


# Replace Word in Original Sentence with a Synonym, with given Probability
def replace_with_synonym(original_text, detector, probability):
    parsed_sentence = detector(original_text)
    new_sentence = []
    for token in parsed_sentence:
        new_word = token
        # Only Replace Words that are Noun, Verb, Adjective, or Adverb
        if token.pos_ in ('NOUN', 'VERB', 'ADJ', 'ADV') and not token.is_stop:
            # Check Probability
            if check_probability(probability=probability):
                replaced = False
                synonyms = wordnet.synsets(token.text)
                for synonym in synonyms:
                    for syn in synonym.lemmas():
                        if syn.name() != token.text:
                            new_word = syn.name().replace('_', '')
                            replaced = True
                        if replaced:
                            break
                    if replaced:
                        break
        new_sentence.append('{0}{1}'.format(new_word, token.whitespace_))
    return ''.join(new_sentence)


def augment_data(input_file, translations, detector, probabilities):

    cnt = 0
    limit = 5

    output_data = {
        "data": []
    }

    question_index = 0

    with open(input_file, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            data_json = {
                "title": article["title"],
                "paragraphs": []
            }
            for para in article["paragraphs"]:
                print("-- Progress: Question #: {0}".format(question_index))
                para_json = {
                    "context": para["context"],
                    "qas": []
                }
                for qa in para["qas"]:
                    # cnt += 1  # toDo: Comment for Dev Purposes
                    question_counter = 0
                    original_question = qa['question']

                    # Write Original Data
                    qa_json = copy.copy(qa)
                    qa_json['technique'] = 'Original'
                    qa_json['question'] = original_question
                    qa_json['original'] = original_question
                    qa_json['named_entities_match'] = True
                    qa_json['id'] = '{0}{1}'.format(qa['id'], question_counter)
                    question_counter += 1
                    para_json['qas'].append(qa_json)

                    # Write Augmented Data with Back Translation from different Languages:
                    for language in translations:
                        back_translated_question = translations[language][str(question_index)]
                        # translate_text(original_text=original_question, translators=translators[language])
                        qa_json = copy.copy(qa)
                        qa_json['technique'] = language
                        qa_json['question'] = back_translated_question
                        qa_json['original'] = original_question
                        qa_json['named_entities_match'] = detect_named_entities(original_text=original_question,
                                                                                back_translated_text=back_translated_question,
                                                                                detector=detector)
                        qa_json['id'] = '{0}{1}'.format(qa['id'], question_counter)
                        question_counter += 1
                        para_json['qas'].append(qa_json)

                    # Write Augmented Data with Shuffled Sentence:
                    qa_json = copy.copy(qa)
                    qa_json['technique'] = 'Shuffle'
                    qa_json['question'] = shuffle_text(original_text=original_question)
                    qa_json['original'] = original_question
                    qa_json['named_entities_match'] = True
                    qa_json['id'] = '{0}{1}'.format(qa['id'], question_counter)
                    question_counter += 1
                    para_json['qas'].append(qa_json)

                    # Write Augmented Data with Probabilities
                    for probability in probabilities:
                        qa_json = copy.copy(qa)
                        qa_json['technique'] = 'Synonym_{0}'.format(probability)
                        qa_json['question'] = replace_with_synonym(original_text=original_question,
                                                                   detector=detector,
                                                                   probability=probability)
                        qa_json['original'] = original_question
                        qa_json['named_entities_match'] = True
                        qa_json['id'] = '{0}{1}'.format(qa['id'], question_counter)
                        question_counter += 1
                        para_json['qas'].append(qa_json)

                    question_index += 1
                    if cnt >= limit:
                        break
                data_json['paragraphs'].append(para_json)
                if cnt >= limit:
                    break
            output_data['data'].append(data_json)
            if cnt >= limit:
                break
    print("Outputting Json:")
    # print(json.dumps(output_data, indent=4))  #toDo: Comment for Dev Purposes
    return output_data


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)


if __name__ == '__main__':
    print('- Running Data Augmentation Pipeline:')

    # toDo: Move to Parameter Arguments
    # toDo: Add error handling/ cleanup for Argument Values

    train_file = './data/train-v2.0.json'
    augmented_file = './data/train-augmented-v2.0.json'
    # translate_languages = 'French'  #toDo: Comment out
    translate_languages = 'French,Spanish,Chinese'
    synonym_probabilities = '30,60'

    # Download NLTK Dependencies
    nltk.download('wordnet')
    nltk.download('punkt')

    # Setup NER Detector
    spacy.prefer_gpu()
    entity_detector = spacy.load('en_core_web_trf')

    # Setup Translators
    language_translations = {}
    for translate_language in translate_languages.split(','):
        language_translations[translate_language] = get_translations(language=translate_language)

    # Setup Probabilities:
    probabilities_list = []
    for prob in synonym_probabilities.split(','):
        probabilities_list.append(int(prob))

    # Augment File
    augmented_data = augment_data(input_file=train_file,
                                  translations=language_translations,
                                  detector=entity_detector,
                                  probabilities=probabilities_list)

    save(filename=augmented_file,
         obj=augmented_data,
         message='Augmented Data')


