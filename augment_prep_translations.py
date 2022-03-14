import json
from tqdm import tqdm
from transformers import pipeline
import time
import torch
from os import listdir


# Get Translators for English to Target Language and back to English
def get_translators(language):
    if language == 'French':
        trans_model = 'Helsinki-NLP/opus-mt-en-fr'
        back_trans_model = 'Helsinki-NLP/opus-mt-fr-en'
    elif language == 'Spanish':
        trans_model = 'Helsinki-NLP/opus-mt-en-es'
        back_trans_model = 'Helsinki-NLP/opus-mt-es-en'
    elif language == 'Chinese':
        trans_model = 'Helsinki-NLP/opus-mt-en-zh'
        back_trans_model = 'Helsinki-NLP/opus-mt-zh-en'
    elif language == 'Hindi':
        trans_model = 'Helsinki-NLP/opus-mt-en-hi'
        back_trans_model = 'Helsinki-NLP/opus-mt-hi-en'
    elif language == 'Arabic':
        trans_model = 'Helsinki-NLP/opus-mt-en-ar'
        back_trans_model = 'Helsinki-NLP/opus-mt-ar-en'
    else:
        raise Exception("Translation not supported for: {0}".format(language))

    return pipeline(model=trans_model, device=0, batch_size=16), pipeline(model=back_trans_model, device=0, batch_size=16)


def translate_text_batch(original_batch, batch_id, translators):
    source_translator, target_translator = translators
    source_translations = [x['translation_text'] for x in source_translator(original_batch)]
    target_translations = [x['translation_text'] for x in target_translator(source_translations)]
    target_dict = {}
    for i, v in enumerate(target_translations):
        target_dict[i+batch_id] = v
    return target_dict


def collect_questions(input_file):

    cnt = 0
    limit = 100
    questions_cnt = 0

    questions_list = []

    with open(input_file, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                for qa in para["qas"]:
                    #cnt += 1  # toDo: Comment out to run in Dev Mode
                    questions_cnt += 1
                    questions_list.append(qa['question'])
                    if cnt >= limit:
                        break
                if cnt >= limit:
                    break
            if cnt >= limit:
                break
    print("- Total Questions Collected: {0}".format(questions_cnt))
    return questions_list


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def merge_files(input_directory, output_file):
    print("-- Merging Records from Directory: {0} to File: {1}".format(input_directory, output_file))
    output_json = {}
    # Get List of all Json files in directory:
    file_names = listdir(input_directory)
    for file_name in file_names:
        if file_name.endswith('-questions.json'):
            merge_file = '{0}/{1}'.format(input_directory, file_name)
            with open(merge_file, "r") as fh:
                questions = json.load(fh)
                for question_id in questions:
                    output_json[question_id] = questions[question_id]
    print("-- Writing # Questions : {0} to {1}".format(len(output_json.keys()), output_file))
    save(output_file, output_json, "Merging File")


if __name__ == '__main__':

    start_time = time.time()
    print('- Running Data Augmentation Pipeline:')

    # toDo: Move to Arguments
    # toDo: Add error handling/ cleanup for Argument Values
    train_file = './data/train-v2.0.json'
    augmented_file = './data/train-augmented-v2.0.json'
    translated_file = './data/Translations/{0}/{1}-questions.json'
    translated_directory = './data/Translations/{0}'
    final_translated_file = './data/{0}-translations.json'
    translate_languages = 'French,Spanish,Chinese'
    # translate_languages = 'French,Spanish,Chinese,Hindi,Arabic'
    synonym_probabilities = '30,60'

    # toDo: Create Directories for Translations

    # Setup Translators
    language_translators = {}
    for translate_language in translate_languages.split(','):
        language_translators[translate_language] = get_translators(language=translate_language)
        language_translators[translate_language] = get_translators(language=translate_language)

    language_load_time = time.time()
    print("Done Loading Translators: {0}".format(language_load_time - start_time))

    # Collect List of Questions:
    original_questions = collect_questions(input_file=train_file)
    batch_size = 1000
    # Translate Languages
    for language in language_translators:
        # Run in Batches of 10K
        for batch in range(0, len(original_questions), batch_size):
            with torch.no_grad():
                print('- {0}, Batch: {1} to {2}'.format(language, batch, batch+batch_size))
                language_translate_start_time = time.time()
                back_translated_batch = translate_text_batch(original_batch=original_questions[batch: batch+batch_size],
                                                             batch_id=batch,
                                                             translators=language_translators[language])
                language_translate_end_time = time.time()
                print("-- Translating Time: {0}: {1}".format(language, language_translate_end_time - language_translate_start_time))
                save(filename=translated_file.format(language, batch),
                     obj=back_translated_batch,
                     message='{0} Translation'.format(language))

    # Merge Translations into Single File
    for translate_language in translate_languages.split(','):
        print("{0}: Merging Records for Translations into single File".format(translate_language))
        merge_files(input_directory=translated_directory.format(translate_language),
                    output_file=final_translated_file.format(translate_language))
