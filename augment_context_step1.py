# Tweak these Args and run:

# args.train_file
# args.train_record_file
# args.train_eval_file


"""Download and pre-process SQuAD and GloVe.

Usage:
    > source activate squad
    > python setup.py

Pre-processing code adapted from:
    > https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py

Author:
    Chris Chute (chute@stanford.edu)
"""

import ujson as json
import argparse
import torch
import random
from codecs import open
from tqdm import tqdm
from nltk.corpus import wordnet
import nltk
import spacy
from os import listdir

# Download NLTK Dependencies
nltk.download('wordnet')
nltk.download('punkt')

from transformers import pipeline


# Get Translators for English to Target Language and back to English
def get_translators(language):
    if language == 'French':
        trans_model = 'Helsinki-NLP/opus-mt-en-fr'
        back_trans_model = 'Helsinki-NLP/opus-mt-fr-en'
    elif language == 'Chinese':
        trans_model = 'Helsinki-NLP/opus-mt-en-zh'
        back_trans_model = 'Helsinki-NLP/opus-mt-zh-en'
    else:
        raise Exception("Translation not supported for: {0}".format(language))

    return pipeline(model=trans_model, device=0, batch_size=16, truncation=True), \
           pipeline(model=back_trans_model, device=0, batch_size=16, truncation=True)


def calculate_spans(spans, context_size, context):
    if len(spans) == 0:
        # No Answers, Augment entire Context
        merged_spans = []
    else:
        spans.sort(key=lambda span: span[0])
        merged_spans = [spans[0]]
    for current in spans:
        previous = merged_spans[-1]
        # There must be smaller than 5 words Gap between each span
        if current[0] <= previous[1] + 5:
            previous[1] = max(previous[1], current[1])
        else:
            merged_spans.append(current)
    context_spans = []
    iterate_spans = merged_spans
    if iterate_spans and iterate_spans[0][0] == 0:
        previous_end = iterate_spans[0][1]
        iterate_spans = iterate_spans[1:]
    else:
        previous_end = 0

    for current in iterate_spans:
        context_spans.append([previous_end, current[0]])
        previous_end = current[1]

    if iterate_spans and iterate_spans[-1][1] < context_size:
        context_spans.append([iterate_spans[-1][1], context_size])

    context_only = False
    # No Questions, Augment entire Context
    if len(merged_spans) == 0:
        # context_spans.append([0, context_size])
        context_only = True
    # Edge case where only 1 Answer, starting at beginning
    elif len(context_spans) == 0:
        context_spans.append([merged_spans[0][1], context_size])

    if context_only:
        return [{
            "span": [0, context_size],
            "text": context,
            "type": "context"
        }]
    elif merged_spans[0][1] < context_spans[0][0]:
        first = merged_spans
        last = context_spans
        first_type = "answer"
        last_type = "context"
    else:
        first = context_spans
        last = merged_spans
        first_type = "context"
        last_type = "answer"

    output = []
    for a, b in zip(first, last):
        output.append({
            "span": [a[0], a[1]],
            "text": context[a[0]: a[1]],
            "type": first_type
        })
        output.append({
            "span": [b[0], b[1]],
            "text": context[b[0]: b[1]],
            "type": last_type
        })
    output.append({
        "span": first[-1],
        "text": context[first[-1][0]: first[-1][1]],
        "type": first_type
    })
    return output
    # return merged_spans, context_spans


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


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def translate_text_batch(original_batch, batch_id, translators):
    source_translator, target_translator = translators
    source_translations = [x['translation_text'] for x in source_translator(original_batch)]
    target_translations = [x['translation_text'] for x in target_translator(source_translations)]
    target_dict = {}
    for i, v in enumerate(target_translations):
        target_dict[i+batch_id] = v
    return target_dict


def shuffle_text(original_text):
    text_split = original_text.split(' ')
    random.shuffle(text_split)
    return ' '.join(text_split)


def process_file(filename, data_type):

    # Setup NER Detector
    # spacy.prefer_gpu()
    # entity_detector = spacy.load('en_core_web_trf')

    print(f"Pre-processing {data_type} : Augmenting Context...")

    paragraph = 0
    output_spans = {}
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"]
                context_size = len(context)
                para_answers = []
                for qa in para["qas"]:
                    for ans in qa["answers"]:
                        answer_span = [ans['answer_start'], ans['answer_start'] + len(ans['text'])]
                        para_answers.append(answer_span)
                calculate_spans(para_answers, context_size, context)
                output_spans[paragraph] = calculate_spans(para_answers, context_size, context)
                paragraph += 1


    # Collect Context List for Language Augmentation
    context_list = []
    for i in output_spans:
        for span in output_spans[i]:
            if span['type'] == 'context':
                context_list.append(span['text'])
    print(len(context_list))

    context_index = 0
    for i in tqdm(output_spans):
        for x, span in enumerate(output_spans[i]):
            if span['type'] == 'context':
                output_spans[i][x]['Synonym_30'] = replace_with_synonym(original_text=span['text'],
                                                                        detector=entity_detector,
                                                                        probability=30)
                # output_spans[i][x]['Synonym_60'] = replace_with_synonym(original_text=span['text'],
                #                                                         detector=entity_detector,
                #                                                         probability=60)
                # output_spans[i][x]['French'] = translated_contexts['French'][context_index]
                # output_spans[i][x]['Chinese'] = translated_contexts['Chinese'][context_index]
                # output_spans[i][x]['Shuffle'] = shuffle_text(original_text=span['text'])
                context_index += 1
    save(filename='./data/context-Augmentation-temp.json',
         obj=output_spans,
         message='Temp Context Augmentation')

    # Setup Translators
    translate_languages = 'Chinese'
    language_translators = {}
    for translate_language in translate_languages.split(','):
        language_translators[translate_language] = get_translators(language=translate_language)
        language_translators[translate_language] = get_translators(language=translate_language)

    batch_size = 1000
    # translated_contexts = {}
    for language in translate_languages.split(','):
        # Run in Batches of 1K
        #language_contexts = []
        for batch in tqdm(range(0, len(context_list), batch_size)):
            if batch < 62000:
                continue
            with torch.no_grad():
                print('- {0}, Batch: {1} to {2}'.format(language, batch, batch+batch_size))
                back_translated_batch = translate_text_batch(original_batch=context_list[batch: batch+batch_size],
                                                             batch_id=batch,
                                                             translators=language_translators[language])
                # Save Each Run and Concat Later
                save(filename='./data/Translations/{0}/{1}-contexts.json'.format(language, batch),
                     obj=back_translated_batch,
                     message='{0} Translation'.format(language))


    translated_directory = './data/Translations/{0}'
    final_translated_file = './data/{0}-contexts.json'

    # Merge Translations into Single File
    for translate_language in translate_languages.split(','):
        print("{0}: Merging Records for Translations into single File".format(translate_language))
        merge_files(input_directory=translated_directory.format(translate_language),
                    output_file=final_translated_file.format(translate_language))


def merge_files(input_directory, output_file):
    print("-- Merging Records from Directory: {0} to File: {1}".format(input_directory, output_file))
    output_json = {}
    # Get List of all Json files in directory:
    file_names = listdir(input_directory)
    for file_name in file_names:
        if file_name.endswith('-contexts.json'):
            merge_file = '{0}/{1}'.format(input_directory, file_name)
            with open(merge_file, "r") as fh:
                questions = json.load(fh)
                for question_id in questions:
                    output_json[question_id] = questions[question_id]
    print("-- Writing # Questions : {0} to {1}".format(len(output_json.keys()), output_file))
    save(output_file, output_json, "Merging File")

def pre_process(args):
    process_file(args.train_file, "train")


if __name__ == '__main__':
    # Get command-line args
    # args_ = get_setup_args()

    # Download resources
    # download(args_)

    # Import spacy language model
    nlp = spacy.blank("en")

    parser = argparse.ArgumentParser('Args')
    args_ = parser.parse_args()


    args_.train_file = './data/train-v2.0.json'

    pre_process(args_)
