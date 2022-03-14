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

import numpy as np
import os
import spacy
import ujson as json
import urllib.request

from args import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile
import copy


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def detect_named_entities(original_text, back_translated_text, detector):
    original_detection = detector(original_text)
    back_translated_detection = detector(back_translated_text)

    original_ner = set([x.text for x in original_detection.ents])
    back_translated_ner = set([x.text for x in back_translated_detection.ents])

    return original_ner == back_translated_ner


def augment_contexts(args, detector, augment_context_method):

    print("--> Augmenting Context: {0}".format(augment_context_method))

    # 1. Read Contexts File:
    context_augment_file = './data/{0}-contexts.json'.format(augment_context_method)
    with open(context_augment_file, "r") as fh:
        augmented_contexts = json.load(fh)

    # 2. Read Base File:
    print("- Reading Base File: {0}".format(args_.context_base_file))
    context_id = 0
    with open(args_.context_base_file, "r") as fh:
        source = json.load(fh)
        new_source = {}
        for article in tqdm(source):
            previous_span_end = 0
            new_span_list = []
            for id, span in enumerate(source[article]):
                new_span = copy.deepcopy(span)
                new_span_start = previous_span_end
                if span["type"] == "context":
                    # print(french_contexts[str(context_id)])
                    old_context = span["text"]

                    # Added Space After new Context to Fix Issue
                    if id == 0:
                        new_context = augmented_contexts[str(context_id)] + " "
                    elif id == len(source[article]) - 1:
                        augmented_contexts[str(context_id)] + " "
                    else:
                        new_context = " " + augmented_contexts[str(context_id)] + " "
                    if not detect_named_entities(old_context, new_context, detector):
                        new_context = old_context

                    new_span_end = previous_span_end + len(new_context)
                    new_span["old_text"] = old_context
                    new_span["text"] = new_context
                    # print("Old Context: {0} , New Context: {1}".format(len(old_context), len(new_context)))

                    # ToDo: Compare old context with new context - NER changed?
                    context_id += 1
                else:
                    new_span_end = previous_span_end + len(span["text"])
                new_span["new_span"] = [new_span_start, new_span_end]
                previous_span_end = new_span_end
                new_span_list.append(new_span)
            new_source[article] = new_span_list
    return new_source


def get_new_answer_start(answer_start, spans):
    # print(answer_start)
    # print(json.dumps(spans))
    for span in spans:
        old_span = span["span"]

        if old_span[0] <= answer_start < old_span[1]:
            # Found Span
            answer_diff = answer_start - old_span[0]
            new_span_start = span["new_span"][0]
            return True, new_span_start + answer_diff

    # toDo: Edge Case Missed, 0.1% of questions affected. Drop from Augmented
    return False, None
    # raise Exception("No Span found for Answer!")


def augment_data(input_file, new_contexts, augment_context_method):
    output_data = {
        "data": []
    }

    if augment_context_method == "French":
        uuid = "1"
    elif augment_context_method == "Chinese":
        uuid = "2"
    else:
        raise Exception("Only French & Chinese Supported")

    para_index = 0

    with open(input_file, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            data_json = {
                "title": article["title"],
                "paragraphs": []
            }
            for para in article["paragraphs"]:
                context_spans = new_contexts[str(para_index)]
                new_context = ''.join([span["text"] for span in context_spans])
                old_context = para["context"]

                para_index += 1

                # Append Paragraph with Augmented Context
                para_augmented_json = {
                    "context": ''.join([span["text"] for span in context_spans]),
                    "language": augment_context_method,
                    "qas": []
                }

                good = 0
                bad = 0

                for qa in para["qas"]:
                    # cnt += 1  # toDo: Comment for Dev Purposes
                    question_counter = 0
                    original_question = qa['question']

                    # Prep Augmented Context Questions
                    # Replace Answer Spans to point to new Spans in Augmented Context
                    qa_augmented = copy.deepcopy(qa)
                    qa_augmented["id"] += uuid
                    found = True
                    if len(qa_augmented["answers"]) > 0:
                        for id, answer in enumerate(qa_augmented["answers"]):
                            found, new_answer_start = get_new_answer_start(answer_start=answer["answer_start"],
                                                                           spans=context_spans)
                            qa_augmented["answers"][id]["answer_start"] = new_answer_start
                    # toDo: Fix Edge case later
                    if found:
                        para_augmented_json["qas"].append(qa_augmented)
                        good += 1
                    else:
                        bad += 1
                # data_json['paragraphs'].append(para_original_json)
                data_json['paragraphs'].append(para_augmented_json)
            output_data['data'].append(data_json)
        print("-- Good: {0}, Bad: {1}".format(good, bad))
        return output_data


def pre_process(args, augment_context_method):
    train_file = './data/train-augmented-v2.0.json'
    augmented_file = './data/train-augmented-Context-{0}-v2.0.json'

    # Setup NER Detector
    spacy.prefer_gpu()
    entity_detector = spacy.load('en_core_web_trf')

    # # Process training set and use it to decide on the word/character vocabularies
    context_spans = augment_contexts(args, entity_detector, augment_context_method)
    # print(json.dumps(context_spans))

    # Augment File
    augmented_data = augment_data(input_file=train_file, new_contexts=context_spans, augment_context_method=augment_context_method)
    save(augmented_file.format(augment_context_method), augmented_data, message="Augmented Context")


if __name__ == '__main__':
    # Get command-line args
    args_ = get_setup_args()

    # ToDo: Update Augmentation Params
    # augment_context_method = "French"
    args_.augment_method = ["French", "Chinese", "Original"]

    args_.context_base_file = './data/context-Augmentation-temp.json'
    pre_process(args_, "French")
    pre_process(args_, "Chinese")

