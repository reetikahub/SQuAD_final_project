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

from args import get_setup_args
from codecs import open
from tqdm import tqdm
import copy


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def process_file(args, input_file, language, uuid):
    output_data = {
        "data": []
    }
    print("Processing File: {0}".format(input_file))

    with open(input_file, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            data_json = {
                "title": article["title"],
                "paragraphs": []
            }
            for para in article["paragraphs"]:
                # Skip wrong language
                # if para["language"] != language:
                #     continue
                para_augmented_json = {
                    "context": para["context"],
                    "language": language,
                    "qas": []
                }
                for qa in para["qas"]:
                    # Data Validation Checks
                    if qa["technique"] not in args.augment_question_method \
                            or not qa["named_entities_match"] \
                            or (qa["technique"] != "Original" and qa["question"] == qa["original"]):
                        continue
                    # toDo: Remove for others:: Added for Only picking Original from Synonym
                    if qa["technique"] == "Original" and not qa["id"].endswith("0"):
                        continue
                    qa_augmented = copy.deepcopy(qa)
                    qa_augmented["id"] += str(uuid)
                    qa_augmented.pop("original")
                    qa_augmented.pop("named_entities_match")
                    # Make sure Question is not a Duplicate
                    if qa_augmented["question"] not in [x["question"] for x in para_augmented_json["qas"]]:
                        para_augmented_json["qas"].append(qa_augmented)
                # Make sure Context is not a Duplicate
                if para_augmented_json["context"] not in [x["context"] for x in data_json["paragraphs"]]:
                    data_json['paragraphs'].append(para_augmented_json)
            output_data['data'].append(data_json)
        return output_data


def merge_jsons(output_json_list):
    final_output_data = []
    for output_json in output_json_list:
        final_output_data += output_json["data"]
    return {
        "data": final_output_data
    }

def pre_process(args):
    # Get Output Json

    output_json_list = []

    # Get Original Json
    # output_json_list.append(process_file(args, args_.qa_augmented_fie, "Original", 0))

    # Get Augmented Jsons
    for i, language in enumerate(args_.augment_context_method):
        print("- Processing Context for: {0}".format(language))
        input_file = args.augment_base_file.format(language)
        output_json_list.append(process_file(args, input_file, language, i+1))

    # Merge All files
    merged_json = merge_jsons(output_json_list)

    # Save Merged data
    save(args.augment_save_file.format(args.name), merged_json, message="Merged Data: {0}".format(args.name))


if __name__ == '__main__':
    # Get command-line args
    args_ = get_setup_args()

    # Download resources
    # download(args_)

    # Import spacy language model
    # nlp = spacy.blank("en")

    # ToDo: Update Augmentation Params

    args_.name = "context_french_chinese"

    args_.augment_context_method = ["French", "Chinese"]
    args_.augment_question_method = ["Original"]

    args_.qa_augmented_fie = './data/train-augmented-v2.0.json'
    args_.augment_base_file = './data/train-augmented-context-{0}-v2.0.json'
    args_.augment_save_file = './data/train-augmented-{0}-v2.0.json'

    pre_process(args_)

