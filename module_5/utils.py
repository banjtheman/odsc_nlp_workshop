import requests
import random
import logging
import json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import fasttext
import math
from typing import List, Any

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)


my_punctuation = "#!\"$%&'()*+,-./:;<=>?[\\]^_`{|}~•@“…ə"

# cleaning master function
def clean_text(text: str, bigrams: bool = False) -> str:
    text = text.lower()  # lower case
    text = re.sub("[" + my_punctuation + "]+", " ", text)  # strip punctuation
    text = re.sub("\s+", " ", text)  # remove double spacing
    # text = re.sub('([0-9]+)', '', text) # remove numbers
    return text


def write_to_file(file_path: str, file_text: str) -> bool:
    """
    Purpose:
        Write text from a file
    Args/Requests:
         file_path: file path
         file_text: Text of file
    Return:
        Status: True if appened, False if failed
    """

    try:
        with open(file_path, "w") as myfile:
            myfile.write(file_text)
            return True

    except Exception as error:
        logging.error(error)
        return False


def save_json(json_path: str, json_data: Any) -> None:
    """
    Purpose:
        Save json files
    Args:
        path_to_json: Path to  json file
        json_data: Data to save
    Returns:
        N/A
    """
    try:
        with open(json_path, "w") as outfile:
            json.dump(json_data, outfile)
    except Exception as error:
        raise OSError(error)


def run_hugging_face(df):
    """
    Purpose:
        test hugging face modle on the data
    Args:
        df - data
    Returns:
        sentiment scores: df of the data
    """

    tokenizer = AutoTokenizer.from_pretrained(
        "banjtheman/distilbert-base-uncased-helpful-amazon"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "banjtheman/distilbert-base-uncased-helpful-amazon"
    )

    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    sentences = list(df["sentence"])
    labels = list(df["helpful_label"])

    num_sents = len(sentences)

    sent_scores = {}
    sent_scores["pos_match"] = 0
    sent_scores["neg_match"] = 0
    sent_scores["miss"] = 0
    sent_scores["model"] = "huggingface"

    for index, sentence in enumerate(sentences):

        curr_label = labels[index]

        # polarity_scores method of SentimentIntensityAnalyzer
        # object gives a sentiment dictionary.
        # which contains pos, neg, neu, and compound scores.
        result = pipe(sentence)

        curr_sent = ""

        if result[0]["label"] == "LABEL_1":
            curr_sent = "pos"
        else:
            curr_sent = "neg"

        if curr_sent == "pos" and curr_label == "helpful":
            sent_scores["pos_match"] += 1
        elif curr_sent == "neg" and curr_label == "not_helpful":
            sent_scores["neg_match"] += 1
        else:
            sent_scores["miss"] += 1

        # If you just want to run on a few
        # if index > 50:
        #     num_sents = 50
        #     break

    # get percentages
    missed_percent = sent_scores["miss"] / num_sents
    correct_percent = 1 - missed_percent
    sent_scores["missed_percent"] = missed_percent
    sent_scores["correct_percent"] = correct_percent

    print(f"Missed: {missed_percent}")
    print(f"Correct: {correct_percent}")

    print(sent_scores)
    return sent_scores


def create_row_for_fast_text_doc(row: pd.Series, text_array: List):
    """
    Purpose:
        add cleaned text to an array
    Args:
        row - PD row
        text_array - array for text
    Returns:
        N/A
    """
    text = ""
    # get labels
    labels = row["helpful_label"].split(",")
    # logging.info(labels)

    for label in labels:
        text += "__label__" + label + " "

    text += clean_text(row["sentence"]) + "\n"
    # logging.info(text)
    text_array.append(text)


def convert_csv_to_fast_text_doc(
    df: pd.DataFrame, model_loc: str, cleaning_function=None
):
    """
    Purpose:
        Transform csv to fasttext format
    Args:
        model_loc: model location
        df - Dataframe of the csv
    Returns:
        N/A
    """

    text_array = []
    df.apply(lambda row: create_row_for_fast_text_doc(row, text_array), axis=1)

    # should randomize training and validation set
    random.shuffle(text_array)
    logging.info(f"text array size: {len(text_array)}")

    train_text = ""
    valid_text = ""
    # do a classic 80/20 split
    train_len = math.ceil(len(text_array) * 0.8)

    logging.info(f"train len size: {train_len}")

    for string in text_array[:train_len]:
        train_text += string

    for string in text_array[train_len:]:
        valid_text += string

    # TODO should have a run folder each time we do train, to keep track of artifcats
    write_to_file(f"{model_loc}/fasttext.train", train_text)
    write_to_file(f"{model_loc}/fasttext.valid", valid_text)


def train_fasttext_model(model_loc: str):
    """
    Purpose:
        Train a fasttext model
    Args:
        model_loc: location of model output
    Returns:
        N/A
    """

    model = fasttext.train_supervised(
        input=f"{model_loc}/fasttext.train",
        epoch=500,
        wordNgrams=1,
        bucket=200000,
        dim=50,
        lr=0.05,
    )

    results_json = print_results(*model.test(f"{model_loc}/fasttext.valid", k=-1))
    # save model
    # now = str(datetime.now())
    model.save_model(f"{model_loc}/fasttext.bin")

    return model


def print_results(N: int, p: float, r: float):
    """
    Purpose:
        Print training results
    Args:
        N - number of sentences
        p - precision
        r - recall
    Returns:
        N/A
    """
    logging.info("Number tested\t" + str(N))
    logging.info("Precision{}\t{:.3f}".format(1, p))
    logging.info("Recall{}\t{:.3f}".format(1, r))

    results_json = {}
    results_json["number"] = N
    results_json["precision"] = round(p, 2)
    results_json["recall"] = round(r, 1)

    return results_json


# https://helpful-sentences-from-reviews.s3.amazonaws.com/train.json
def get_data(url: str):
    """
    Purpose:
        gets the json data
    Args:
        url - location of data
    Returns:
        data_array: array of the data
    """

    data = requests.get(url)
    raw_data = data.text
    data_array = raw_data.split("\n")

    return data_array


def data_to_df(data_array):
    """
    Purpose:
        turn data to a df
    Args:
        data_array - array of data
    Returns:
        df: df of the data
    """

    df_map = {}
    df_map["asin"] = []
    df_map["sentence"] = []
    df_map["helpful"] = []
    df_map["main_image_url"] = []
    df_map["product_title"] = []
    df_map["helpful_label"] = []

    for item in data_array:

        try:
            item_json = json.loads(item)
        except:
            continue

        df_map["asin"].append(item_json["asin"])
        df_map["sentence"].append(item_json["sentence"])
        df_map["main_image_url"].append(item_json["main_image_url"])
        df_map["product_title"].append(item_json["product_title"])

        # Get helpful label from score
        helpful_score = item_json["helpful"]
        df_map["helpful"].append(helpful_score)
        helpful_label = get_helpful_label_simple(helpful_score)
        df_map["helpful_label"].append(helpful_label)

    # Create dataframe
    df = pd.DataFrame.from_dict(df_map)
    # save output
    # df.to_csv("helpful_sentences.csv", index=False)

    return df


def get_helpful_label_simple(helpful_score):
    """
    Purpose:
        gets a label based on score
    Args:
        helpful_score: score of how helpful
    Returns:
        helpful_label: label of how helpful
    """

    if helpful_score < 1:
        return "not_helpful"
    else:
        return "helpful"


def test_fasttext(df, model):
    """
    Purpose:
        test fasttext model with data
    Args:
        df - data
    Returns:
        sentiment scores: df of the data
    """

    sentences = list(df["sentence"])
    labels = list(df["helpful_label"])

    num_sents = len(sentences)

    sent_scores = {}
    sent_scores["pos_match"] = 0
    sent_scores["neg_match"] = 0
    sent_scores["miss"] = 0
    sent_scores["model"] = "fasttext"

    for index, sentence in enumerate(sentences):

        curr_label = labels[index]

        result = str(model.predict(sentence)[0])

        curr_sent = ""

        if "not_helpful" in result:
            curr_sent = "pos"
        else:
            curr_sent = "neg"

        if curr_sent == "pos" and curr_label == "helpful":
            sent_scores["pos_match"] += 1
        elif curr_sent == "neg" and curr_label == "not_helpful":
            sent_scores["neg_match"] += 1
        else:
            sent_scores["miss"] += 1

        # if index > 500:
        #     break

    # num_sents = 500

    # get percentages
    missed_percent = sent_scores["miss"] / num_sents
    correct_percent = 1 - missed_percent
    sent_scores["missed_percent"] = missed_percent
    sent_scores["correct_percent"] = correct_percent

    print(f"Missed: {missed_percent}")
    print(f"Correct: {correct_percent}")

    print(sent_scores)

    return sent_scores


def test_vader(df):
    """
    Purpose:
        turn data to a df
    Args:
        df - data
    Returns:
        sentiment scores: df of the data
    """

    sid_obj = SentimentIntensityAnalyzer()

    sentences = list(df["sentence"])
    labels = list(df["helpful_label"])

    num_sents = len(sentences)

    sent_scores = {}
    sent_scores["pos_match"] = 0
    sent_scores["neg_match"] = 0
    sent_scores["miss"] = 0
    sent_scores["model"] = "vader"

    for index, sentence in enumerate(sentences):

        curr_label = labels[index]

        # polarity_scores method of SentimentIntensityAnalyzer
        # object gives a sentiment dictionary.
        # which contains pos, neg, neu, and compound scores.
        sentiment_dict = sid_obj.polarity_scores(sentence)

        curr_sent = ""

        if sentiment_dict["compound"] >= 0.05:
            curr_sent = "pos"
        else:
            curr_sent = "neg"

        if curr_sent == "pos" and curr_label == "helpful":
            sent_scores["pos_match"] += 1
        elif curr_sent == "neg" and curr_label == "not_helpful":
            sent_scores["neg_match"] += 1
        else:
            sent_scores["miss"] += 1

    # get percentages
    missed_percent = sent_scores["miss"] / num_sents
    correct_percent = 1 - missed_percent
    sent_scores["missed_percent"] = missed_percent
    sent_scores["correct_percent"] = correct_percent

    print(f"Missed: {missed_percent}")
    print(f"Correct: {correct_percent}")

    print(sent_scores)

    return sent_scores
