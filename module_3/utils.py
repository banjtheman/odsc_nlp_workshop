import requests
import json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from typing import List, Any


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

    print(sent_scores)

    # get percentages
    print(f"Pos correct: {sent_scores['pos_match']/num_sents}")
    print(f"Neg correct: {sent_scores['neg_match']/num_sents}")
    print(f"Missed: {sent_scores['miss']/num_sents}")

    return sent_scores
