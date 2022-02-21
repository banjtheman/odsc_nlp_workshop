import requests
import json
import pandas as pd


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
