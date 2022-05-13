# Python imports
import logging
import os

# Project imports
import utils as helpful_funcs

# 3rd party imports
from metaflow import FlowSpec, Parameter, step, card
import numpy as np

# How to run
# python helpful_flow.py run --output_dir test_run


class HelpfulFlow(FlowSpec):
    """
    This flow will run the Helpful pipeline
    """

    output_dir = Parameter(
        "output_dir",
        default="test_run",
        help="Location of output files",
        required=True,
    )

    # The helpful training data
    train_data = "https://helpful-sentences-from-reviews.s3.amazonaws.com/train.json"
    test_data = "https://helpful-sentences-from-reviews.s3.amazonaws.com/test.json"

    @card
    @step
    def start(self):
        """
        This is the 'start' step. All flows must have a step named 'start' that
        is the first step in the flow. We will download the data
        """

        # Make output dir
        cmd = f"mkdir -p {self.output_dir}"
        os.system(cmd)

        # Get raw data
        self.raw_data_train = helpful_funcs.get_data(self.train_data)
        self.raw_data_test = helpful_funcs.get_data(self.test_data)
        self.next(self.prepare_data)

    @card
    @step
    def prepare_data(self):

        """
        prepare data
        """
        # Transfrom raw data to a dataframe
        self.df_train = helpful_funcs.data_to_df(self.raw_data_train)
        self.df_test = helpful_funcs.data_to_df(self.raw_data_test)

        # save df to output folder
        self.df_train.to_csv(
            f"{self.output_dir}/helpful_sentences_train.csv", index=False
        )
        self.df_test.to_csv(
            f"{self.output_dir}/helpful_sentences_test.csv", index=False
        )

        # We can call N functions to run in parallel
        self.next(self.vader_run, self.fasttext_start, self.huggingface_split)

    @card
    @step
    def vader_run(self):

        """
        Run vader on data
        """
        # Transfrom raw data to a dataframe
        self.results = helpful_funcs.test_vader(self.df_test)
        self.run_name = "vader"

        self.next(self.join)

    @card
    @step
    def fasttext_start(self):

        """
        Convert data to fasttext format
        """

        helpful_funcs.convert_csv_to_fast_text_doc(self.df_train, self.output_dir)
        self.next(self.fasttext_train)

    @card
    @step
    def fasttext_train(self):

        """
        Train fasttext model
        """

        # Note the fasttext_model cant be saved by metaflow, so we just eval here
        fasttext_model = helpful_funcs.train_fasttext_model(self.output_dir)
        self.results = helpful_funcs.test_fasttext(self.df_test, fasttext_model)
        self.run_name = "fasttext"

        self.next(self.join)

    @card
    @step
    def huggingface_split(self):

        """
        Split data into 5
        """
        # TODO we can prob split based on max workers
        self.helpful_list = np.array_split(self.df_test, 5)

        self.next(self.huggingface_predict, foreach="helpful_list")

    @card
    @step
    def huggingface_predict(self):

        """
        Predict with huggingface model
        """

        self.run_name = "huggingface_"
        self.results = helpful_funcs.run_hugging_face(self.input)

        self.next(self.huggingface_join)

    @card
    @step
    def huggingface_join(self, inputs):
        """
        Combine huggingface scores
        """

        self.results = [input.results for input in inputs]
        self.run_names = [input.run_name for input in inputs]

        print("Huggingface Results")
        print(self.results)
        print(self.run_names)

        sent_scores = {}
        sent_scores["pos_match"] = 0
        sent_scores["neg_match"] = 0
        sent_scores["miss"] = 0
        sent_scores["model"] = "huggingface"

        for index, result in enumerate(self.results):

            sent_scores["pos_match"] += result["pos_match"]
            sent_scores["neg_match"] = result["neg_match"]
            sent_scores["miss"] = result["miss"]

        num_sents = (
            sent_scores["pos_match"] + sent_scores["neg_match"] + sent_scores["miss"]
        )
        missed_percent = sent_scores["miss"] / num_sents
        correct_percent = 1 - missed_percent
        sent_scores["missed_percent"] = missed_percent
        sent_scores["correct_percent"] = correct_percent

        self.run_name = "huggingface"
        self.results = sent_scores

        self.next(self.join)

    @card
    @step
    def join(self, inputs):
        """
        Save data artifacts from the runs
        """

        self.results = [input.results for input in inputs]
        self.run_names = [input.run_name for input in inputs]

        print("Final Results")
        print(self.results)
        print(self.run_names)

        for index, result in enumerate(self.results):

            curr_name = self.run_names[index]

            # save outputs
            helpful_funcs.save_json(
                f"{self.output_dir}/{curr_name}_results.json", result
            )

        self.next(self.end)

    @card
    @step
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow. It will print a "Done and Done"
        """

        logging.info("Done and Done")


if __name__ == "__main__":
    loglevel = logging.INFO
    logging.basicConfig(
        format="%(asctime)s |%(levelname)s: %(message)s", level=loglevel
    )
    HelpfulFlow()
