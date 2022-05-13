# Python imports
import logging
import os

# Project imports
import utils as helpful_funcs

# 3rd party imports
from metaflow import FlowSpec, Parameter, step, card

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
