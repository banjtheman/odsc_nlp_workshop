# Preparing Data

We will extract the helpful review data from [AWS open data registry](https://registry.opendata.aws/helpful-sentences-from-reviews/) and prepare it for analysis. 

The data is in a json format not suitable for machine learning, so we will create a simple data pipeline to transform the data into a pandas dataframe.

## Steps

1. Run the show command to see the steps of your pipeline

```bash
python helpful_flow.py show
```

2. Run the pipeline to save the output to the test_run directory

```bash
python helpful_flow.py run --output_dir test_run
```

3. View the data 

```bash
head test_run/helpful_sentences_train.csv
head test_run/helpful_sentences_test.csv
```



