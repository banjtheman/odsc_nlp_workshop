# VADER sentiment analysis

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media.

In this module we will run a VADER sentiment analysis on the text to evaluate if we can determine that helpful is equivalent to positive. 

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
cat test_run/vader_results.json
```



