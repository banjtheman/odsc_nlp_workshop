# FastText Text Classification

[FastText](https://fasttext.cc/) is an open-source, free, lightweight library that allows users to learn text representations and text classifiers.

In this module we will update our pipeline to train a fasttext classifer model alongside our VADER model to evaluate the text.

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
cat test_run/fasttext_results.json
```

4. Can you make the fasttext model better? Play around with the `train_fasttext_model` function in utils.py. Here is some [reference material](https://fasttext.cc/docs/en/supervised-tutorial.html) 



