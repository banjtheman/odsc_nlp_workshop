# Huggingface Transformers Classification

[Transformers](https://huggingface.co/docs/transformers/index) provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio. It provides hands-on practitioners the ability to fine-tune NLP models to be used easily in production.

In this module we will update our pipeline to run a huggingface transformer model alongside the others to evaluate the text. We will use a [pretrained model](https://huggingface.co/banjtheman/distilbert-base-uncased-helpful-amazon) trained on this data.

## Steps

1. Run the show command to see the steps of your pipeline

```bash
python helpful_flow.py show
```

2. Run the pipeline to save the output to the test_run directory

```bash
python helpful_flow.py run --output_dir test_run
```

**Note:** The gitpod instance is slow, so you can comment out lines 122-123 and 126 to test only a subset of data if not running on your personal machine.  


3. View the data 

```bash
cat test_run/huggingface_results.json
```

4. Why was the huggingface model so good? You can train your own model with train.py (will take couple hours...) or try out other [models].(https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads)



