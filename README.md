# Building a Parallelized NLP Data Pipeline with Metaflow

Developing a data pipeline that can process multiple tasks comes with numerous barriers. From testing infrastructure, managing data flow, and tracking assets trying to hand roll these features is a strain on engineering effort. 

Luckily with Metaflow we have an open-source framework that provides the necessary tooling to build end-to-end workflows so we can focus on data science and less on engineering.

In this workshop we will use Metaflow to build an NLP pipeline to evaluate 3 different models for predicting how “helpful” a review is on a product page.  We will leverage tools such as huggingface transformers, fasttext, and pandas for data processing tasks.

This is an intermediate workshop that will require knowledge of using python and navigating a terminal. No NLP experience is necessary.

This workshop can be done locally or using Gitpod which provides a cloud-based developer environment ready to begin. Simply click the button below to begin.

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/banjtheman/odsc_nlp_workshop)   

To install the prerequisites on your local machine run
```
pip install -r requirements.txt
```

## Modules

Each module will take you through steps to run a Metaflow pipeline. Each folder has a README with the instructions for the module. 

### Getting Started with Metaflow

In this module we will setup Metaflow using a "hello world" flow to understand the basics of how to construct a Metaflow pipeline.

### Preparing Data

In this module we will extract the helpful review data from [AWS open data registry](https://registry.opendata.aws/helpful-sentences-from-reviews/) and prepare it for analysis. 

### VADER sentiment analysis

In this module we will run a VADER sentiment analysis on the text to evaluate if we can determine that helpful is equivalent to positive.

### FastText Text Classification

In this module we will update our pipeline to train a fasttext model alongside our VADER model to evaluate the text.

### Huggingface Transformers Classification

In this module we will update our pipeline to run a huggingface transformer model alongside the others to evaluate the text.