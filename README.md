# ECE1508_Applied_Deep_Learning
### Members: Kevin, John, Paul. 

## This project is based on the given topic:

### Question Answering: Task-Specific Models versus LLMs
Dataset: Recipe-MPR 

Description: The Recipe-MPR dataset consists of 500 queries by users, each having a set of answers given in five different ways. In this project, the students will train a deep model (suggested to be pretrained and only fine-tuned) on this dataset to answer questions.  The trained model is then to be compared against directly prompting an LLM. We know that beating LLMs is challenging, but obtaining a comparable performance is reasonable. The students are expected to train a model that perform above baseline accuracy which is roughly 65%.

### Proposal (due Oct.7)
https://docs.google.com/document/d/1pxMlcIiSNDtYhra_wOLDqr0A45yOCul6MEtAqfR-4MA/edit?usp=sharing

## Project Structure

This repository is organized into several key directories:

- **`baselines/`**: Implementation of baseline models including aspect-based and monolithic approaches (Dense, Sparse, GPT-3, etc.).
- **`bert_experiments/`**: Scripts and code for training and evaluating BERT model variants.
- **`distilbert/`**: Fine-tuning and evaluation workflows specifically for DistilBERT on the Recipe-MPR dataset.
- **`llamaFineTune/`**: Comprehensive pipeline for fine-tuning Llama models (e.g., Llama-3.2), including data preparation, training, and result analysis.
- **`qwen/`**: Evaluation and training scripts for Qwen models.
- **`data/`**: Contains the Recipe-MPR dataset, including original and augmented versions (QA pairs).
- **`docs/`**: Project documentation, including the final report, progress reports, and the project proposal.
- **`scripts/`**: Utility scripts for various tasks such as downloading models and running specific evaluations.
