# NeurIPS 2024 Paper Submission Repository

This repository contains the code and resources associated with the paper titled *"MILCA: Multiple Instance Learning using Counting and Attention"*.

## Installation and Usage

1. *Clone the repository:*

   ```bash
   git clone <repository-url>
   ```

2. *Install the necessary packages:*

    Using [conda](https://docs.conda.io/projects/conda/en/latest/index.html) and the provided environment.yml file:
    ```bash
    conda env create -f environment.yml
    conda activate MIL
    ```

3. *Model Input Format:*

   The train_test function expects the input to be a list of bags, where each bag is a list of instances, and each instance is a list of features.

   For example, for the wiki data:
   ```python
   len(wiki) # Number of bags
   > 200
   len(wiki[0]) # Number of instances in the first bag
   > 5
   len(wiki[0][0]) # Number of features in the first instance
   > 1282
   ```

   To run the model:

   ```python
   acc, auc, run_time = train_test(wiki, config)
   ```

   The cofig dictionary is in the following format:

   ```python
   'dataset_name': {
        'lr' : [<value>], # Learning Rate
        'bs' : [<value>] , # Batch Size
        'p_cutoff' : [<value>], # P value
        'wd' :  [<value>] # Weight Decay
                }
   ```

## Publicly Available Competitors
In our paper, we compare MILCA with several publicly available models on new dataset and share new metrics. The table below summarizes the results.
![alt text](image.png)





To test the other models, we used the following open-sourced repositories:

- *mi-SVM* and *MI-SVM*: [Repository](https://github.com/garydoranjr/misvm) by Gary Doran.
- *Attention* and *Gated-Attention*: [Repository](https://github.com/AMLab-Amsterdam/AttentionDeepMIL) by Maximilian Ilse and Jakub M. Tomczak Link.