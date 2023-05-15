# CS6024-Age-prediction
Mini project for CS6024 - Algorithmic Approaches to Computational Biology, on the topic: Gene expression analysis for age prediction

## Dataset

We use the *Age prediction using machine learning* dataset, obtained from [here](https://zenodo.org/record/2545213). The file `training_data_normal.tsv` is used for the project. 

The dataset has not been included in the repository since it is openly available. To run the code present in the repository, please download the dataset and save it as `dataset.tsv` in a new folder `data` created in the cloned repository. The helper script [get_dataset.sh](./get_dataset.sh) should do the same when the command `bash get_dataset.sh` is executed.

Once this is done, the directory structure would be as follows: 

```bash
.
├── code
│   └── ...
├── data
│   └── dataset.tsv
├── ...
├── Proposal.pdf
└── README.md
```

## Structure

All source code is present in [code](./code/). Helper scripts are present in [scripts](./code/scripts/) to help with loading data and building models. The used code is also distributed across the notebooks but follows the logical 3-phase sequence of EDA, modeling and result analysis. [Logs](./logs/) and [plots](./plots/) contain cross-validation information and final plots. 
