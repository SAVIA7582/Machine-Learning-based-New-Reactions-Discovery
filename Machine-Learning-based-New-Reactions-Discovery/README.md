# Machine-Learning-based-New-Reactions-Discovery
Welcome to the Machine-Learning-based-New-Reactions-Discovery GitHub repository. Our project utilizes advanced machine learning (ML) techniques to streamline the discovery of novel carbon-carbon (C-C) coupling reactions. This repository contains the code for our ML-guided workflow, designed to facilitate the direct C-C couplings between secondary alcohols and boronic acids, thereby promoting atom economy and environmental sustainability.
Our ML algorithms are trained on a dataset of 705 transformations involving boronic acids and activated moieties. The model demonstrates high predictive accuracy with a correlation coefficient (R²) of 0.98, enabling researchers to prioritize and experimentally validate promising reactions with greater efficiency.

 ## **Key Features:**
- Employs machine learning to guide the discovery of chemical reactions.
- Achieves precise predictions and extrapolations with high correlation coefficients (R² up to 0.98).
- Efficiently screens potential C-C couplings, accelerating the path to experimental validation.
- Applicable across a diverse range of alkenyl boronic acids and secondary alcohols.
- Offers a rapid screening methodology underpinned by robust ML predictions.

# System Requirements
## **Operating Systems:**
- Windows (version 13.8 or higher)
- Ubuntu (version 20.04 or higher)
- MacOS (version 10.13 or higher)
  
## **Required Non-standard Hardware:** 
 	 - None

# Software Dependencies
- Python (version 3.8 or higher)

- Anaconda (conda version 22.9.0 or higher)

- scikit-learn (version 0.24.2 or higher)

- numpy (version 1.20.3 or higher)

- pandas (version 1.3.3 or higher)

- matplotlib (version 3.4.3 or higher)

- tensorflow (version 2.16.1 or higher)

- keras (version 3.3.3 or higher)

- pipe(version 2.2 or higher)


# Installation Guide
To use the code with an Anaconda environment, follow the installation procedure here:
```bash
conda create -n mlnrd python=3.8.4
conda activate mlnrd
conda install numpy
conda install pandas
conda install scikit-learn
conda install matplotlib
conda install seaborn
conda install rdkit
conda install tensorflow==2.16.1
conda install keras==3.3.3
conda install pipe
conda install openpyxl
conda install jupyter notebook
```

If you are new to Anaconda, you can install it from [here](https://www.anaconda.com/).

# Usage

To run the ML model, navigate to the Anaconda interface and execute the provided Jupyter Notebook (`Model.ipynb`). Follow the step-by-step instructions within the notebook to perform the analysis.


## **Expected Runtime:**
- Approximately 30 minutes
  
# Repository Structure
- `dataset.csv`: This dataset used for training the ML model and conducting experiments as detailed in the associated publication.
- `Model.ipynb`: This ipython notebook includes the code for preprocessing dataset,  training the Convolutional Neural Network (CNN) model used in our study, and evaluating the performance of it.
For detailed information on the methodology and experimental setup, please refer to the accompanying research paper.
- `AugmentData.py`: Functions about generating augmented data for reaction smiles.
- `EncodeReaction.py`: Functions about converting reaction smiles to sequences of numbers including extract their properties.
- `VAE.py`: Functions of building VAE encoders and decoders.
- `CNN.py`: Structure of CNN used in our study.
- `Plot.py`: Functions of Drawing related figures.
- `predict reactions.csv`: Smiles with supplemental data of 127 reactions without yields, which is used for predicting their yields by our trained model.

# Support
For any questions or issues, please open an issue on this GitHub repository, and we will assist you as soon as possible.
Thank you for your interest in our ML-based approach to new reaction discovery. We look forward to any contributions or feedback from the community!

