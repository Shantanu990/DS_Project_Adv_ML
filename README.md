**Project Title**: Adversarial Machine Learrning for Time Series: Trojan Reconstruction

**Project Overview**: 
A large telemetry dataset for satellites, titled clean_train_data, was obtained from Kaggle. It contains over 7.36 lakh+ multivariate time series samples across three channels: Channel 44, Channel 45, and Channel 46. Along with the dataset, following files were provided:
•	Clean_model.pt.ckpt – Checkpoint of a model trained on the clean dataset. This file includes the full N-HiTS architecture configuration (input/output chunk length, number of stacks/blocks/layers, dropout rate, activation type, etc.) and the trained weights.
•	Poisoned_model.pt.ckpt – Checkpoint containing weights of a model trained on a poisoned dataset, altered via backdoor injection.
The dataset was poisoned by periodically adding pairs of identical triggers to the clean data. As a result, the poisoned model learned to respond to the presence of a trigger by reproducing its copy within a short forecast horizon. 

**Problem Statement**: In a multivariate time series setting, a trigger pattern was embedded into the training data, poisoning the model. This poisoning causes two key effects:
•	The model’s forecasts become generally distorted.
•	When the trigger pattern appears in the input, the model reproduces the trigger pattern in its output, alongside the induced distortion.

**Project Objectives**: 
1.	Determine whether the observed forecast deviations between the clean and poisoned models are the result of targeted model poisoning rather than stochastic variability.
2.	Reconstruct an approximate representation of the trigger used in the poisoning process.

**Data Source**: [https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data ](https://www.kaggle.com/competitions/trojan-horse-hunt-in-space/overview)

**Project Details**:  
The project report, located in the main branch (file name: Adv ML Trojan Reconstruction report.pdf), provides a full breakdown of the methodology and findings, including:

- Exploratory Data Analysis (EDA) & Initial Model Comparison

- Approach for Trigger Reconstruction

- Optimization Procedure

- Results and Findings

**Tools and libraries used**: Software: Python, Excel; Libraries: torch, darts, pandas, numpy, matplotlib, skopt; Neural Network: N-HiTS; Optimizer: Bayesian, Adam
To execute the NHiTS model install following versions:
!pip install pip==23.3.1
!pip install pytorch-lightning==2.2.1
!pip install u8darts==0.27.0

**Respository Contents**: 
- 1 python script- "Adv_ML_TR.py" for finding candidate trigger and visualizing the forecast differences
- Project report (Adv ML Trojan Reconstruction report.pdf) to provide detailed discription of the project and the results.
- The clean train dataset (clean_train_data.mmr.xlsx)

**How to run the model**: Pre-requisites: Python environment- python 3.9+, an IDE like JupyterLab, MS Excel.
1. Download and unzip the clean_train_data.xlsx dataset file, place this file into the python working directory
2. Find and download model checkpoint files clean_model.pt.ckpt and poisoned_model.pt.ckpt in **Releases** section and place the files into the working directory
3. Open the file Adv_ML_TR.py in a Python IDE, update the file paths to correctly reference the dataset, model checkpoints, and the output location for saving the candidate trigger data. Execute the entire script to perform trigger optimization and generate visualizations of the triggered forecasts.

**Acknowledgment**: Thanks to Kaggle and ESA for publishing the dataset and model dictionaries

