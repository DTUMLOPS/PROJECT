# ehr_classification


## Overall Goal of the Project

This project aims to develop a mortality prediction model for ICU patients using a Deep State Space Model (SSM) approach. Leveraging the PhysioNet/Computing in Cardiology (CinC) Challenge 2012 dataset, which includes time-series data from 12,000 ICU patients, the project emphasizes the implementation of MLOps practices. We are bulding on top of a previous project that focused on the training of models with a focus on accuracy, while this project instead focuses on practices to ensure reproducibility, version control, and collaborative scalability, critical for advancing machine learning applications in healthcare.

## Frameworks and Tools

- **PyTorch**: Used to build and train the model due to its flexibility and low-level control, allowing detailed customization of architecture and training pipelines.

- **Scikit-learn**: Supports model evaluation by providing standardized metrics such as AUROC and accuracy.

- **MLOps Tools**: Enables automated pipeline management, experiment tracking, and continuous integration (CI). These workflows ensure:
  - Robust versioning of datasets and models.
  - Reproducibility of experiments.
  - Smooth collaboration across teams.


## Data

The PhysioNet/CinC Challenge 2012 dataset provides rich time-series data collected during the first 48 hours of ICU admission for 12,000 patients. This dataset includes:

- **Physiological Measurements**: Heart rate, blood pressure, oxygen saturation, etc.
- **Demographic Descriptors**: Age, gender, etc.
- **Outcome-related Information**: Mortality indicators.

This real-world dataset is ideal for developing and testing the proposed model. It is publicly accessible at **[PhysioNet CinC Challenge 2012 Dataset](https://physionet.org/content/challenge-2012/1.0.0/)**.


## Model Architecture

The core model will be based on Deep State Space Models (SSM), particularly suited for capturing temporal dynamics in time-series data. Key features of the architecture include:

- **Integration of LSTM Networks**: Effective modeling of long-term dependencies in time-series data.
- **Balance of Predictive Performance and Interpretability**: Ensuring the model is both accurate and understandable.
- **MLOps Integration**: Seamless workflows for:
  - Efficient deployment.
  - Continuous training.
  - Iterative improvements.

  
A refactor of a previously implemented EHR for MLOps course @ DTU

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   └── processed
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── models/                   # Trained models
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks - 
```

## Setup environment and install requirements
```
pip install invoke
invoke create-environment
conda activate ehr_classification
invoke requirements
invoke dev-requirements
```

## If requirements.txt changes
```
# Just run invoke requirements again
invoke requirements
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
