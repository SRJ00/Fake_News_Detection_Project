# Fake News Detection on BuzzFeed Data

## Project Overview

This project focuses on detecting fake news within the BuzzFeed subset of the FakeNewsNet dataset. It involves Exploratory Data Analysis (EDA) to uncover distinguishing features, engineering content and metadata-based features, and building Random Forest models to classify news articles. The primary goal is to evaluate if incorporating metadata improves model performance (Macro F1-Score) compared to using content alone.

## Dataset

* **Source:** FakeNewsNet - BuzzFeed Subset
* **Data URL:** [https://www.kaggle.com/datasets/mdepak/fakenewsnet](https://www.kaggle.com/datasets/mdepak/fakenewsnet)
    * The specific files used from this dataset are detailed within the Jupyter Notebook.

## Methodology

1.  **Data Loading & Preparation:** Loaded and cleaned news content (CSVs) and auxiliary mapping/interaction files (`.txt`). This included ID mapping, date parsing, and extraction of source and HTML metadata.
2.  **Exploratory Data Analysis (EDA):** Analyzed text characteristics (lengths, readability, sentiment, punctuation), source/author patterns, HTML metadata, temporal trends, multimedia presence, and user engagement metrics.
3.  **Feature Engineering:**
    * **Content-Only:** TF-IDF features from processed article titles and text.
    * **Content + Selected Metadata:** Combined TF-IDF features with a scaled subset of the most influential metadata features (identified via Random Forest feature importance).
4.  **Modeling:**
    * **Algorithm:** Random Forest Classifier.
    * **Comparison:** Trained models on "Content-Only" vs. "Content + Selected Metadata" features.
    * **Evaluation:** Macro F1-Score on a held-out test set.

## Key Results

* EDA identified several differentiators (e.g., title length, exclamation mark usage, source reputation, meta keyword usage, spreader activity).
* The Random Forest model using **Content + Selected Metadata (Macro F1: ~0.693)** outperformed the **Content-Only model (Macro F1: ~0.673)**, indicating the value of well-chosen metadata.

## Repository Contents

* `fakenewsdetection.ipynb`: The main Jupyter Notebook containing all code for EDA, feature engineering, modeling, and analysis.
* `Fake News Identification.pdf`: Project report detailing the process and findings.
* `charts/`: Directory containing selected charts generated during EDA.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SRJ00/Fake_News_Detection_Project.git](https://github.com/SRJ00/Fake_News_Detection_Project.git)
    cd Fake_News_Detection_Project
    ```
2.  **Set up Environment & Dependencies:**
    * Ensure Python 3.x is installed.
    * Install required libraries (see main libraries below or create a `requirements.txt` from the notebook environment):
        ```bash
        pip install pandas numpy scikit-learn nltk matplotlib seaborn plotly wordcloud textstat
        ```
    * Download NLTK resources used in the notebook (e.g., `stopwords`, `punkt`, `wordnet`, `averaged_perceptron_tagger`, `vader_lexicon`).
3.  **Obtain Data:**
    * Download the FakeNewsNet dataset from the Kaggle link provided above.
    * Place the necessary BuzzFeed subset files (e.g., `BuzzFeed_fake_news_content.csv`, `BuzzFeed_real_news_content.csv`, and the `.txt` mapping/interaction files) in a directory accessible by the notebook (e.g., an `input/fakenewsnet/` structure within the project, or adjust paths in the notebook accordingly).
4.  **Run the Notebook:**
    * Open and execute `fakenewsdetection.ipynb` using Jupyter Notebook or JupyterLab.

## Key Dependencies

* pandas
* numpy
* scikit-learn
* nltk
* matplotlib
* seaborn
* plotly
* wordcloud
* textstat

