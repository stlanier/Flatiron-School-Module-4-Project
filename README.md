# Classifying Product Reviews by Sentiment

For my fourth module project for Flatiron School, I chose a natural language processing problem, trying to classify [product reviews](https://data.world/crowdflower/brands-and-product-emotions) as positive, negative, or neither. Key challenges when approaching this problem were the multi-class nature of the dataset and class imbalances––reviews without emotion were much more common than either positive or negative reviews. Because of class imbalances, I used F1 scores (rather than accuracy) to determine model success, and my best models achieved 0.53 F1 scores on test data.

I'm now attempting to build models that excel at detecting negative reviews, focusing on improving recall of negative reviews (WIP).

## Getting Started
### Contents of Repository

* **notebooks**
   * **preprocessing.ipynb** and **preprocessing.py** are the notebook and module where I perform all data preprocessing––tfidf vectorization, bigrams, SMOTE, etc. **_Run this notebook to generate the datasets used in this analysis_**.
   * **eda.ipynb** contains my exploratory data analysis, including explorations of word frequency distributions and n-grams.
   * **multiclass_models** and **analysis_util.py** are the notebook and module where I train models for 3-class classification (negative review, positive review, no sentiment).
   * **negative_review_detection** contains models for solely detecting negative reviews.
* **data** is a directory contain all raw data from [data.world](https://data.world/crowdflower/brands-and-product-emotions) and processed data from various stages of the project.
* **images** is a directory containing images used in this README.
* **presentation.pdf** contains my powerpoint presentation for a non-technical audience.

### Prerequisites

The standard packages for data analysis are required–[NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), and [Matplotlib](https://matplotlib.org/)––as well as the [Natural Language Toolkit](https://www.nltk.org/) for string data processing tools, [imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/api.html) for SMOTE, and [scikit-learn](https://scikit-learn.org/stable/) and [Keras](https://keras.io/) for classification models. Below are examples of their installations using Anaconda.

```
$ conda install -c anaconda numpy
$ conda install pandas
$ conda install -c conda-forge matplotlib
$ conda install -c anaconda nltk
$ conda install -c conda-forge imbalanced-learn
$ conda install scikit-learn
$ conda install -c conda-forge keras
```


## Built With

[Jupyter Notebook](https://jupyter.org) - Documents containing live code and visualizations.

## Contributing

Due to the nature of the assignment, this project is not open to contributions. If, however, after looking at the project you'd like to give advice to someone new to the field and eager to learn, please reach out to me at [stephen.t.lanier@gmail.com].

## Author

**Stephen Lanier** <br/>
[GitHub](https://github.com/stlanier) <br/>
[Personal Website](https://stlanier.github.io)



## Acknowledgments

<a href="https://flatironschool.com"><img src="images/flatiron.png" width="80" height="40"  alt="Flatiron School Logo"/></a>
Special thanks to Jacob Eli Thomas and Victor Geislinger, my instructors at [Flatiron School](https://flatironschool.com), for their encouragement, instruction, and guidance.
