# Capstone Project: 

## Reduce Manufacturing Failures
### Project Overview

 Manufacturing Failures System are to help manufacturing companies, has an imperative to ensure that the recipes for the production of its advanced mechanical components are of the highest quality and safety standards. Parts of doing so is closely monitoring its parts as they progress through the manufacturing processes. Because companies records data at every step along its assembly lines, they have the ability to apply advance analytics to improve these manufacturing processes. If the companies like Bosh use the manufacturing failures system it intimates the failure during the manufacturing process. 
In this project, we will build a manufacturing failure systems to predict internal failures using thousands of measurements and tests made for each component along the assembly line. This would enable Bosh to bring quality products at lower costs to the end user.

### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)


### Code

Template code is provided in the notebook `capstone_project.ipynb` notebook file. 
### Run

In a terminal or command window, navigate to the top-level project directory `Capstone_project/` (that contains this README) and run one of the following commands:

```ipython notebook capstone_project.ipynb```  
```jupyter notebook capstone_project.ipynb```

This will open the iPython Notebook software and project file in your browser.

## Data

The dataset used in this project is `train_numeric.csv` which is downloaded from link (https://www.kaggle.com/c/bosch-production-line-performance/download/train_numeric.csv.zip) unzip the file to get dataset. This dataset has the following attributes:

The data for this project represents measurements of parts as they move through Bosch's production lines. Each part has a unique Id. The goal is to predict which parts will fail quality control (represented by a 'Response' = 1).
The dataset contains an extremely large number of anonymised features. Features are named according to a convention that tells you the production line, the station on the line, and a feature number. E.g. L3_S36_F3939 is a feature measured on line 3, station 36, and is feature number 3939.
On account of the large size of the dataset, we have separated the files by the type of feature they contain: numerical, categorical, and finally, a file with date features. The date features provide a timestamp for when each measurement was taken. Each date column ends in a number that corresponds to the previous feature number. E.g. the value of L0_S0_D1 is the time at which L0_S0_F0 was taken.
In addition to being one of the largest datasets (in terms of number of features) ever hosted on Kaggle, the ground truth for this competition is highly imbalanced. Together, these two attributes are expected to make this a challenging problem.

