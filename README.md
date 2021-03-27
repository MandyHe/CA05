# Name of Project

ML_CA05_A_Logistic Regression


# Project Overview

Cardiovascular Disease (CVD) kills more people than cancer globally. A dataset of real heart patients collected from a 15 year heart study cohort is made available for this assignment. The dataset has 16 patient features. Note that none of the features include any Blood Test information.
• Number of attributes (Columns): 17
• Number of instances (Rows): 3242

# No need to download dataset, the code contain data url

1. CVD dataset: https://github.com/ArinB/CA05-B-Logistic-Regression/raw/master/cvd_data.csv

# Open colab:
https://colab.research.google.com/drive/1Eh7xIi5id5-uKLwaw9IlL7poPM2feA_D?usp=sharing

# Import packages list below:

```bash
#Import packages
import pandas as pd
import numpy as np

#Import viz packages
import matplotlib.pyplot as plt
import seaborn as sns

#Import linear model package
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression


#Import decision tree packages
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
```


# Installation

- Mounted the google drive to make files readable: drive.mount('/content/drive/')
- Enjoy the code


# Contact info

Name: Mandy He
Email: she3@lion.lmu.edu

