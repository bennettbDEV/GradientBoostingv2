<!-- ABOUT THE PROJECT -->
## About The Project
While taking an Intro to AI course at my university, I wanted to go a bit beyond the content presented
and look into an additional machine learning concept. I did some research and found that Gradient Boosting was a commonly used ML framework that is both accurate and flexible. Subsequently, I decided to make an informal write up about the subject and use the technique to classify emails as spam or non-spam.

<br />


## Code Overview
### The Model:
To implement a simple version of a Gradient Boosting Binary Classifier, I used the following pseudo-code as an outline:

<a href="https://en.wikipedia.org/wiki/Gradient_boosting">
    <img src="images\GradientBoostingPsuedoCode_Wikipedia.png" alt="Gradient Boosting Psuedocode">
  </a>

My implementation is in the models.py file and uses a [Decision Tree Regressor](https://scikit-learn.org/dev/modules/generated/sklearn.tree.DecisionTreeRegressor.html), from the scikit-learn library, as the weak learner. 

### Using Gradient Boosting for Spam Email Classification
To build and evaluate my model, I first needed a suitable dataset for training and testing. To do this, I utilized [this](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset) Spam Email Dataset from Kaggle.

Next, I preprocessed the email data by removing non-alphanumeric characters, tokenizing each email, and transforming the text into [TF-IDF](https://builtin.com/articles/tf-idf) feature vectors based on the overall vocab.

To address the potential imbalance between the number of spam and non-spam emails, I applied the [Synthetic Minority Oversampling Technique (SMOTE)](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/) using the [imblearn](https://imbalanced-learn.org/stable/) package. This technique balanced the dataset by oversampling the minority class and undersampling the majority class.

With the data processed, I trained and tested my custom gradient boosting model and also evaluated [XGBoost (eXtreme Gradient Boosting)](https://xgboost.readthedocs.io/en/stable/), which is a widely-used gradient-boosted decision tree library, for comparison.

### The Results
Below, you’ll find 15 iterations of my Gradient Boosting model along with the corresponding loss for each iteration. Additionally, the testing results for both XGBoost and my model are presented. Despite my implementation being much slower, it is almost as accurate as XGBoost, which helps illustrate how powerful Gradient Boosting can be. 


<a>
    <img src="images\Console_Output.png" alt="Gradient Boosting Psuedocode" height=600>
</a>


## Run the Code
To install the dependencies and run the code for yourself, follow these simple steps in your terminal:

1. Clone the repo (into the currect directory)
```sh
git clone https://github.com/bennettbDEV/GradientBoostingv2.git .
```
2. (Optionally) Set up a [virtual environment](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/)
```sh
python -m venv venv
```
2a. Activate the virtual environment - for Windows:
```sh
.\venv\Scripts\activate
```
2b. Activate the virtual environment - for Linux/Mac:
```sh
source venv/bin/activate
```
3. Install necessary packages
```sh
python -m pip install -r requirements.txt
```
4. Run the program!
```sh
python main.py
```
