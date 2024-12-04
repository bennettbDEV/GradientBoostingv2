import csv
import math
from collections import Counter, defaultdict

import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from models import GradientBoostingClassifier


# Data processing
def process_and_save_data(input_file, output_file):
    def _calculate_tf_idf(words, complete_vocab, document_count, document_freq):
        word_counts = Counter(words)
        total_words = len(words)

        tf = {word: count / total_words for word, count in word_counts.items()}

        # Calculate TF-IDF
        current_email_vector = []
        for word in complete_vocab:
            # Get TF value for word, 0 if word not in email
            tf_value = tf.get(word, 0)
            # Add 1 to avoid division by zero
            idf = math.log(document_count / (1 + document_freq[word]))
            current_email_vector.append(tf_value * idf)
        return current_email_vector

    vocab = set()
    email_data = []
    word_in_doc_count = defaultdict(int)

    # Take in data from the csv
    with open(input_file, mode="r") as file:
        csvFile = csv.reader(file)

        # Extract field names through first row
        _ = next(csvFile)

        for line in csvFile:
            # Email data is first element in line. Tokenize and remove nonalphanumeric characters from text
            text = line[0].lower()
            text = "".join(char for char in text if char.isalpha() or char == " ")
            words = text.split()

            word_set = set(words)

            # Add words to vocab set for later use
            vocab.update(word_set)

            for word in word_set:
                word_in_doc_count[word] += 1

            # Create tuple with the list of words comprising the email, and the label if its spam or not. 1 = spam, 0 = non-spam
            text_label_pair = (words, line[1])
            email_data.append(text_label_pair)

    email_count = len(email_data)

    email_vectors = []
    labels = []

    # Calculate TF-IDF for each email with respect to words in vocab
    for words, label in email_data:
        # Calculate TF-IDF
        current_email_vector = _calculate_tf_idf(
            words, vocab, email_count, word_in_doc_count
        )

        # Add vectors x and labels y
        email_vectors.append(current_email_vector)
        labels.append(int(label))

    # Turn arrays into numpy arrays for more functionality
    email_vectors = np.array(email_vectors)
    labels = np.array(labels)

    # Use SMOTE (Synthetic Minority Oversampling Technique) to oversample & undersample the data set to balence it 
    # I.E to balence the number of spam and non spam emails
    smote_enn = SMOTEENN(random_state=7)
    email_vectors, labels = smote_enn.fit_resample(email_vectors, labels)

    # Save data
    np.savez(output_file, email_vectors=email_vectors, labels=labels)

# Processed data retrieval
def load_data(file_path):
    data = np.load(file_path)

    # Extract the email vectors and labels
    email_vectors = data["email_vectors"]
    labels = data["labels"]

    return email_vectors, labels


def use_XGBoost(training, testing, training_labels):
    # Using library:
    model = XGBClassifier(
        # All hyperparameters are assigned typical values for initial training
        n_estimators=100,  # Number of trees
        learning_rate=0.1,  # Step size shrinkage
        max_depth=3,  # Maximum depth of a tree
        subsample=0.8,  # Fraction of samples used for each tree
        colsample_bytree=0.8,  # Fraction of features used for each tree
    )
    model.fit(training, training_labels)

    # Make predictions
    predictions = model.predict(testing)
    return predictions



def main():
    # Uncomment below line if running for first time
    process_and_save_data("emails.csv", "output.npz")
    email_vectors, labels = load_data("output.npz")

    # Split data into training and testing groups
    training, testing, training_labels, testing_labels = train_test_split(
        email_vectors, labels, test_size=0.25, stratify=labels, random_state=7
    )

    # Appying the algorithms:
    # Using XGBoost
    xgboost_predictions = use_XGBoost(training, testing, training_labels)

    # Using basic GradientBoost implementation
    model = GradientBoostingClassifier(
        iterations=12, max_depth=3
    )  # In this case, 12 iterations is around when the model stops improving
    model.fit(training, training_labels)
    simple_gboost_predictions = model.predict(testing)

    # Evaluate the models
    print("XGBoost Classification Report:")
    print(
        classification_report(
            testing_labels,
            xgboost_predictions,
            labels=[0, 1],
            target_names=["Non-Spam","Spam"],
            zero_division=0,
        )
    )
    accuracy = accuracy_score(testing_labels, xgboost_predictions)
    print(f"XGBoost Accuracy: {accuracy:.2f}")


    print("Simple Gradient Boost Classification Report:")
    print(
        classification_report(
            testing_labels,
            simple_gboost_predictions,
            labels=[0, 1],
            target_names=["Non-Spam","Spam"],
            zero_division=0,
        )
    )
    accuracy = accuracy_score(testing_labels, simple_gboost_predictions)
    print(f"Simple Gradient Boost Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()