import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error

# Load the dataset
data = pd.read_csv('amazon_baby.csv')

# Preprocess the data
data['label'] = data['rating'].apply(lambda x: 'positive' if x > 3 else 'negative' if x < 3 else 'neutral')
data = data[['review', 'label']]

# Handle NaN values in the 'review' column
data['review'].fillna('', inplace=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['label'], test_size=0.2, random_state=42)

# Convert reviews to numerical features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict labels and convert to ratings
def convert_label_to_rating(label):
    if label == 'positive':
        return np.random.choice([4, 5])
    elif label == 'negative':
        return np.random.choice([1, 2])
    else:
        return 3

predictions = knn.predict(X_test)
ratings = [convert_label_to_rating(label) for label in predictions]

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(classification_report(y_test, predictions))

# Print model parameters
print(f'Parameter size: {knn.get_params()["n_neighbors"]}')

# Confusion Matrix Plot
plt.figure(figsize=(8, 6))
confusion_mat = confusion_matrix(y_test, predictions)
sns.heatmap(confusion_mat, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Distribution of Predicted Ratings Plot
plt.figure(figsize=(8, 6))
plt.hist(ratings, bins=5, edgecolor='black', alpha=0.7)
plt.xticks(np.arange(1, 6))
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Distribution of Predicted Ratings')
plt.show()

# Distribution of True Ratings Plot
plt.figure(figsize=(8, 6))
plt.hist(data['rating'], bins=5, edgecolor='black', alpha=0.7)
plt.xticks(np.arange(1, 6))
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Distribution of True Ratings')
plt.show()

# Ratio of Training and Test Data Plot
train_size = len(X_train)
test_size = len(X_test)
total_size = train_size + test_size

plt.figure(figsize=(6, 6))
labels = ['Training Data', 'Test Data']
sizes = [train_size, test_size]
colors = ['gold', 'lightcoral']
explode = (0.1, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Ratio of Training and Test Data')
plt.show()

# Analyze positive and negative reviews
positive_reviews = data[data['label'] == 'positive']['review']
negative_reviews = data[data['label'] == 'negative']['review']

print('Top 5 Positive Reviews:')
print('\n'.join(positive_reviews.head(5)))
print('\nTop 5 Negative Reviews:')
print('\n'.join(negative_reviews.head(5)))
