
# Amazon Baby Product Reviews Sentiment Analysis

This project applies a K-Nearest Neighbors (KNN) classifier to analyze sentiment in Amazon baby product reviews. The primary objective is to predict whether a review is positive, neutral, or negative based on the text content.

## Project Overview

The project involves the following steps:
1. **Data Loading and Preprocessing**: Load the dataset, handle missing values, and create sentiment labels.
2. **Feature Extraction**: Convert the text reviews into numerical features using TF-IDF vectorization.
3. **Model Training**: Train a KNN classifier on the processed data.
4. **Prediction and Evaluation**: Predict sentiment labels for the test set, evaluate the model's performance, and visualize the results.
5. **Visualization**: Generate plots to illustrate the distribution of predicted and true ratings, as well as the confusion matrix.

## Key Features

- **Data Handling**: Preprocess and clean the review data.
- **Feature Engineering**: Use TF-IDF vectorization for transforming text data.
- **Modeling**: Implement KNN for classification.
- **Evaluation**: Calculate accuracy, mean squared error, and generate a classification report.
- **Visualization**: Create visual representations of data distributions and model performance.

## Installation

To run this project, ensure you have the following packages installed:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install these packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/amazon-baby-reviews-sentiment-analysis.git
    cd amazon-baby-reviews-sentiment-analysis
    ```

2. Run the Python script:
    ```bash
    python sentiment_analysis.py
    ```

## Results

- **Accuracy**: The overall accuracy of the model.
- **Mean Squared Error**: The MSE of the model predictions.
- **Confusion Matrix**: A heatmap showing the confusion matrix.
- **Rating Distributions**: Histograms showing the distribution of predicted and true ratings.

## Conclusion

This project demonstrates a basic application of machine learning for sentiment analysis of product reviews. The KNN classifier provides a simple yet effective approach for classifying review sentiments.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Dataset: Amazon baby product reviews
- Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn
<h2 id="donation">Donation</h2>

<p>If you find this project helpful, consider making a donation:</p>
<p><a href="https://nowpayments.io/donation?api_key=REWCYVC-A1AMFK3-QNRS663-PKJSBD2&source=lk_donation&medium=referral" target="_blank">
     <img src="https://nowpayments.io/images/embeds/donation-button-black.svg" alt="Crypto donation button by NOWPayments">
</a></p>


