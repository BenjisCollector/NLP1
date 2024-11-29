# Advanced Sentiment Analysis: Naive Bayes Classifier and Rule-Based Methods

## Project Overview
This project demonstrates a comprehensive approach to sentiment analysis using both Naive Bayes and rule-based models on datasets consisting of movie reviews and Nokia product feedback. The focus is on comparing these models to assess their effectiveness in understanding nuanced language, particularly handling negations and intensifiers in sentiment analysis.

## Objective
The primary objective is to compare two different sentiment analysis methods:
1. A Naive Bayes classifier that uses statistical probabilities.
2. An enhanced rule-based classifier that incorporates handling for negations and intensifiers to improve the accuracy of sentiment predictions.

The aim is to demonstrate advanced NLP techniques and the application of Python in developing and evaluating models that can predict sentiment with high accuracy from textual data.

## Technical Approach
- **Data Preparation:** Text data is preprocessed using regular expressions to handle various data sources and split into training and test datasets.
- **Feature Engineering:** Sentiment scoring is applied to convert words into numerical data using predefined positive and negative word lists.
- **Model Training:** Both a Naive Bayes classifier and a rule-based classifier are trained with methods suited to their respective paradigms.
- **Model Evaluation:** Both models are evaluated on unseen data. Key performance metrics such as accuracy, precision, recall, and F1-score are used to compare their effectiveness.

## Detailed Workflow
1. **Reading Data:** The data consists of labelled sentences from movie reviews and Nokia product feedback.
2. **Sentiment Dictionary Construction:** Positive and negative word lists are compiled into a sentiment dictionary.
3. **Dataset Splitting:** Data is partitioned into training and testing sets to validate the models' generalization.
4. **Model Training and Testing:**
   - **Naive Bayes Classifier:** Uses probability calculations based on word frequencies.
   - **Rule-Based Classifier:** Applies logical rules to handle negations and intensifiers, enhancing the understanding of context.
5. **Performance Comparison:** The outcomes of the Naive Bayes and rule-based classifiers are compared to determine which method better handles complex language nuances.

## Technologies & Libraries Used
- **Python:** Main programming language.
- **Pandas & NumPy:** For data manipulation and numerical operations.
- **Matplotlib & Seaborn:** For visualisation of the models’ performance.
- **Regular Expressions:** Essential for text data preprocessing.

## Conclusion & Key Takeaways
This analysis highlights the strengths and limitations of both probabilistic and rule-based approaches in sentiment analysis. The project shows how different methodologies can be tailored to enhance model performance, especially in handling linguistic subtleties such as negations ("not great") which are crucial for accurate sentiment analysis. This comparison not only deepens the understanding of NLP applications but also showcases potential enhancements for more robust sentiment analysis models.

### Note
This project is positioned as a high-value addition to a CV, demonstrating not just technical NLP skills but also the ability to engage in critical analysis and methodology comparison in machine learning.
