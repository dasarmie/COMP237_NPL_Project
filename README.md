# COMP237_NPL_Project
Text classifer - detect spam/not spam email

Build a simple text classifier using the  **"Bag of words"**  language model & the **Navie Bayes classifier** to detect if an email is spam/not spam. Many spammers and scammers are very good at making junk and spam email that may seem believable. The purpose of your model is to filter the spam comments by training an Naive Bayes classifier.

### Requeriments

1. Load the data into a pandas data frame.
   
2. Carry out some basic data exploration and present your results. (Note: You only need two columns for this project, make sure you identify them correctly, if any doubts ask your professor).
   
3. Using *nltk toolkit* classes and methods prepare the data for model building, refer to the third lab tutorial in module 11 (Building a Category text predictor ). Use count_vectorizer.fit_transform().
   
4. Present highlights of the output (initial features) such as the new shape of the data and any other useful information before proceeding.
   
5. Downscale the transformed data using *tf-idf* and again present highlights of the output (final features) such as the new shape of the data and any other useful information before proceeding.
   
6. Use pandas.sample to shuffle the dataset, set frac =1.

7. Using pandas split your dataset into 75% for training and 25% for testing, make sure to separate the class from the feature(s). (Do not use test_train_ split).

8. Fit the training data into a *Naive Bayes classifier*.

9. Cross validate the model on the training data using 5-fold and print the mean results of model accuracy.

10. Test the model on the test data, print the confusion matrix and the accuracy of the model.

11. As a group come up with 6 new comments (4 comments should be non spam and 2 comment spam) and pass them to the classifier and check the results. You can be very creative and even do more :grinning:.

12. Present all the results and conclusions.

13. Drop code, report and power point presentation into the project assessment folder for grading.
