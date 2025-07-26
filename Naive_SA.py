#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from nltk.corpus import stopwords
import re
import os
import math



class NaiveBayesClassifier:
    
    """
    
      A Naive Bayes classifier for movie review sentiment analysis.
      Classifies reviews as positive (1) or negative (0).       
      
    """
    
    def __init__(self, D = 600, D_pos = 300, D_neg = 300):
        """
        Initializes the classifier with total and per class document counts,

        Parameters
        ----------
        D : int
            The total number of documents. 
        D_pos : int
            Number of positive documents. The default is 300.
        D_neg : int
            Number of negative documents. The default is 300.


        """
        
        self.__D = D
        self.__D_pos = D_pos
        self.__D_neg = D_neg
        
    
    def process_data(self, reviews):
        """
        Removes noise  (punctutation, stopwords etc.) from the datasets.

        Parameters
        ----------
        reviews : list of str
            List of raw review strings.

        Returns
        -------
        list of list of str
            Tokenized and cleaned words per review.

        """
        stopWords = set(stopwords.words('english'))
        cleaned_reviews = []
        for review in reviews:
           review = re.sub(r'[^\w\s]', '', review)
           review = review.replace('\n', ' ')
           review = review.strip().lower()
           words = [w for w in review.split() if w not in stopWords]
           if words:
               cleaned_reviews.append(words)
        return cleaned_reviews 



    
    def process_files(self, folder_name):
        """
        Reads all reviews from a folder and returns as a list.

        Parameters
        ----------
        folder_name : str
            DFolder containing text files of negative and postive reviews.

        Returns
        -------
        all_reviews : list of str, str
            All reviwes on per line
        folder_name : str
            the folder name

        """
        all_reviews = []
        files = os.listdir(folder_name)
        for filename in files:
            if filename.endswith('.txt'):
                filepath = os.path.join(folder_name, filename)
                with open(filepath, "r") as file:
                    for line in file:
                        line = line.strip()
                        if line:
                            all_reviews.append(line)
        return all_reviews, folder_name  


    def split_dataset(self, list_reviews):
        """
        Split the datasets into 80% training subset and 20% testing subset.

        Parameters
        ----------
        list_reviews : list of str
            List of reviews 

        Returns
        -------
        training_set : list of str
            training reviwes 
        testing_set : list of str
            testing reviews

        """
        split_index = int(0.8 * len(list_reviews))
        training_set = list_reviews[:split_index]
        testing_set = list_reviews[split_index:]
        return training_set, testing_set
    
    
    
    def count_reviews(self, list_reviews, folder_name):
        """
        Cleans all reviews and counts word occurrences per class label.
        
        Parameters
        ----------
        list_reviews : a list of str
             Raw reviews
        folder_name : str
             'pos' or 'neg' to determine label.
             

        Returns
        -------
        dict: dictionary of reviwes labeled as 0 and 1

        """
        #clean data and make a list of words 
        list_reviews = self.process_data(list_reviews)
        class_label = {}
        class_value = 1 if folder_name == 'pos' else 0
        for word in list_reviews:
            for w in word:
                key = (w, class_value)
                class_label[key] = class_label.get(key, 0) + 1 # count the occurance.
        return class_label

    
    def train_naiveBayes(self, training_pos, training_neg, reviews_pos, reviews_neg):
        """
        Trains the naive Bayes model: calculates logprior and loglikelihood.

        Parameters
        ----------
        training_pos : dict of str
            Word counts for postitve classes
        training_neg : dict of str
            Word counts for negative classes
        reviews_pos : list of str
            positve reviews
        reviews_neg : list of str
            negative revoews

        Returns
        -------
        logprior : float
            The difference between the prior probabilities of each class.
        loglikelihood : dict
            Word loglikelihoods.

        """
        
        prior_prob_pos = self.__D_pos / self.__D
        prior_prob_neg = self.__D_neg / self.__D
        
        # Create a vocab for both classes
        tokenized_pos = self.process_data(reviews_pos)  # list of lists
        tokenized_neg = self.process_data(reviews_neg)  # list of lists
        all_words = [word for review in tokenized_pos for word in review] + \
                    [word for review in tokenized_neg for word in review]

        V = set(all_words)
        V_size = len(V)
        N_pos = sum(training_pos.values())
        N_neg = sum(training_neg.values())
        
        loglikelihood = {} # containing loglikelihoood for each word 
        for word in V:
            freq_pos = training_pos.get((word, 1), 0)
            freq_neg = training_neg.get((word, 0), 0)
            
            # Add-1 smoothing 
            prob_word_pos = (freq_pos + 1) / (N_pos + V_size)
            prob_word_neg = (freq_neg + 1) / (N_neg + V_size)
            
            
            # Store loglikelihood for each word
            loglikelihood[word] = math.log(prob_word_pos / prob_word_neg)
            #loglikelihood[word] = math.log(prob_word_neg)
        
        logprior = math.log(prior_prob_pos) - math.log(prior_prob_neg)
        
        return logprior, loglikelihood
    
    
    
    def predict_naiveBayes(self, review, logprior, loglikelihood):
        """
        Predicts sentiment for review

        Parameters
        ----------
        review : list of str
            Cleaned words in the review.
        logprior : float
            Prior log-odds
        loglikelihood : dict 
            Word loglikelihoods
            

        Returns
        -------
        score : float
            The probability that review belongs to the positve or negative class.

        """
        
        score = logprior

        for word in review:
            if word in loglikelihood:
                score += loglikelihood[word]
        return score
    
    
    
    def test_naiveBayes(self, test_data_pos, test_data_neg,  logprior, loglikelihood):
        """
        Calculates classifier accuracy on test set.
        
        
        
        Parameters
        ----------
        test_data_pos : list of list of str
           List of (review)
        test_data_neg : list of list of str
            List of (review)
       logprior : float
           Prior log-odds.
       loglikelihood : dict
           Word log-likelihoods.

        Returns
        -------
        float: 
            Classification accuracy.

        """
        correct_pred = 0
        
        # remove empty  reviews for calculating total length
        test_data_pos_nonempty = [r for r in test_data_pos if len(r) > 0]
        test_data_neg_nonempty = [r for r in test_data_neg if len(r) > 0]

        total = len(test_data_pos_nonempty) + len(test_data_neg_nonempty)
        for review in self.process_data(test_data_pos):
            label = 1
            predict_score = self.predict_naiveBayes(review, logprior, loglikelihood)
            predict_label = 1 if predict_score > 0 else 0
            if label == predict_label:
                correct_pred += 1
        
        for review in self.process_data(test_data_neg):
            label = 0
            predict_score = self.predict_naiveBayes(review, logprior, loglikelihood)
            predict_label = 1 if predict_score > 0 else 0
            if label == predict_label:
                correct_pred += 1
       
        accuracy = correct_pred / float(total)
        return accuracy
            
            
    
    def error_analysis(self, test_data_pos, test_data_neg, logprior, loglikelihood):
        """
        Writes an error table of misclassified reviews to an output file called error_analysis.txt
        
        
        Returns
        -------
        None.
        """
        table_pos = []
        for review in self.process_data(test_data_pos):
            actual_label = 1
            predict_score = self.predict_naiveBayes(review, logprior, loglikelihood)
            predict_label = 1 if predict_score > 0 else 0
            if actual_label != predict_label:
                table_pos.append(f"{actual_label}        {predict_label}            {' '.join(review)}\n ")
        table_neg = []
        for review in self.process_data(test_data_neg):
            actual_label = 0
            predict_score = self.predict_naiveBayes(review, logprior, loglikelihood)
            predict_label = 1 if predict_score > 0 else 0
            if actual_label != predict_label:
                table_neg.append(f"{actual_label}        {predict_label}            {' '.join(review)}\n")
    
    
             
        with open('error_analysis.txt', 'w', encoding='utf8') as file:
            file.write("Truth    Predicted    Review \n")
            for entry_pos, entry_neg in zip(table_pos, table_neg):
                 file.write(entry_pos + '\n')
                 file.write(entry_neg + '\n')

        
                
                
            
        
    def predict_sentiment(self, filename, logprior, loglikelihood):
        """
        Prints the sentiment of review to a text file LionKing_Output.txt.

        Parameters
        ----------
        filename : File
            LionKing_moviesReviwes.

        Returns
        -------
        None.

        """
        with open(filename, 'r') as file:
            text = file.read()
            reviews = [text]
            cleaned = self.process_data(reviews)
            for word in cleaned:
                score = self.predict_naiveBayes(word, logprior, loglikelihood)
    
            sentiment = 'Positve' if score > 0 else 'Negative'
            with open('LionKing_Output.txt', 'w') as out:
                out.write(f"Sentiment: {sentiment} (score: {score})\n")
                out.write("Review \n")
                out.write(text)
                
                

if __name__ == "__main__":
    
    
    
    classifier = NaiveBayesClassifier()
    
    # reading the moviews file 
    pos_reviews, folder_pos = classifier.process_files('pos')
    neg_reviews, folder_neg = classifier.process_files("neg") 
    
    # split dataset into train and test subsets
    training_set_pos,testing_set_pos = classifier.split_dataset(pos_reviews)
    training_set_neg, testing_set_neg = classifier.split_dataset(neg_reviews)
    
    # count_reviews for postive files as dict
    pos_class_train = classifier.count_reviews(training_set_pos, folder_pos)
    # count_reviwes for negative files as dict
    neg_class_train = classifier.count_reviews(training_set_neg, folder_neg)
   
    
    # train
    logprior, loglikelihood = classifier.train_naiveBayes(pos_class_train, neg_class_train, pos_reviews, neg_reviews)
    # test
    accuracy = classifier.test_naiveBayes(testing_set_pos, testing_set_neg,  logprior, loglikelihood) * 100
    print(f'Naive Bayes accuracy = {accuracy:.4f}%')
    #classifier.error_analysis(testing_set_pos, testing_set_neg, logprior, loglikelihood)
    #classifier.predict_sentiment('LionKing_MovieReviews.txt', logprior, loglikelihood)
   

    
    
    
    

    