Download Yelp data using this link: https://drive.google.com/file/d/1vURehJBBoTQCwz7RT2F2MlplWx_xXtap/view?usp=sharing

Run classifier.py with no arguments and it will produce one file called words.txt and classifier.pickle for futures 
uses. words.txt is the 2000 most common words from the given data, and classifier.pickle is the classifier. 
classifier.py also splits the data into 80-20 train and test sets. It will give you the confusion matrix as well as the 
classification report. You can run classify.py with a similar dataset to produce the predictions given a json file with 
the correct inputs. classify.py will use the words.txt and classifier.pickle to predict the the stars if stars isn't one 
of the inputs for the json file.