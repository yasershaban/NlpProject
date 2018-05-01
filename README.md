# NlpProject
This project for NLP in FCIH
That classify news paper to categories [business, entertainment, sport, politic, tech] using BBC_DataSet_News_Classification
using nltk, sklearn, numpy, pandas

RUN PREREQUISITE
=================
nltk, scikit-klearn, scipy, numpy, pandas 

you should run nlp.py from terminal as python nlp.py

python nlp.py  ==> print classification_report for each algo [NaiveBayes, SGDClassifier]
                   when BBC_Dataset is the Train and Test sets
                   
python nlp.py "path/filename.txt"  ==> print the class of the news contain in the "filename.txt"
                                       you can use "test.txt" that found in the folder
