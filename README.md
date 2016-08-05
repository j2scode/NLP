# NLP
Introduction to Natural Language Processing 

##Assignment 1

###Performance of Feature Set 1:
The first feature set contained the following features:

From the Stack
•	Stack [0] Form – Word
•	Stack [0] Features – a list of syntactic features
•	Stack [0] Left Most Dependent 
•	Stack [0] Right Most Dependent 

From the Buffer
•	Buffer [0] Form – Word
•	Buffer [0] Features – a list of syntactic features
•	Buffer [0] Left Most Dependent 
•	Buffer [0] Right Most Dependent 

The performance was as follows:
•	UAS: 0.23819956184 
•	LAS: 0.129456283609

###Performance of Feature Set 2:
The following features were added to feature set 1:

Stack
•	Stack [0] Lemma – stem of the word
•	Stack [0] CTAG – course-grained part of speech tag
•	Stack [0] TAG – fine-grained part of speech tag

The performance of feature set 2 was as follows:
•	UAS: 0.735112527385 
•	LAS: 0.634335789683
