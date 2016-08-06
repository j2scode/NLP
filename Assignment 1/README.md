# Introduction to Natural Language Processing
## Introduction
One task of natural language processing (NLP) is to transform text in natural language into representations that computers can use to perform many different tasks such as error correction, machine translation, information extraction, predictive text processing and interactive question answering.  Dividing text into sentences and then into words, assigning parts of speech to words, and deriving syntactic and semantic representations for sentences are among the processes involved in allowing computers to “understand” and manipulate text.  In NLP, data driven, transition-based, dependency parsing is the problem of taking sentences and determining which parts depend on others, and in what way.  

Transition-based dependency parsing models the parsing problem as a series of decisions or, transitions, to be made between parsing configurations.  A configuration is comprised of:
-	Buffer - an input buffer containing all the unprocessed words in a sentence
-	Stack – holding all the partially processed words in a sentence
-	Arc Set – a set of head and dependent arcs, along with a dependency label, that defines the current tree

The objective of the parser is to determine the “transitions” to move from an initial configuration, an unprocessed sentence, to a terminal configuration in which the sentence is processed and the Arc Set contains all dependencies from which a dependence graph can be rendered.  The challenge is to determine which action or transition should be taken in each of the unboundedly many states encountered as the parser progresses.  

Combined with treebank-induced classifiers, memory based learners or support vector machines, dependency parsing algorithms can be used to create accurate disambiguating parsers, in particular for dependency-based syntactic representations. 

## Project Overview
This project examines the performance accuracy and complexity of Joakim Nivre’s Arc-Eager Transition-Based Dependency Parser (AE), one of the most widely used dependency parsers.  Danish, English, and Swedish training and test data sets were provided by the CoNLL-X shared task on multilingual dependency parsing.  The CoNLL data format is a tab-separated text file, where the ten fields were: 

1. ID - a token counter, which restarts at 1 for each new sentence 
2. FORM - the word form, or a punctuation symbol 
3. LEMMA - the lemma or the stem of the word form, or an underscore if this is not available 
4. CPOSTAG - course-grained part-of-speech tag 
5. POSTAG - fine-grained part-of-speech tag 
6. FEATS - unordered set of additional syntactic features, separated by | 
7. HEAD - the head of the current token, either an ID or 0 if the token links to the root node. The data is not guaranteed to be projective, so multiple HEADs may be 0. 
8. DEPREL - the type of dependency relation to the HEAD. The set of dependency relations depends on the treebank. 
9. PHEAD - the projective head of the current token. 
10. PDEPREL - the dependency relationship to the PHEAD, if available.

Support Vector Machines learned the Danish, English, and Swedish language models used by the AE Parser on the test data.  I measured the performance in terms of the unlabeled and labeled attachment scores (UAS and LAS, respectively).  

## Assignment 1: Implement Transition Operations
### Purpose
The purpose of this assignment was to implement the four AE operations: left_arc, right_arc, shift, and reduce.  The operations are summarized as follows:
-	Shift (pushes) the next input token onto the stack λ1. 
-	Reduce (pops) the token on top of the stack λ1. It is important to ensure that the parser does not pop the top token if it has not been assigned a head, since it will be left unattached.
-	Right-Arc transition adds an edge from the token on top of the stack λ1 to the next input token and involves pushing the token onto the stack. 
-	Left-Arc adds an edge from the next input token to the token on top of the stack λ1 and  involves popping the token from the stack. This transition is only allowed when the top token is not the root node and is not dependent on any other node.

### Features
A total of eight features were trained, four features pertaining to the node on the top of the stack and four features associated with the next node in the input buffer and they were:
-	FORM - the word form, or a punctuation symbol 
-	FEATS - unordered set of additional syntactic features, separated by | 
-	LDEP – left most dependent
-	RDEP – right most dependent

### Performance
The algorithm was executed on the Swedish training and test sets.  The unlabeled and labeled attachment scores, UAS, and LAS respectively were: 
-	UAS: 0.23819956184
-	LAS: 0.129456283609

The rather abysmal scores suggest that new features be added to the model, which we examine in the next assignment.
## Assignment 2: Feature Selection and Implementation
### Purpose
The purpose of this assignment is to explore the relationship between features and model performance.  

### Features
To improve performance, I added the following new features:
#### Stack [0] Features
- LEMMA - the lemma or the stem of the word form, or an underscore if this is not available 
- CPOSTAG - course-grained part-of-speech tag 
- POSTAG - fine-grained part-of-speech tag

## Stack[1] Features
- FORM - the word form, or a punctuation symbol
- LEMMA - the lemma or the stem of the word form, or an underscore if this is not available 
- CPOSTAG - course-grained part-of-speech tag 
- POSTAG - fine-grained part-of-speech tag
-	FEATS - unordered set of additional syntactic features, separated by | 
	
#### Buffer [0] Features
- LEMMA - the lemma or the stem of the word form, or an underscore if this is not available 
- CPOSTAG - course-grained part-of-speech tag 
- POSTAG - fine-grained part-of-speech tag

#### Buffer [1] Features
- FORM - the word form, or a punctuation symbol
- LEMMA - the lemma or the stem of the word form, or an underscore if this is not available 
- CPOSTAG - course-grained part-of-speech tag 
- POSTAG - fine-grained part-of-speech tag
-	FEATS - unordered set of additional syntactic features, separated by | 

#### Buffer[2] Features
- FORM - the word form, or a punctuation symbol
- LEMMA - the lemma or the stem of the word form, or an underscore if this is not available 
- CPOSTAG - course-grained part-of-speech tag 
- POSTAG - fine-grained part-of-speech tag
-	FEATS - unordered set of additional syntactic features, separated by | 

### Performance
The new features were implemented for the English, Swedish and Danish data sets.  The following summarizes labeled and unlabeled attachment scores.

#### Danish Model
- UAS 0.791616766467
- LAS 0.705988023952	

#### English Model
- UAS 0.830188679245
- LAS 0.767295597484

#### Swedish Model
- UAS 0.772555267875
- LAS 0.656442939653	|  

### Complexity
The Arc-Eager Transition-Based Dependency Parser (AE Parser) has an overall time and space, best and worse-case complexity of O(n), where n is the number of nodes (words) in the input sentence.  By disallowing non-projective dependency arcs, the AE Parser algorithm skips many of the node pairs that are considered by non-projective algorithms, thereby optimizing time and space measures.  

The features added in assignment 2 are so-called “static” features, in that their values do not change for a word during processing.  Dynamic features must be stored and indexed for each transition, thereby increasing computational cost.  Unlike dynamic features, the static features are memory resident or stored locally where they can be retrieved quickly.  Thusly, significant performance accuracy improvements were achieved with no increase in computational complexity.
