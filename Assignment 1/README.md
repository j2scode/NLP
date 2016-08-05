# Introduction to Natural Language Processing
## Assignment 1 Part 1
### Feature Set 1:
The first feature set contained the following features:

From the Stack
- Stack [0] Form – Word
- Stack [0] Features – a list of syntactic features
- Stack [0] Left Most Dependent 
- Stack [0] Right Most Dependent 

From the Buffer
- Buffer [0] Form – Word
- Buffer [0] Features – a list of syntactic features
- Buffer [0] Left Most Dependent 
- Buffer [0] Right Most Dependent 

### Performance of Feature Set 1
The performance was as follows:
- UAS: 0.23819956184 
- LAS: 0.129456283609

## Assignment 1 Part 2
### Feature Set
The following features were added to feature set 1:

Stack
- Stack [0] Lemma – stem of the word
- Stack [0] CTAG – course-grained part of speech tag
- Stack [0] TAG – fine-grained part of speech tag

### Performance of Feature Set 2
The performance of feature set 2 was as follows:
- UAS: 0.735112527385 
- LAS: 0.634335789683

### Complexity and Performance
The Arc-Eager Transition-Based Dependency Parser (AE Parser) has an overall time and space, best and worse-case complexity of O(n), where n is the number of nodes (words) in the input sentence.  By disallowing non-projective dependency arcs, the AE Parser algorithm skips many of the node pairs that are considered by non-projective algorithms, thereby optimizing time and space measures. 

The AE Parser performs well and is only slightly over-performed by the non-projective dependency parsers with a margin that is directly related to the proportion of non-projective dependencies in the data. 


