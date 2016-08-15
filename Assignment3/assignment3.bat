ECHO Training and predicting using KNN and SVM algorithms
python main.py data/English-train.xml data/English-dev.xml KNN-English.answer SVM-English.answer bestfile English
python main.py data/Spanish-train.xml data/Spanish-dev.xml KNN-Spanish.answer SVM-Spanish.answer bestfile Spanish
python main.py data/Catalan-train.xml data/Catalan-dev.xml KNN-Catalan.answer SVM-Catalan.answer bestfile Catalan

ECHO Evaluating KNN results
scorer2 KNN-English.answer data/English-dev.key data/English.sensemap
scorer2 KNN-Spanish.answer data/Spanish-dev.key 
scorer2 KNN-Catalan.answer data/Catalan-dev.key 

ECHO Evaluating SVM results
scorer2 SVM-English.answer data/English-dev.key data/English.sensemap
scorer2 SVM-Spanish.answer data/Spanish-dev.key 
scorer2 SVM-Catalan.answer data/Catalan-dev.key 
