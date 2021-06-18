# SummVD's GitHub page

**This code runs with Python 3.8**

## To run the experiments like in the paper we have built a user-friendly script :

1. Download via the google drive the datasets, word embedding and parameters : [Drive folder](https://drive.google.com/drive/folders/1QjobC4w9G7nd2eva5sURUQ5Ys3s93gan?usp=sharing)  
**You need to keep the same file structure**, which means that you have to put every file/folder like in the google drive in a folder called SummVD.

2. Download the github SummVD.py file, and put it to the **root of the SummVD folder**.

3. You need to have the packages from the requirements.txt with at least the same version on the most important packages which are : **gensim==3.8.3 / scipy==1.6.3 / rouge-score==0.0.4 / sklearn == 0.17.2 / nltk == 3.5 / numpy == 1.19.2**

4. You can now run the experiments from the root of your SummVD folder as follow :  
```
"python SummVD.py --dataset_name "dataset_name" --size_output 0"  
```
**With dataset_name one of the following :** "duc", "reddit", "cnn", "reddit_tifu", "MN" (for Multi-News), "pubmed", "xsum". (size_output 0 means that we test the full length dataset. To run n examples, simply put n instead).

**Step 5** : As you will see, the output summaries are written in the output folder like : "dataset_name.txt". From here you can run our Rouge scorer by using : 
```
"python SummVD.py --dataset_name "dataset_name" --scoring True"
```






## Experimental functions :

To run the full length dataset, for size_output put the value 0, a custom value will take the n first documents and abstracts of the dataset.

If you want to try a custom dataset, you will have to call it : dataset_name + "_documents.pkl" and the gold : dataset_name + "_gold.pkl". You will then be able to run it with :  
```
--dataset_name "dataset_name" --nb_sentences n --optimisation_rate xxx  
```
Optimisation rate (optionnal) means xxx * len(documents) optimisation of parameters. By default it is set to 0.004. Arrange yourself to have at least 15 or more examples in order to get the best results.  
The .pkl data has to work with :  
```
documents = pickle.load(open(dataset_name + "_documents.pkl", "rb"))
gold = pickle.load(open(dataset_name + "_gold.pkl", "rb"))
```

If you want to try a custom word embedding, it has to be in .pkl format. it has to work as follow : wordembedding["word"] => vector.  
In that case, simply add :  
```
--word_embedding_path "./Word Embedding Model/my_word_embedding.pkl"
```

If you only want to use our Rouge scoring you will have to use : python SummVD.py --dataset_name "dataset_name" --scoring True  
With the dataset added as dataset_name + "_documents.pkl" and the gold : dataset_name + "_gold.pkl" in Datasets folder.  
Your summaries will have to be in output folder as "dataset_name.txt"

If parameters already exist for a dataset, it will automatically choose it.  
We present here the parameters associated with the number of sentences announced in the paper.  
Better results can be achieved with different number of sentences, refered as "n", as talked about in the paper.  
To obtain those results, you will have to run the optimisation with the number of sentences wanted.
And if you want to do your own optimisation, don't copy the parameters file in your repository. The method will create it automatically for you.  
In this case you can add to the original script the following :  
```
--optimisation_rate 0.004 --nb_sentences n  
```


<!--
**SummVD/SummVD** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
