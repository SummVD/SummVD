#%% Setup
from sklearn.cluster import AgglomerativeClustering   
from sklearn.cluster import KMeans
import numpy as np
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
from nltk.corpus import stopwords
import pickle
import time
from scipy.spatial import distance
import statistics
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import os
from sklearn.decomposition import PCA
from rouge_score import rouge_scorer
from operator import itemgetter
#import subprocess
import argparse


#%% Functions

def load_dataset(dataset_name):
    articles=[]
    abstracts = []
    summariesOracle = None
    R1 = None
    R2 = None
    RL = None
    
    if dataset_name in ["cnn","xsum","reddit_tifu", "pubmed", "MN", "duc", "reddit"]:

        
        if(dataset_name == "MN"): 
            file = open("./Datasets/MultiNewsArticles",'rb')
            articles = pickle.load(file)
            file = open("./Datasets/MultiNewsGold",'rb')
            abstracts = pickle.load(file)
                    
        if(dataset_name == "reddit"): 
            file = open("./Datasets/RedditArticles.pkl",'rb')
            articlesR = pickle.load(file)
            articles =[]
            for article in articlesR:
                articles.append(article[0])
            
            abstracts = []
            with open("./Datasets/RedditGold", "r", encoding = "utf-8") as fh:
                    for line in fh:
                        abstracts.append(line)
                        
        if(dataset_name == "reddit_tifu"): 
            file = open("./Datasets/RedditTifuArticles",'rb')
            articles = pickle.load(file)
            file = open("./Datasets/RedditTifuGold",'rb')
            abstracts = pickle.load(file)
                
        if(dataset_name == "duc"):
            file = open("./Datasets/DUCArticles",'rb')
            articles = pickle.load(file)
            file = open("./Datasets/DUCGold",'rb')
            abstracts = pickle.load(file)
        
        if(dataset_name == "xsum"): 
            file = open("./Datasets/XSumArticles",'rb')
            articles = pickle.load(file)
            file = open("./Datasets/XSumGold",'rb')
            abstracts = pickle.load(file)
                        
        if(dataset_name == "pubmed"): 
            file = open("./Datasets/PubMedArticles",'rb')
            articles = pickle.load(file)
            file = open("./Datasets/PubMedGold",'rb')
            abstracts = pickle.load(file)

            
        if(dataset_name == "cnn"): 
            file = open("./Datasets/CNNArticles",'rb')
            articles = pickle.load(file)
            file = open("./Datasets/CNNGold",'rb')
            abstracts = pickle.load(file)
                    
    else:

        file = open(str("./Datasets/"+dataset_name+"_documents.pkl"),'rb')
        articles = pickle.load(file)
        
        file = open(str("./Datasets/"+dataset_name+"_gold.pkl"),'rb')
        abstracts = pickle.load(file)
    
    
    articlesCl = []  
    for article in articles:
        articlesCl.append(article.replace("”", "").rstrip("\n"))
    articles = articlesCl
    
    articlesCl = []  
    for article in abstracts:
        articlesCl.append(article.replace("”", "").rstrip("\n"))
    abstracts = articlesCl
    
    return articles, abstracts
       

# return a list of of list of size "size"
def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append(list())
    return list_of_objects

def separateSentences(article, nlp = None):
    """if nlp == None:
        nlp = spacy.load('en_core_web_sm') """
    clean_desc = []
    sentences = sent_tokenize(article)
    for sentence in sentences:
        clean_desc.append(str(sentence))
    return clean_desc

# Text processing function

def textCleaningVocabList(articles, processing, stem = False, lem = False):
    stop_words = set(stopwords.words("english"))  
    lem = WordNetLemmatizer()
    #nlp = spacy.load('en_core_web_sm')
    vocabList = []
    articlesCleaned = []
    articlesCleanedSentences = []
    articlesSentences = []
    
    for article in articles:
        desc = article.lower() 
        split_text2 = desc.split()
        
        sentencesArticle = separateSentences(article)
        articlesSentences.append(sentencesArticle)
        sentencesArticleCleaned = []
        
        if processing == True: 
            desc = re.sub('[^a-zA-Z]', ' ', desc)
            desc = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",desc)
            desc = re.sub("(\\d|\\W)+"," ",desc)
            
            split_text = desc.split()
            #split_text = [word for word in split_text if not word in stop_words and len(word) > 2]  # attention changement
            
            if stem == True:
                split_text2 = [stemmer.stem(word) for word in split_text if not word in stop_words and len(word) > 2]
            elif lem == True:
                split_text2 = [lem.lemmatize(word) for word in split_text if not word in stop_words and len(word) > 2]
            else:
                split_text2 = split_text
            
            for sentence in sentencesArticle:
                desc = sentence.lower() 
                desc = re.sub('[^a-zA-Z]', ' ', desc)
                desc = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",desc)
                desc = re.sub("(\\d|\\W)+"," ",desc)
                
                split_text = desc.split()
                #split_text = [word for word in split_text if not word in stop_words and len(word) > 2]  # attention changement
                
                if stem == True:
                    split_text3 = [stemmer.stem(word) for word in split_text if not word in stop_words and len(word) > 2]
                elif lem == True:
                    split_text3 = [lem.lemmatize(word) for word in split_text if not word in stop_words and len(word) > 2]
                else:
                    split_text3 = split_text
                    
                split_text = " ".join(split_text3)
                sentencesArticleCleaned.append(split_text)
            articlesCleanedSentences.append(sentencesArticleCleaned)
        
        for word in split_text2:
            if word not in vocabList:
                vocabList.append(word)
                
        split_text = " ".join(split_text2)
        articlesCleaned.append(split_text)
        
        if processing == False:
            articlesCleanedSentences.append(sentencesArticle)
        
    return vocabList, articlesCleaned, articlesSentences, articlesCleanedSentences

# vectorizing function
# return un dictionnary word:vector
def w2vVectorizeToDict(vocabList, model, normalizeModels):
    dicVecWord = dict()
    cptNotInVocab = 0
    cptInVocab = 0
    from sklearn.preprocessing import normalize
    
    if isinstance(model, list):
        for word in vocabList:
            liste = []
            cptScope = 0
            for modell in model:
                
                listeVecs = []
                for wordd in vocabList:
                    try:
                        listeVecs.append(modell[wordd])
                    except:
                        pass
                  
                if normalizeModels == True:
                    listeVecs = normalize(listeVecs, norm='l2', axis=0, copy=False, return_norm=False)
                    
                vecMoy = np.mean(listeVecs, axis = 0)
                #print(listeVecs)
                #print(vecMoy)
                #print(len(vecMoy))
                
                try:
                    liste.append(modell[word])
                except:
                    liste.append(vecMoy)
                    cptScope += 1
                    if cptScope == 3:
                        cptNotInVocab +=1
              
            if normalizeModels == True:
                liste = normalize(liste, norm='l2', axis=0, copy=False, return_norm=False)
                    
            dicVecWord[word] = np.concatenate(liste)
            #print(len(dicVecWord[word]))
            cptInVocab +=1

                
    else:
        for word in vocabList:
            try:
                vector = model[word]
                dicVecWord[word] = vector
                cptInVocab +=1
            except:
                cptNotInVocab +=1
    #print(cptNotInVocab, "words not in models vocab // ", cptInVocab, "were in", "ration tot words / not in :", str((cptNotInVocab/(cptInVocab+cptNotInVocab))*100))
    return dicVecWord#, cptNotInVocab

# each vector is clusterised, we do 1 list per cluster with each vector from the cluster
#returns the list of these lists
def clusteringToVocabLists(dicVecWord, nbCluster, method):
    nbClusterPca = nbCluster
    nbCluster = int((nbCluster/100) * len(dicVecWord))
    if nbCluster <1:
        nbCluster = 1
    listesVocab = init_list_of_objects(nbCluster)
    #print("nb cluster", nbCluster)
    
    if(method == "kmeans"):
        clustering = KMeans(n_clusters= nbCluster, random_state=0, verbose = 0).fit(list(dicVecWord.values()))
    elif(method == "hierarchique"):
        clustering = AgglomerativeClustering(n_clusters = nbCluster).fit(list(dicVecWord.values())) # On lui donne les vecteurs du dico sous forme de liste
    elif(method == "OPTICS"):
        from sklearn.cluster import OPTICS
        clustering = OPTICS(n_jobs= -1, min_samples = 2, metric = "cosine", cluster_method = "dbscan").fit(list(dicVecWord.values()))
    else:
        print("erreur nom de methode de clustering")
    
    #dictionnaire mot:cluster_ref ou mot:représentant
    dicWordClus = dict()
    
    
    #print(clustering.labels_.tolist())
    if method == "OPTICS":
        maxindex, maxelement = max(enumerate(clustering.labels_.tolist()), key=itemgetter(1))
        listesVocab = init_list_of_objects(maxelement +1)
        
        for index in range(len(clustering.labels_.tolist())):
            if (clustering.labels_.tolist()[index] != -1):
                listesVocab[clustering.labels_.tolist()[index]].append(list(dicVecWord.keys())[index])
                dicWordClus[list(dicVecWord.keys())[index]] = listesVocab[clustering.labels_.tolist()[index]][0]
            else:
                listesVocab.append([list(dicVecWord.keys())[index]])
                dicWordClus[list(dicVecWord.keys())[index]] = str(index)
            
    
    elif method != "acp":
        for index in range(len(clustering.labels_.tolist())):
            if (clustering.labels_.tolist()[index] != -1):
                #print(index, len(list(dicVecWord.keys())))
                listesVocab[clustering.labels_.tolist()[index]].append(list(dicVecWord.keys())[index])
                dicWordClus[list(dicVecWord.keys())[index]] = listesVocab[clustering.labels_.tolist()[index]][0]
       
    return listesVocab, dicWordClus

# the reprensentant of a cluster is the TF
def clusteringToVocabListsV2(dicVecWord, nbCluster, method, article):
    nbCluster = int((nbCluster/100) * len(dicVecWord))
    if nbCluster <1:
        nbCluster = 1
    listesVocab = init_list_of_objects(nbCluster)
    #print("nb cluster", nbCluster)
    
    if(method == "kmeans"):
        clustering = KMeans(n_clusters= nbCluster, random_state=0, verbose = 0).fit(list(dicVecWord.values()))
    elif(method == "hierarchique"):
        clustering = AgglomerativeClustering(n_clusters = nbCluster).fit(list(dicVecWord.values())) # On lui donne les vecteurs du dico sous forme de liste
    elif(method == "OPTICS"):
        from sklearn.cluster import OPTICS
        clustering = OPTICS(n_jobs= -1, min_samples = 2, metric = "cosine", cluster_method = "dbscan").fit(list(dicVecWord.values()))
    else:
        print("erreur nom de methode de clustering")
    
    #dictionnaire mot:cluster_ref ou mot:représentant
    dicWordClus = dict()
    
    
    #print(clustering.labels_.tolist())
    if method == "OPTICS":
        maxindex, maxelement = max(enumerate(clustering.labels_.tolist()), key=itemgetter(1))
        listesVocab = init_list_of_objects(maxelement +1)
        
        for index in range(len(clustering.labels_.tolist())):
            if (clustering.labels_.tolist()[index] != -1):
                listesVocab[clustering.labels_.tolist()[index]].append(list(dicVecWord.keys())[index])
                dicWordClus[list(dicVecWord.keys())[index]] = listesVocab[clustering.labels_.tolist()[index]][0]
            else:
                listesVocab.append([list(dicVecWord.keys())[index]])
                dicWordClus[list(dicVecWord.keys())[index]] = str(index)
            
    
    elif method != "acp":
        for index in range(len(clustering.labels_.tolist())):
            if (clustering.labels_.tolist()[index] != -1):
                #print(index, len(list(dicVecWord.keys())))
                listesVocab[clustering.labels_.tolist()[index]].append(list(dicVecWord.keys())[index])
                
                # selection du mot dans le cluster ayant le tf le plus imp
                maxi = 0
                ind = 0
                for n, word in enumerate(listesVocab[clustering.labels_.tolist()[index]]):
                    tf = article.count(word)
                    if tf > maxi:
                        ind = n
                        maxi = tf
                dicWordClus[list(dicVecWord.keys())[index]] = listesVocab[clustering.labels_.tolist()[index]][ind]
       
    return listesVocab, dicWordClus

def transformArticleVocabularyV2(article, dicWordClus):
    articleTransformed =[]
    split_text = article.split()
    for word in split_text:
        try:
            articleTransformed.append(dicWordClus[word])
        except:
            articleTransformed.append(word)
            
    articleTransformed = " ".join(articleTransformed)
    
    return articleTransformed

# using clustering V2
def launcherSVD(wrt, articles, abstracts, dataset_name, processing, model, nbCluster, acpComponents, 
                kd = False, ordreClus = "avant",
                size = 50, output_file = "./output/", method = "hierarchique", 
                stem = False, lem = False, Scorer = True,
                normalizeAfterwards = False, normalizeVariance = False, pond = False,
                addAxes = 0, addWordPerAxe = 0, axeNumber = 0, normalizeAllAxes = True, reverse = False, normalizeModels = False):
    
    if dataset_name in ["xsum", "xsum2", "reddit_tifu", "reddit_tifu2", "duc"]:
        processin = True
        stem = False
        lem = True
        #print("lem parameters")
        
    else:
        processin = False
        stem = False
        lem = False
        #print("no proc parameters")
    
    
    open(output_file+"{}.txt".format(dataset_name), "w", encoding = "utf-8")
    summaries = []
    r1final = []
    r2final = []
    rLfinal = []
    scorer = rouge_scorer.RougeScorer(['rouge1', "rouge2", 'rougeL'], use_stemmer=True)

    
    for percent in nbCluster:
        start_time = time.time()
        r1score = []
        r2score = []
        rLscore = []
        positionBias = []
        
        acpWordsIndexHistoryFar = []
        
        varianceParAxe= init_list_of_objects(acpComponents)
        for index, article in enumerate(articles[:size]):
            ind = index

            vocabList, articleCleaned, articleSentences, articleCleanedSentences = textCleaningVocabList([article], processing, stem, lem)
            
            dicVecWord = w2vVectorizeToDict(vocabList, model, normalizeModels) #word:vec  pour chaque mot qui existe dans le vocab
            
            if ordreClus == "avant":
                listesVocabClus, dicWordClus = clusteringToVocabListsV2(dicVecWord, percent, method, article)
                #listesVocabClus, dicWordClus = clusteringToVocabLists(dicVecWord, percent, method)
            

            #SVD
            pca1 = PCA(n_components= acpComponents + addAxes, svd_solver = "full")
            acpWords = []
        

            pcaWordsTransformed = pca1.fit_transform(list(dicVecWord.values()))
            #print(pcaWordsTransformed.shape) # ex duc : (1042, 7) (718, 7)
            
            # on swap les vecteurs de taille 300 dans le dico par les vecteurs de dimensions k
            dicVecWord2 = dicVecWord.copy()
            dicVecWordSaved = dicVecWord.copy()

            for n, element in enumerate(dicVecWord.keys()):
                dicVecWord2[element] = pcaWordsTransformed[n]
            dicVecWord = dicVecWord2 #nos vecs ont ici acpComponents(k)-d
            
            if kd == False:
                dicVecWord = dicVecWordSaved # on reprend nos vec en 300d
            
            if ordreClus != "avant":
                listesVocabClus, dicWordClus = clusteringToVocabListsV2(dicVecWord, percent, method, article)
                #listesVocabClus, dicWordClus = clusteringToVocabLists(dicVecWord, percent, method)
    
            for indidi, variance in enumerate(pca1.explained_variance_ratio_):
                varianceParAxe[indidi].append(variance)     

            bestWords = []
            mincosListe = []
            
            if kd == False:
                dicVecWord = dicVecWordSaved # again 300D
                dicVecWord2 = 0
                dicVecWordSaved = 0
            
            for dimmension in pca1.components_:
                #print(dimmension.shape)
                minCos = 1
                vecWordBest=""
                dicWordCos = dict() #  order words using their cos distance to axes
                for index, vecWord in enumerate(list(dicVecWord.values())): # comparing each vector to its original dimension with 300d axes
                    
                    if kd == False:
                        distcos = distance.cosine(vecWord, dimmension)
                    else:
                        distcos = distance.cosine(vecWord, dimmension[:acpComponents])
                        
                    if distcos < minCos:                 
                        minCos = distcos
                        vecWordBest = list(dicVecWord.keys())[index]
                        dicWordCos[vecWordBest] = minCos
                        
                dicWordCos = sorted(dicWordCos.items(), key= lambda x: x[1], reverse=False) 
                
                for value in dicWordCos[: 1+ addWordPerAxe]:
                    bestWords.append(value[0])

                mincosListe.append(minCos)
            
            acpWords = bestWords
                        
            acpSynonyme = []
            for word in acpWords:
                acpSynonyme.append(dicWordClus[word])
            
            articlesTransformed = []
            articlesTransformed.append(transformArticleVocabularyV2(articleCleaned[0], dicWordClus))
            
            articlesCleanedSentencesTransformed = []    
            articleTransfo = []
            for sentence in articleCleanedSentences[0]:
                articleTransfo.append(transformArticleVocabularyV2(sentence, dicWordClus))
            articlesCleanedSentencesTransformed.append(articleTransfo)
  
            
            #Extraction of sentences, taking in count clusters synonyms

            acpSentences = []
            acpHistoryFar = []
            acpSentencesPonderatedVariance = dict() # store the sentences with their counting ponderated variance score
            
            indexes = []
            
            listeDesPalliers = []
            for pallier in range(0,((addWordPerAxe + 1)*acpComponents),addWordPerAxe+1):
                listeDesPalliers.append(pallier)

            for indexaz, synonyme in enumerate(acpSynonyme): # for an article, each representing a synonym
                maxi = 0
                indexo = 0
                for indexi, sentence in enumerate(articlesCleanedSentencesTransformed[0]):
                    count = sentence.count(synonyme)
                    
                    if normalizeVariance == True:
                        indi = indexaz
                        while indi not in listeDesPalliers:
                            indi -= 1
                        for indd in range(len(listeDesPalliers)):
                            if indi == listeDesPalliers[indd]:
                                lindice = indd
                        if pond == False:
                            count = count #/ pca1.explained_variance_ratio_[lindice] # au lieu de indi
                        else:
                            count = count / pca1.explained_variance_ratio_[lindice] # au lieu de indi
                    
                    if addWordPerAxe != 0:
                        if indexaz in listeDesPalliers:
                            for indexayy in range(1, addWordPerAxe+1):
                                if indexaz + indexayy < len(acpSynonyme):
                                    
                                    if normalizeVariance == False:
                                        count += sentence.count(acpSynonyme[indexaz + indexayy]) # add same axe word counting
                                    else:
                                        if pond == True:
                                            count += sentence.count(acpSynonyme[indexaz + indexayy]) / pca1.explained_variance_ratio_[lindice]
                                        else:
                                            count += sentence.count(acpSynonyme[indexaz + indexayy]) #/ pca1.explained_variance_ratio_[lindice]
                    
                        
                    if normalizeVariance == True:
                        if normalizeAllAxes == True:
                            try:
                                acpSentencesPonderatedVariance[sentence] += count # on considère tt les axes 
                                #acpSentencesPonderatedVariance[articleSentences[0][indexi]] += count
                            except:
                                acpSentencesPonderatedVariance[sentence] = count
                                #acpSentencesPonderatedVariance[articleSentences[0][indexi]] = count
                        else:
                            if indexaz == axeNumber: 
                                acpSentencesPonderatedVariance[sentence] = count
                                #acpSentencesPonderatedVariance[articleSentences[0][indexi]] = count

                        
                    if count > maxi:   # take the sentence which has the count>>
                        if indexi not in indexes: # condition to have different sentences
                            indexo = indexi
                            maxi = count
                            
                # NOT USED
                if normalizeAfterwards == True:
                    lenmini = 0
                    for indexi, sentence in enumerate(articlesCleanedSentencesTransformed[0]):
                        count = sentence.count(synonyme)
                        if count == maxi:   # take the sentence which has the most apparition of synonym
                            if len(sentence) < lenmini: #
                                lenmini = len(sentence)
                                indexo = indexi
                
                if indexaz in listeDesPalliers: # add the best sentences which count the best sentence per axis
                    indexes.append(indexo)
                    acpSentences.append(articleSentences[0][indexo])
                    #acpSentences.append(articleCleanedSentences[0][indexo])
                
               
            
            if normalizeVariance == True:
                acpSentences = []
                acpSentencesPonderatedVariance = sorted(acpSentencesPonderatedVariance.items(), key= lambda x: x[1], reverse= reverse)    

                for value in acpSentencesPonderatedVariance[:acpComponents]:
                    acpSentences.append(value[0])            

            acpWordsIndexHistoryFar.append(acpHistoryFar)
            summaries.append(acpSentences) # contient toutes les (meilleures) phrases par synonyme, donc si + de acpcomponents phrases, alors pb car + de phrase que ce qui devrait y avoir
            
            #print("--- %s seconds ---" % (time.time() - start_time))
            
            if wrt == True:
                
                with open(output_file+"{}.txt".format(dataset_name), "a", encoding = "utf-8") as fh:
                    fh.write(" ".join(acpSentences)+"\n")
            
            if Scorer == True:
                summ = acpSentences
                r1score.append(scorer.score(" ".join(summ),
                          abstracts[ind])["rouge1"][2])
                r2score.append(scorer.score(" ".join(summ),
                          abstracts[ind])["rouge2"][2])
                rLscore.append(scorer.score(" ".join(summ),
                          abstracts[ind])["rougeL"][2])
                
                
        print("Summary written in output folder !")
                
            
        if Scorer == True:
            r1final.append(round(statistics.mean(r1score),5))
            r2final.append(round(statistics.mean(r2score),5))
            rLfinal.append(round(statistics.mean(rLscore),5))
            
            


    
    maxi = 0
    indexMax =0

    # opti R1+R2+RL
    """
    for index in range(len(r1final)):
        #print(len(r1final))
        print(r1final[index], r2final[index], rLfinal[index])
        if (r1final[index] + r2final[index] + rLfinal[index]) > maxi:
            maxi = r1final[index] + r2final[index] + rLfinal[index]
            indexMax = index"""
            
    # opti R2
    for index in range(len(r1final)):
        #print(len(r1final))
        #print(r1final[index], r2final[index], rLfinal[index])
        if (r2final[index]) > maxi:
            maxi = r2final[index]
            indexMax = index

        
    maxrouge = [r1final[indexMax],r2final[indexMax],rLfinal[indexMax]]
    varianceParAxeMoy = []
    for listeVariance in varianceParAxe:
        varianceParAxeMoy.append(statistics.mean(listeVariance))
    

    return maxrouge, nbCluster[indexMax], varianceParAxeMoy 

def countWordsPerConc(dataset_name, size):
    articles, abstracts = load_dataset(dataset_name)
    if dataset_name in ["xsum", "xsum2", "reddit_tifu", "reddit_tifu2", "duc"]:
        processin = True
        stem = False
        lem = True
        #print("lem parameters")
        
    else:
        processin = False
        stem = False
        lem = False
        #print("no proc parameters")
    
    
    if dataset_name == "cnn" or dataset_name == "xsum" or  dataset_name == "reddit_tifu":
        acpComponents = 3
    if dataset_name == "pubmed" or dataset_name == "MN":
        acpComponents = 9
    if dataset_name == "reddit":
        acpComponents = 4
    if dataset_name == "duc":
        acpComponents = 7
        
    article = " ".join(articles[:size])
    index = 0
    vocabList, articleCleaned, articleSentences, articleCleanedSentences = textCleaningVocabList([article], processin, stem, lem)
    #print(articleCleaned[:40])
    print("nombre de mot :", len(word_tokenize(" ".join(articleCleaned))), "taille vocab :", len(vocabList), 'nombre de phrases :', len(articleCleanedSentences[0]))
    print()

def launcherSVDCONC(dataset_name, size):
    
    print("loading word embedding model")
    file = open("./Word Embedding Model/glove_word_embedding.pkl", "rb")
    model = pickle.load(file)
    print("done")
    articles, abstracts = load_dataset(dataset_name)



    if dataset_name in ["xsum", "xsum2", "reddit_tifu", "reddit_tifu2", "duc"]:
        processin = True
        stem = False
        lem = True
        #print("lem parameters")
        
    else:
        processin = False
        stem = False
        lem = False
        #print("no proc parameters")
    
    
    if dataset_name == "cnn" or dataset_name == "xsum" or  dataset_name == "reddit_tifu":
        acpComponents = 3
    if dataset_name == "pubmed" or dataset_name == "MN":
        acpComponents = 9
    if dataset_name == "reddit":
        acpComponents = 4
    if dataset_name == "duc":
        acpComponents = 7
    
    
    acpWordsIndexHistoryFar = []
    
    varianceParAxe= init_list_of_objects(acpComponents)
    
    article = " ".join(articles[:size])
    
    index = 0

    vocabList, articleCleaned, articleSentences, articleCleanedSentences = textCleaningVocabList([article], processin, stem, lem)
    
    start_time = time.time()
    dicVecWord = w2vVectorizeToDict(vocabList, model, False) #word:vec  pour chaque mot qui existe dans le vocab
    

    listesVocabClus, dicWordClus = clusteringToVocabListsV2(dicVecWord, 100, "hierarchique", article)
    #listesVocabClus, dicWordClus = clusteringToVocabLists(dicVecWord, percent, method)

    
    #SVD
    pca1 = PCA(n_components= acpComponents, svd_solver = "full")
    acpWords = []


    pcaWordsTransformed = pca1.fit_transform(list(dicVecWord.values()))
    #print(pcaWordsTransformed.shape) # ex duc : (1042, 7) (718, 7)
    

    for indidi, variance in enumerate(pca1.explained_variance_ratio_):
        varianceParAxe[indidi].append(variance)     

    bestWords = []
    mincosListe = []
    
    
    for dimmension in pca1.components_:
        #print(dimmension.shape)
        minCos = 1
        vecWordBest=""
        dicWordCos = dict() #  order words using their cos distance to axes
        for index, vecWord in enumerate(list(dicVecWord.values())): # comparing each vector to its original dimension with 300d axes
            

            distcos = distance.cosine(vecWord, dimmension)

            if distcos < minCos:                 
                minCos = distcos
                vecWordBest = list(dicVecWord.keys())[index]
                dicWordCos[vecWordBest] = minCos
                
        dicWordCos = sorted(dicWordCos.items(), key= lambda x: x[1], reverse=False) 
        
        for value in dicWordCos[: 1]:
            bestWords.append(value[0])

        mincosListe.append(minCos)
    
    acpWords = bestWords
                
    acpSynonyme = []
    for word in acpWords:
        acpSynonyme.append(dicWordClus[word])
    
    articlesTransformed = []
    articlesTransformed.append(transformArticleVocabularyV2(articleCleaned[0], dicWordClus))
    
    articlesCleanedSentencesTransformed = []    
    articleTransfo = []
    for sentence in articleCleanedSentences[0]:
        articleTransfo.append(transformArticleVocabularyV2(sentence, dicWordClus))
    articlesCleanedSentencesTransformed.append(articleTransfo)
  
    
    #Extraction of sentences, taking in count clusters synonyms

    acpSentences = []
    acpSentencesPonderatedVariance = dict() # store the sentences with their counting ponderated variance score
    
    indexes = []
    
    listeDesPalliers = []
    for pallier in range(0,((1)*acpComponents),1):
        listeDesPalliers.append(pallier)

    for indexaz, synonyme in enumerate(acpSynonyme): # for an article, each representing a synonym
        maxi = 0
        indexo = 0
        for indexi, sentence in enumerate(articlesCleanedSentencesTransformed[0]):
            count = sentence.count(synonyme)
            
                
            if count > maxi:   # take the sentence which has the count>>
                if indexi not in indexes: # condition to have different sentences
                    indexo = indexi
                    maxi = count
                    

        
        if indexaz in listeDesPalliers: # add the best sentences which count the best sentence per axis
            indexes.append(indexo)
            acpSentences.append(articleSentences[0][indexo])
            #acpSentences.append(articleCleanedSentences[0][indexo])

    print("--- %s seconds ---" % (time.time() - start_time))
    

def clus_heuristic(model, articles, abstracts, dataset_name, size, numberSentencesOP = 0, step = -10, 
             kd = False, ordreClus = "avant", paramfile = "data"): #parametersAvant300D parametersApresKD
    
    if os.path.isfile('./Datasets/{}.pkl'.format(paramfile)) == True:
        file = open('./Datasets/{}.pkl'.format(paramfile), "rb") # faudra décommenter en dessous et mettre rb
        paramss = pickle.load(file)
        
        if isinstance(paramss[0], list):
            for para in paramss:
                if para[0] == dataset_name:
                    params = para
                    #print("heuristic found")
                    return params
        else:
            if paramss[0] == dataset_name:
                params = paramss
                return params

    if numberSentencesOP == 0:
        if dataset_name in ["cnn", "reddit", "reddit_tifu", "xsum"]:
            numberSentencesOP = 3
        if dataset_name == "duc":
            numberSentencesOP = 7
        if dataset_name in ["MN", "pubmed"]:
            numberSentencesOP = 9

    wrt = True
    if dataset_name in ["xsum", "xsum2", "reddit_tifu", "reddit_tifu2", "duc"]:
        processin = True
        stemVar = False
        lemVar = True
        
    else:
        processin = False
        stemVar = False
        lemVar = False
        
    size = 150
        
    if dataset_name == "reddit":
        size = 48
    if dataset_name == "duc":
        size = 50
        
    #print("heuristic")
        
    acpComponents = numberSentencesOP
    
    listeRougeScores = []
    percentCorresponding = []
    params = []
    
    # tries the multi document heuristic
    bestTemp = []
    temp = [0,0,0]
    addW=0
    print("multi, kd:", kd, "clustering:", ordreClus)
    for addWord in range(6):
        if addWord == 0:
            bestTemp, bestPercent, x = launcherSVD(wrt, articles, abstracts, dataset_name, processin, model , [i for i in np.arange(100, 0, step)], acpComponents = acpComponents, 
                    size = size, method = "hierarchique", kd = kd, ordreClus = ordreClus,
                    stem = stemVar, lem = lemVar, Scorer = True,
                    normalizeAfterwards = False, normalizeVariance = False, 
                    pond=False, addAxes = 0, addWordPerAxe = addWord, axeNumber= 0, normalizeAllAxes= False, reverse = False, normalizeModels= False)
            addW = addWord
            print(bestTemp, "params :", bestPercent, addWord, "md")
        else:
            temp, percent, x = launcherSVD(wrt, articles, abstracts, dataset_name,  processin, model , [i for i in np.arange(100, 0, step)], acpComponents = acpComponents, 
                    size = size, method = "hierarchique",  kd = kd, ordreClus = ordreClus,
                    stem = stemVar, lem = lemVar, Scorer = True, 
                    normalizeAfterwards = False, normalizeVariance = False, 
                    pond=False, addAxes = 0, addWordPerAxe = addWord, axeNumber= 0, normalizeAllAxes= False, reverse = False, normalizeModels= False)
            addW = addWord
            print(temp, "params :", percent, addWord, "md")
            
        somme = 0
        bestSomme = 0
        for val in bestTemp[1:2]:
            bestSomme += val
        for val in temp[1:2]:
            somme += val
        if somme > bestSomme:
            bestTemp = temp
            bestPercent = [percent, addW]
            
    listeRougeScores.append(bestTemp)
    percentCorresponding.append(bestPercent)
    #print(bestTemp)
    #print(bestPercent)
    

            
    
    # tries the single document heuristic
    
    bestTemp = []
    temp = [0,0,0]
    axx=0
    print("single, kd:", kd, "clustering:", ordreClus)
    for axe in range(3):
        if axe == 0:
            bestTemp, bestPercent, x = launcherSVD(wrt, articles, abstracts, dataset_name, processin, model , [i for i in np.arange(100, 100+step, step)], acpComponents = acpComponents, 
                        size = size, method = "hierarchique", kd = kd, ordreClus = ordreClus,
                        stem = stemVar, lem = lemVar, Scorer = True,
                        normalizeAfterwards = False, normalizeVariance = True, 
                        pond=False, addAxes = 0, addWordPerAxe = 0, axeNumber= axe, normalizeAllAxes= False, reverse = False, normalizeModels= False)
            axx = axe
            print(bestTemp, "params :", bestPercent, axe, "sd")
        else:
            temp, percent, x = launcherSVD(wrt, articles, abstracts, dataset_name, processin, model , [i for i in np.arange(100, 100+step, step)], acpComponents = acpComponents, 
                        size = size, method = "hierarchique", kd = kd, ordreClus = ordreClus,
                        stem = stemVar, lem = lemVar, Scorer = True,
                        normalizeAfterwards = False, normalizeVariance = True, 
                        pond=False, addAxes = 0, addWordPerAxe = 0, axeNumber= axe, normalizeAllAxes= False, reverse = False, normalizeModels= False)
            axx = axe
            print(temp, "params :", percent, axe, "sd")
            
        somme = 0
        bestSomme = 0
        for val in bestTemp[1:2]:
            bestSomme += val
            
        for val in temp[1:2]:
            somme += val
            
        if somme > bestSomme:
            bestTemp = temp
            bestPercent = [percent, axx]
            #print(axx, "wtf")
        elif axx == 0:
            bestPercent = [bestPercent, axx]
    
            
    listeRougeScores.append(bestTemp)
    percentCorresponding.append(bestPercent)
    #print(bestTemp)
    #print(bestPercent)
    
    
    # select the best heuristic between single document and multidocument heuristic
    tempSum = []
    for heuristic in listeRougeScores:
        val = 0
        for rouge in heuristic:
            val += rouge
        tempSum.append(val)
    if tempSum[0]>tempSum[1]:
        per = percentCorresponding[0]
        bestPercent = []
        bestPercent.append(dataset_name)
        for val in per:
            bestPercent.append(val)
        bestPercent.append("md")
    else:
        per = percentCorresponding[1]
        bestPercent = []
        bestPercent.append(dataset_name)
        for val in per:
            bestPercent.append(val)
        bestPercent.append("sd")
    params = bestPercent
    
    # à décommenter ici
    if os.path.isfile('./Datasets/{}.pkl'.format(paramfile)) == False:
        file = open('./Datasets/{}.pkl'.format(paramfile),'wb')
        pickle.dump([params], file)
    else:
        file = open('./Datasets/{}.pkl'.format(paramfile), 'rb')
        para = pickle.load(file)
        para.append(params)
        
        file = open('./Datasets/{}.pkl'.format(paramfile), 'wb')
        pickle.dump(para, file)
    print("\nbest params : ", params)
    return params
    
def launchFull(model, articles, abstracts, dataset_name, params, size = 0, step = -5, dataset = "cnn", 
               acpComponents = 0, kd = False, ordreClus = "avant", output_file = "./output/"):
    
    wrt = True
    # Pre-processing
    if dataset_name in ["xsum", "xsum2", "reddit_tifu", "reddit_tifu2", "duc"]:
        processin = True
        stemVar = False
        lemVar = True
        
    else:
        processin = False
        stemVar = False
        lemVar = False

        
        
        
    if acpComponents == 0:
        if dataset in ["cnn", "reddit", "reddit_tifu", "xsum"]:
            acpComponents = 3
        if dataset == "duc":
            acpComponents = 7
        if dataset in ["MN", "pubmed"]:
            acpComponents = 9

    size = size
    
    
    # multi document heuristic
    if params[3] == "md":
        bestTemp, bestPercent, x = launcherSVD(wrt, articles, abstracts, dataset_name, processin, model , [int(params[1])],
            acpComponents = acpComponents, kd = kd, ordreClus = ordreClus,
            size = size, output_file = output_file, method = "hierarchique",
            stem = stemVar, lem = lemVar, Scorer = True, 
            normalizeAfterwards = False, normalizeVariance = False, 
            pond=False, addAxes = 0, addWordPerAxe = int(params[2]), axeNumber= 0, normalizeAllAxes= False, reverse = False, normalizeModels= False)
    
    
    
    # single document heuristic
    else:
        bestTemp, bestPercent, x = launcherSVD(wrt, articles, abstracts, dataset_name, processin, model , [int(params[1])],
            acpComponents = acpComponents, kd= kd, ordreClus = ordreClus,
            size = size, output_file = output_file, method = "hierarchique", 
            stem = stemVar, lem = lemVar, Scorer = True,
            normalizeAfterwards = False, normalizeVariance = True, 
            pond=False, addAxes = 0, addWordPerAxe = 0, axeNumber= int(params[2]), normalizeAllAxes= False, reverse = False, normalizeModels= False)

def textRank(dataset_name, size):
    from gensim.summarization.summarizer import summarize

    print("loading",dataset_name)
    articles, abstracts = load_dataset(dataset_name)
    print("done")
    if dataset_name == "cnn" or dataset_name == "xsum" or  dataset_name == "reddit_tifu":
        nbsents = 3
    if dataset_name == "pubmed" or dataset_name == "MN":
        nbsents = 9
    if dataset_name == "reddit":
        nbsents = 4
    if dataset_name == "duc":
        nbsents = 7
    
    start_time = time.time()
        
    import warnings
    warnings.filterwarnings("ignore")
    articlesRouge=[]
    
    textrank = []
    cpt = 0
    if size > len(articles):
        size = len(articles)
        
    article = " ".join(articles[:size])
    ratio =  nbsents / len(sent_tokenize(article))
    #print(ratio)
    
    textrank.append(separateSentences(summarize(article, ratio = ratio))) 
    #print(len(separateSentences(summarize(articles[index], ratio = ratio))))
    cpt +=1
    articlesRouge.append(article)
    #abstractsRouge.append(abstracts[index])

    print("--- %s seconds ---" % (time.time() - start_time))
        
        
def summVD(word_embedding, dataset_name = "", heuristic_rate = 0, testSize = 100, numberSentencesOP = 0,
           kd = False, ordreClus= "avant"):
    
    print("loading word embedding model")
    file = open(word_embedding, "rb")
    word_embedding = pickle.load(file)
    print("done")
    
    # Loading dataset
    if dataset_name in ["cnn", "xsum","reddit_tifu", "pubmed", "MN", "duc", "reddit"]:
        articles, abstracts = load_dataset(dataset_name)     
    # For custom dataset
    else:
        file = open(str("./Datasets/"+dataset_name+"_documents.pkl"),'rb')
        articles = pickle.load(file)
        
        file = open(str("./Datasets/"+dataset_name+"_gold.pkl"),'rb')
        abstracts = pickle.load(file)
    
    
    if len(articles) == len(abstracts):
        size = int(heuristic_rate * len(articles))
        params = clus_heuristic(word_embedding, articles, abstracts, dataset_name, size, numberSentencesOP, step = -10, 
                          kd = kd, ordreClus = ordreClus)
        if testSize == 0:
            testSize = len(articles)
        launchFull(word_embedding, articles, abstracts, dataset_name, params, size = testSize , step = -10, 
                   dataset = dataset_name, acpComponents = numberSentencesOP, kd = kd, ordreClus = ordreClus)
        
        
        
    else:
        print("Error, documents and gold length don't match")
        
        
def scoring(dataset_name):
    file = "./output/{}.txt".format(dataset_name)
    abstracts = []
    with open(file, "r", encoding = "utf-8") as fh:
        for line in fh:
            abstracts.append(line)
    scorer = rouge_scorer.RougeScorer(['rouge1', "rouge2", 'rougeL'], use_stemmer=True)
    articles, gold = load_dataset(dataset_name)
    
    r1score = []
    r2score = []
    rLscore = []
    for index in range(len(abstracts)):
        r1score.append(scorer.score(gold[index],
                  abstracts[index])["rouge1"][2])
        r2score.append(scorer.score(gold[index],
                  abstracts[index])["rouge2"][2])
        rLscore.append(scorer.score(gold[index],
                  abstracts[index])["rougeL"][2])
        
    print("R1:",statistics.mean(r1score),"R2:",statistics.mean(r2score),"RL:",statistics.mean(rLscore) )
    
    
def read_arguments():
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_embedding_path", type=str, default="./Word Embedding Model/glove_word_embedding.pkl")
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--heuristic_rate", type=float, default=0.005)
    parser.add_argument("--size_output", type=int, default=0) # number of document to summaries
    parser.add_argument("--nb_sentences", type=int, default=0) # number of sentence per summary
    parser.add_argument("--scoring", type=bool, default = False) # to score a file named "dataset_name"
    parser.add_argument("--kdparam", type=str, default = "False") # number of dimmension for vectors in clustering methods
    parser.add_argument("--ordreClus", type=str, default="avant")
    parser.add_argument("--longDocument", type=str, default = None) # to run the concatenated document 

    return parser.parse_args()
    
    
    

def main():
    # read arguments
    args=read_arguments() 
    word_embedding_path = args.word_embedding_path
    dataset_name = args.dataset_name
    heuristic_rate = args.heuristic_rate
    size_output = args.size_output
    nb_sentences = args.nb_sentences
    scoringP = args.scoring
    longDocument = args.longDocument 
    
    if args.kdparam == "False":
        kdpara = False
    else: 
        kdpara = True

    ordreClus = args.ordreClus
    print()
    
    if longDocument == "textrank":
        """for size_output in range(200, 501, 50):
            print("taille concaténation :", size_output)
            textRank(dataset_name, size_output)
        for size_output in range(600, 1001, 100):
            print("taille concaténation :", size_output)
            textRank(dataset_name, size_output)"""
            
        textRank(dataset_name, size_output)
        
    elif longDocument == "summvd":
        """for size_output in range(200, 501, 50):
            print("taille concaténation :", size_output)
            launcherSVDCONC(dataset_name, size_output)
            #countWordsPerConc(dataset_name, size_output)
        for size_output in range(600, 1001, 100):
            print("taille concaténation :", size_output)
            launcherSVDCONC(dataset_name, size_output)
            #countWordsPerConc(dataset_name, size_output)"""
            
        #countWordsPerConc(dataset_name, size_output)
        launcherSVDCONC(dataset_name, size_output)
            
    else:
        if scoringP == True:
            scoring(dataset_name)
        else:    
            summVD(word_embedding_path, dataset_name = dataset_name, heuristic_rate = heuristic_rate, testSize = size_output, numberSentencesOP = nb_sentences, kd = kdpara, ordreClus = ordreClus)
            scoring(dataset_name)

#%%
if __name__ == "__main__":
    main()   


