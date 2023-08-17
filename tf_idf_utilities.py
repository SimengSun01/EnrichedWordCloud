#TF_IDF Utility
import numpy as np
import pandas as pd
import os
import sys
import gensim 
import sklearn
import scipy
from gensim.corpora import Dictionary
from gensim.models import LdaModel, TfidfModel
from gensim.matutils import jensen_shannon
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer


#function to add labels to the original dataframe 
#the data entered should at least have the following columns:
#MID, Lemmatized_Sentence, GeneID,Symbol
def generate_fg_bg(df, tf): #tf as the target transcription factor, a string in capital 
#df as the input dataframe 
#considered as the positive label/foreground, class 1
#return 2 dataframes: background dataframe, class 0; foreground dataframe, class 1
	df['Class'] = df['Symbol'].apply(lambda x: 1 if tf in x else 0)
	fore_df = df.loc[(df['Class'] == 1)]
	back_df = df.loc[(df['Class'] == 0)]
	#drop rows with null values 
	fore_df.dropna(subset=['Lemmatized_Sentence'])
	back_df.dropna(subset=['Lemmatized_Sentence'])
	return fore_df, back_df

def doc_to_topics(fg, bg, num_topics = None): #take the foreground and background dfs as the input 
	combined_docs = pd.concat([fg['Lemmatized_Sentence'], bg['Lemmatized_Sentence']], ignore_index=True) #concatenate the 2 input dataframes
	combined_docs = combined_docs.tolist()
	combined_docs = [d.split() for d in combined_docs]#create a sublist for each artile
	#create a dictionary based on combined_docs 
	dictionary = Dictionary(combined_docs)
	#construct word<->PMID mappings 
	fg['Lemmatized_Sentence_temp'] = fg['Lemmatized_Sentence'].apply(lambda x: x.split())
	bg['Lemmatized_Sentence_temp'] = bg['Lemmatized_Sentence'].apply(lambda x: x.split())
	corpus_fg = [dictionary.doc2bow(doc) for doc in fg['Lemmatized_Sentence_temp']]
	corpus_bg = [dictionary.doc2bow(doc) for doc in bg['Lemmatized_Sentence_temp']]
	#TF-IDF model to weight terms 
	tfidf = TfidfModel(dictionary=dictionary)
	#Objects of this class realize the transformation between word-document
	#co-occurrence matrix (int) into a locally/globally weighted TF-IDF matrix
	#(positive floats).
	corpus_tfidf_fg = tfidf[corpus_fg]
	corpus_tfidf_bg = tfidf[corpus_bg]
	#defualt number of topics for the LDA model = 10
	num_topics = 10
	lda = LdaModel(corpus_tfidf_fg, id2word=dictionary, num_topics=num_topics)
	#get the topic distributions for the foreground and background documents
	topics_fg = lda[corpus_tfidf_fg]
	topics_bg = lda[corpus_tfidf_bg]#generate topic probability distribution for a document
	return topics_fg,topics_bg

def document_selection_hard_cutoff(topics_fg,topics_bg,back_df,similar=True,cutoff = 0.025): 
	#fg as the foreground corpus topic probability distribution, 
	#bg as the background corpus topic probability distribution, mode is a string input, 
	#back_df to be the background data
	#indicating selecting the most similar/distinct documents, default is True
	#this function returns the most similar/distinct articles based on the cutoff, defualt cutoff = 0.025, cutoff in [0,1]

	avg_topic_dist_fg = np.mean([doc for doc in topics_fg], axis=0) #average topic distribution of foreground documents
	#apply Jensen-Shannon Divergence to caluclate the distance between probability distribution 
	js_divs = [jensen_shannon(doc, avg_topic_dist_fg) for doc in topics_bg]
	sorted_js_divs = sorted(js_divs)
	if similar:# select the most similar 2.5% articles 
		threshold_index = int(len(sorted_js_divs) * cutoff)
		threshold= sorted_js_divs[threshold_index]
		docs_indices = [index for index, js_div in enumerate(js_divs) if js_div <= threshold]
		filtered_df = back_df.iloc[docs_indices]
	else:# select the most distinct 2.5% articles 
		dis_js_divs = sorted(js_divs, reverse=True)
		threshold_index = int(len(dis_js_divs) * cutoff)
		threshold= dis_js_divs[threshold_index]
		docs_indices = [index for index, js_div in enumerate(js_divs) if js_div >= threshold]
		filtered_df = back_df.iloc[docs_indices]
	return filtered_df

def rank_biserial(n1,n2,U): #function to calcualte the effect size for Mann-Whitney U Test with unequal
	#sample sizes 
	return 1 - (2 * U) / (n1 * n2)

def tf_idf_enrichment(fg, filtered_df,tf): #tf as the transcription factor 
	fg_corpus = fg['Lemmatized_Sentence'].tolist()
	bg_corpus =filtered_df['Lemmatized_Sentence'].tolist()
	corpus = fg_corpus+bg_corpus
	#initialize vectorizer 
	vectorizer = TfidfVectorizer()
	#transform the sentences to vectors
	X = vectorizer.fit_transform(corpus)#Compressed Sparse Row format
	#get a matrix of tf-idf scores
	X_fg, X_bg = X[:len(fg_corpus), :], X[len(fg_corpus):, :]
	words = vectorizer.get_feature_names_out()
	word_enrichment =[]
	#apply Mann-Whitney U Test to select "enriched" terms 
	for i, word in enumerate(words):
		
		scores_fg = X_fg[:, i].toarray().ravel()
		scores_bg = X_bg[:, i].toarray().ravel()
		U, p_value = mannwhitneyu(scores_fg, scores_bg, alternative='greater')
		r = rank_biserial(len(scores_fg), len(scores_bg), U)
		word_enrichment.append((word, p_value, r))	
		#tf-idf scores for the current word in the foreground and background sets
        #essentially, asking: for word i, based on its
    	#tf-idf scores from doc 0 to doc x(fore), and doc 0 to doc y(back), are the 2 score distributions from the same distribution?
    	#len(scores_fg) == number of foreground documents 
    	#len(scores_bg) == number of background documents
    	#U, p_value = mannwhitneyu(scores_fg, scores_bg, alternative='greater')#since we aim to find the most enriched words in the foreground, set alternative == greater
    		#r = rank_biserial(len(scores_fg), len(scores_bg), U)#caluclate effect size
    	#word_enrichment.append((word, p_value, r))

	word_enrichment_df= pd.DataFrame(word_enrichment, columns=['word', 'p_value', 'Rank_biserial'])
	word_enrichment_df = word_enrichment_df[word_enrichment_df['Rank_biserial'] < 0]#only keep the one with negative values 
	word_enrichment_df = word_enrichment_df.sort_values(['p_value','Rank_biserial'], ascending=[True, True])
	word_enrichment_df = word_enrichment_df[word_enrichment_df['p_value'] < 0.05] #only keep the statistically important terms
	#measure the magnitutde of effect size;raw value in range [-1,1]; the closer its abs is to 1, the larger it is 
	word_enrichment_df.drop(word_enrichment_df[word_enrichment_df['word']==tf.lower()].index, inplace = True)#drop the transcription factor itself
	return word_enrichment_df








