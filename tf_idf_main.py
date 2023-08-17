import os
import sys
import numpy as np
import pandas as pd
import sklearn 
import scipy 
import re
import json
import gzip
import info2json_nlp #generate the json file if it does not already exist 
import data_filter
import tf_idf_utilities
import word_cloud
from sklearn.utils import shuffle

#this script finishes the following tasks:
#from xml to json 
#from json (pubmeddb_nlp) to parquet/csv, based on huamn transcription factors
#from the csv, calcualte tf-idf
#generate word cloud

#from xml to json 
#initial setup 
os.chdir(sys.path[0])
inputdir = os.getcwd()
#check if the pubmed json file already exists
#if it's not, we run the info2json_nlp script to convert the data into 
#json, in a data base style for further information extraction 
jsondir = os.path.join(inputdir, "utils_tfidf","data","pubmeddb_nlp.json.gz")
if not os.path.exists(jsondir): #note: we put info2json_nlp in the same direcotry as the main script
	#other wise use this: sys.path.append(path/to/info2json_nlp) before importing the script 
	info2json_nlp.main()
#if exists, we tailor the json to a csv consisting only csvs 
tfdir = os.path.join(inputdir, "utils_json","data","tfactors.txt")
with open(tfdir, 'r') as f:
    tflist = f.read().splitlines() #read the transcrition factors as a list 


#Data PreProcessing 
#now that we have info2json_nlp.json.gz in the direcotry
#further process the data using functions in data_filter 
#convert the json.gz file to a csv with the following main columns: PMID, Title, Text, and Words
print(tflist[0:10])
output_path = os.path.join(inputdir, "utils_tfidf","data")
if not os.path.exists(output_path):
	os.makedirs(output_path)
json2csv_output_path = os.path.join(inputdir, "utils_tfidf","data","complete_output.csv")
if not os.path.exists(json2csv_output_path):
	data_filter.json2csv(jsondir, json2csv_output_path)#convert json.gz to a csv for further processing 
#mapping GeneID & PMID 
gene_path = os.path.join(inputdir,'utils_json','data', 'gene2pubmedid.json')
gene_output_path = os.path.join(inputdir, "utils_tfidf","data","output_geneID.csv")
if not os.path.exists(gene_output_path):
	data_filter.pmid_gene(json2csv_output_path,gene_path,gene_output_path)
human_output_path = os.path.join(inputdir, "utils_tfidf","data","all_human_abstracts.csv")
if not os.path.exists(human_output_path):
	data_filter.human_abstract(gene_output_path,tflist,human_output_path)
###########filter out outlier articles
human_df = pd.read_csv(human_output_path)
temp_df =data_filter.remove_outliers(human_df)
temp_df.to_csv(human_output_path,index=False)
#lemmatize the sentences 
output_path_lemmatize = os.path.join(inputdir, "utils_tfidf","data","lemmatized_abstracts.csv")
if not os.path.exists(output_path_lemmatize): #if the lemmatized csv hasn't been created
	output_path_lemmatize =data_filter.lemmatize_csv(human_output_path,output_path)
	lemma_df = pd.read_csv(output_path_lemmatize)

else: 
	lemma_df = pd.read_csv(output_path_lemmatize)



temp_df_dir = os.path.join(inputdir, "utils_tfidf","temp_data")
if not os.path.exists(temp_df_dir):
	os.makedirs(temp_df_dir)
tf_res_dir = os.path.join(inputdir,"tf_idf_result")
if not os.path.exists(tf_res_dir):
	os.makedirs(tf_res_dir) #create a directory to store tf_idf scores if it doesn't already exist 
word_cloud_res_dir = os.path.join(inputdir,"word_cloud_results")
if not os.path.exists(word_cloud_res_dir):
	os.makedirs(word_cloud_res_dir)
#Compute TF-IDF & Word Cloud for every Transcription Factor 
##############use functions from tf_idf_utilities################### 

for tf in tflist:
	tf_filtered_path = os.path.join(inputdir,"utils_tfidf",'temp_data',f'{tf}_temp.csv.gz')
	
	if not os.path.exists(tf_filtered_path):
                
		fore_df, back_df =tf_idf_utilities.generate_fg_bg(lemma_df,tf)
		if fore_df.empty:#if the foreground does not have data: we skip the transcription factor,
                	print(f'{tf} does not have sufficient information')
               		continue
		topics_fg, topics_bg =tf_idf_utilities.doc_to_topics(fore_df,back_df) #use default = 10 at this moment 
		df_filtered =tf_idf_utilities.document_selection_hard_cutoff(topics_fg,topics_bg,back_df)
		df_final = pd.concat([fore_df,df_filtered],axis=0,ignore_index=True)
		df_final = shuffle(df_final).reset_index(drop=True)
		df_final.to_csv(tf_filtered_path,compression='gzip') #write to csv if it doesn't already exist
		#thus, {tf}_temp.csv.gz should have both class 1 and class 0 data 
	else: #if it already exsits, just read the file
                
		df_final = pd.read_csv(tf_filtered_path,compression='gzip')
		fore_df = df_final[df_final['Class'] == 1]
		df_filtered = df_final[df_final['Class'] == 0]#work as the background 
		
	#if the foreground does not have data: we skip the transcription factor, 
	#if the tf_idf results don't already exist:
	tf_res_path = os.path.join(tf_res_dir, f"{tf}_TFIDF_scores.csv")
	if not os.path.exists(tf_res_path):
		#then calcualte the tf_idf results of this transcription factor
                
		tfidf_df =tf_idf_utilities.tf_idf_enrichment(fore_df,df_filtered,tf)
		tfidf_df.to_csv(tf_res_path)
	else:
		tfidf_df = pd.read_csv(tf_res_path)

	##Word Cloud 
	word_cloud_res_path = os.path.join(word_cloud_res_dir,f"wordcloud_{tf}.png")
	if not os.path.exists(word_cloud_res_path):
		word_cloud.word_cloud_vis(tfidf_df,tf,word_cloud_res_path)
		print(f'{tf} Finished')

	else:
		print(f'{tf} Finished')







