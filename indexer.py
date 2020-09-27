from nltk.stem import PorterStemmer
from nltk import word_tokenize, bigrams
from hashlib import sha256
from urllib.parse import urlparse
from collections import defaultdict
from lxml import html
from math import log,sqrt
import os
import psutil
import json
import io
import pickle
import urllib.parse
import ujson

docIDHashMap = dict()
pageRankDict = dict()
#in megabytes
RESOURCE_LIMIT = 3000

#returns process usage in megabytes
def getCurrentMemoryUsage(l):
    process = psutil.Process(os.getpid())
    return process.memory_info().rss/pow(1000,2)

#borrowed from Assignment 2
def get_urlhash(url):
    parsed = urlparse(url)
    # everything other than scheme.
    return sha256(
        f"{parsed.netloc}/{parsed.path}/{parsed.params}/"
        f"{parsed.query}/{parsed.fragment}".encode("utf-8")).hexdigest()

#borrowed from Assignment 2  
def normalize(url):
    if url.endswith("/"):
        return url.rstrip("/")
    return url

#check if word is alphanumeric or ascii
def valid(word):
    if (word.isalnum()):
        try:
            word.encode('ascii')
        except UnicodeEncodeError:
            return False
        return True
    return False

def clearFile(file):
    f = open(file, 'w')
    f.write("")
    f.close()
    
#returns a list of tokens and their respective frequencies
def tokenFrequency(text):
    ps = PorterStemmer()
    tokens = [ps.stem(t).lower() for t in word_tokenize(text) if valid(t)]
    tokens.extend([",".join(b) for b in bigrams(tokens)])
    frequencyDict = defaultdict(int)
    for t in tokens:
        frequencyDict[t]+=1
    return [(t, frequencyDict[t]) for t in set(tokens)]

#retrieves both the partial indices and all the files in the index that start with
#a word from the alphabet
def retrieveFileList(mode):
    files = []
    for root, _, file in os.walk("index"):
        for doc in file:
            if doc.endswith('.txt') and "data" in doc and mode == "DATA":
                files.append(root + "/" + doc)
            elif "data" not in doc and "DS" not in doc and mode == "ALPHABET":
                files.append(root + "/" + doc)
    return files

#writes the postings to the file
def writeToFile(counter, postings):
    with open(f"index/data_dump_{counter}.txt", 'w+') as fp:
        for token,postingList in postings.items():
            fp.write(token + ";" + ujson.dumps(postingList) + '\n')
     
#returns all the text in bold tags given an html tree       
def getBoldText(tree):
    ps = PorterStemmer()
    bold_list = []
    if tree.getroot() == None:
        return []
    for bold_text in tree.xpath('//b//text()'):
        tokenized_b = word_tokenize(bold_text)
        for bold_word in tokenized_b:
            if valid(bold_word):
                bold_list.append(ps.stem(bold_word))
    for bold_text in tree.xpath('//strong//text()'):
        tokenized_b = word_tokenize(bold_text)
        for bold_word in tokenized_b:
            if valid(bold_word):
                bold_list.append(ps.stem(bold_word))
    return bold_list

#returns all the links leaving a page given an html tree
def getOutwardLinks(tree, url):
    if tree.getroot() == None:
        return set()
    hrefs = tree.xpath('//a/@href')
    links = set()
    for href in hrefs:
        links.add(get_urlhash(urllib.parse.urldefrag(urllib.parse.urljoin(url, href))[0]))
    return links

#retrieves all text in the title given an html tree
def getTitleText(tree):
    ps = PorterStemmer()
    title_list = []
    if tree.getroot() == None:
        return []
    for title_text in tree.xpath('//title//text()'):
        tokenized_title = word_tokenize(title_text)
        for t in tokenized_title:
            if valid(t):
                title_list.append(ps.stem(t))
    return title_list

#retrieves all text in tags h1-h6 given an html tree
def getHeadingText(tree):
    ps = PorterStemmer()
    headings = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    heading_list = []
    if tree.getroot() == None:
        return []
    for heading in headings:
        for heading_text in tree.xpath(f'//{heading}//text()'):
            tokenized_heading = word_tokenize(heading_text)
            for heading_word in tokenized_heading:
                if valid(heading_word):
                    heading_list.append(ps.stem(heading_word))
    return heading_list
            
#returns an html tree given json content
def getTree(content):
    parser = html.HTMLParser(remove_comments=True, recover = True)
    tree = html.parse(io.BytesIO(content.encode('utf-8', 'ignore')), parser)
    return tree

#returns all text without html tags given an html tree
def getText(tree):
    root = tree.getroot()
    if root != None:
        return root.text_content()
    return ""

#checks to see if a link is a fragment or not
def valid_link(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.fragment != "":
        return False
    return True
    
#constructs a postings list given a list of json files and writes said postings to disk
def constructPartialIndicesHelper(files):
    fileCounter = 0
    global docIDHashMap
    postings = defaultdict(list)
    for doc in files:
        try:
            data = ujson.load(open(doc, 'r'))
            url = normalize(data['url'])
            html_content = data['content']
            if valid_link(url):
                docID = get_urlhash(url)
                docIDHashMap[docID] = url
                tree = getTree(html_content)
                text = getText(tree)
                bold_text = getBoldText(tree)
                title_text = getTitleText(tree)
                heading_text = getHeadingText(tree)
                tokenStemList = tokenFrequency(text)
                for token,frequency in tokenStemList:
                    field_dict = {'bold': False, 'title': False, 'heading': False}
                    if token in bold_text:
                        field_dict['bold'] = True
                    if token in title_text:
                        field_dict['title'] = True
                    if token in heading_text:
                        field_dict['heading'] = True
                    postings[token].append((docID, frequency, field_dict))
                if getCurrentMemoryUsage() > RESOURCE_LIMIT:
                    print(getCurrentMemoryUsage())
                    print(f"About to write to file {fileCounter}")
                    writeToFile(fileCounter, postings)
                    fileCounter+=1
                    postings = defaultdict(list)
                    print(getCurrentMemoryUsage())

                    
        except UnicodeDecodeError:
            print("ERROR")
        
    print(f"About to write to file {fileCounter}")
    writeToFile(fileCounter, postings)

#retrieves all files in the DEV folder and creates partial indices
def constructPartialIndices():
    files = []
    for root, _, file in os.walk("DEV"):
        for doc in file:
            files.append(root + "/" + doc)
    constructPartialIndicesHelper(files)
    

def mergeLists(list1, list2):
    l1 = ujson.loads(list1)
    l2 = ujson.loads(list2)
    list1_dict = {key:(val,field_dict) for key,val,field_dict in l1}
    for doc,freq,field_dict in l2:
        if doc in list1_dict:
            print(list1_dict[doc])
            list1_dict[doc][0]+=freq
        else:
            list1_dict[doc] = (freq,field_dict)
    return [(key,value[0],value[1]) for key,value in list1_dict.items()]

#creates a txt file for every file in the alphabet as well as a nums file
#for numeric tokens
def createLetterFiles():
    for letter in "abcdefghijklmnopqrstuvwxyz":
        f = open("index/"+letter + ".txt", "w+")
        f.write("")
        f.close()
    f = open("index/nums.txt", "w+")
    f.write("")
    f.close()                          

#takes all the data and writes it to one of the "alphabet" files
def mergerHelper(current_file):
    word_dict = dict()
    f = open(current_file, 'r')
    for line in f:
        current_token, current_list = line.split(";", 1)
        first_letter = current_token[0]
        alphabet =  first_letter if first_letter.isalpha() else "nums"  
        if alphabet in word_dict:
            if current_token in word_dict[alphabet]:
                newList = mergeLists(word_dict[alphabet][current_token], current_list)
                word_dict[alphabet][current_token] = newList
            else:
                word_dict[alphabet][current_token] = ujson.loads(current_list)
        else:
            word_dict[alphabet] = dict()
            word_dict[alphabet][current_token] = ujson.loads(current_list) 
    f.close()
    for alphabet in word_dict:
        file_to_write = open("index/"+ alphabet + ".txt", 'a')
        for current_token,current_list in word_dict[alphabet].items():
            file_to_write.write(current_token + ";" + ujson.dumps(current_list)+"\n")
        file_to_write.close()

def mergeIndices():
    for f in retrieveFileList("DATA"):
        mergerHelper(f)
        os.remove(f)
    sortFiles()
        
#merges/sorts all the alphabet files such that tokens that appear multiple
#times in their respective file are merged into one entry
def sortFilesHelper(file):
    print(f"Beginning to sort {file}")
    f = open(file, 'r')
    temp_dict = dict()
    for line in f:
        current_token, current_list = line.split(";", 1)
        if current_token in temp_dict:
            temp_dict[current_token] = mergeLists(ujson.dumps(temp_dict[current_token]), current_list)
        else:
            temp_dict[current_token] = ujson.loads(current_list)
    f.close()
    clearFile(file)
    f_write = open(file, 'w')
    for current_token,current_list in sorted(temp_dict.items()):
        f_write.write(current_token + ";" + ujson.dumps(current_list) + "\n")
    f_write.close()
    
def sortFiles():
    files = retrieveFileList("ALPHABET")
    for f in files:
        sortFilesHelper(f)

#adds td-idf score for every token
def addTF_IDFHelper(file,N):
    print(f"Beginning to compute tf-idf-score {file}")
    f = open(file, 'r')
    temp_dict = dict()
    for line in f:
        current_token, current_list = line.split(";", 1)
        current_list = ujson.loads(current_list)
        df = len(current_list)
        current_list = [doc[:3] + [calculateTF_IDF(doc[1], df, N),] for doc in current_list]
        temp_dict[current_token] = ujson.dumps(current_list)
    f.close()
    clearFile(file)
    f_write = open(file, 'w')
    for current_token,current_list in temp_dict.items():
        f_write.write(current_token + ";" + current_list + "\n")
    f_write.close()
    f.close()
        
        
def addTF_IDF():
    files = retrieveFileList("ALPHABET")
    N = len(docIDHashMap)
    for f in files:
        addTF_IDFHelper(f,N)
     
   
def getNormalizationValue(d):
    result = 0
    for weight in d.values():
        result += pow(weight, 2)
    return sqrt(result)
        
#maps a document and a term to their respective td-idf scores
def docWeightMap():
    files = retrieveFileList("ALPHABET")
    docWeightDict = dict()
    for file in files:
        f = open(file, 'r')
        for line in f:
            current_token,current_list = line.split(";", 1)
            for doc, _, _,weight in ujson.loads(current_list):
                if doc not in docWeightDict:
                    docWeightDict[doc] = dict()
                docWeightDict[doc][current_token] = weight
        f.close()
    for doc in docWeightDict:
        normalizationValue = getNormalizationValue(docWeightDict[doc])
        for term in docWeightDict[doc]:
            docWeightDict[doc][term] = docWeightDict[doc][term]/normalizationValue
    return docWeightDict
        
def calculateTF_IDF(tf, df, N):
    return (1 + log(tf)) * (log(N/df))

#normalizes all td-idf scores using a normalization factor
def addNormalizedIdfHelper(file,docWeights):
    print(f'Adding normalized tf-idf to {file}')
    f = open(file, 'r')
    temp_dict = dict()
    for line in f:
        current_token, current_list = line.split(";", 1)
        current_list = ujson.loads(current_list)
        current_list = [(doc,freq,fields,docWeights[doc][current_token]) for doc,freq,fields,_ in current_list]
        temp_dict[current_token] = ujson.dumps(current_list)
    f.close()
    clearFile(file)
    f_write = open(file, 'w')
    for current_token,current_list in temp_dict.items():
        f_write.write(current_token + ";" + current_list + "\n")
    f_write.close()
    f.close()
    
def addNormalizedIdf():
    docWeights = docWeightMap()
    files = retrieveFileList("ALPHABET")
    for f in files:
        addNormalizedIdfHelper(f, docWeights)
        
#"indexes the index" by mapping each token to its offset in its respective file
def createOffsetIndex():
    offset_index = dict()
    for file in retrieveFileList("ALPHABET"):
        f = open(file, 'r')
        offset = 0
        offset_index[file] = dict()
        for line in iter(f.readline, ''):
            token,_ = line.split(";")
            offset_index[file][token] = offset
            offset+=len(line)
    return offset_index

#splits the index into a high and low tiers based on their page rank scores
#cutoff score chosen is slightly lower than the median page rank score
def createTieredIndexHelper(file,cutoff_score, pageRankScores):
    print(f'Beginning to split {file} into tiers')
    upper_dict = dict()
    lower_dict = dict()
    f = open(file, 'r')
    for line in f:
        current_token, current_list = line.split(";")
        current_list = ujson.loads(current_list)
        
        upper_list = [doc for doc in current_list if pageRankScores[doc[0]]>=cutoff_score]
        lower_list = [doc for doc in current_list if pageRankScores[doc[0]]<cutoff_score]
        if len(upper_list) > 0:
            upper_dict[current_token] = ujson.dumps(upper_list)
        if len(lower_list) > 0:
            lower_dict[current_token] = ujson.dumps(lower_list)
    f.close()
    clearFile(file)
    lower_write = open(file, 'w')
    for current_token,current_list in lower_dict.items():
        lower_write.write(current_token + ";" + current_list + "\n")
    lower_write.close()
    if "nums" in file:
        upper_write = open("index/nums_high.txt", 'w')
    else:
        upper_write = open(f'index/{current_token[0]}_high.txt', 'w')
    for current_token,current_list in upper_dict.items():
        upper_write.write(current_token + ";" + current_list + "\n")
    upper_write.close()
    

def createTieredIndex():
    files = retrieveFileList("ALPHABET")
    pageRankScores = getPageRankScores()
    cutoff_score = .2
    for f in files:
        createTieredIndexHelper(f, cutoff_score, pageRankScores)
    
def run():
    print("Creating letter files..")
    createLetterFiles()
    print("..Done")
    print("Constructing partial indices..")
    constructPartialIndices()
    print("..Done")
    print("Merging Indices..")
    mergeIndices()
    print("..Done")
    print("Sorting files..")
    sortFiles()
    print("..Done")
    print("Calculating tf-idf for each document...")
    addTF_IDF()
    print("Normalizing tf-idf values...")
    addNormalizedIdf()
    print("Pickling hash map..")
    pickle_hashmap()
    print("..Done")
    print("Pickling posting length dict...")
    pickle_PostingLength()
    print("..Done")
    print("Constructing page rank dictionary...")
    constructPageRankDict()
    print("...Done")
    print("Calculating page rank scores...")
    calculatePageRankScores()
    print("..Done")
    print("Creating tiered index...")
    createTieredIndex()
    print("...Done")
    print("Pickling offset index dict...")
    pickle_offsetIndex()
    print("Done!")
    
#pickles hashmap for use in the controller class
def pickle_hashmap():
    with open('hash_map.pickle', 'wb') as file:
        pickle.dump(dict(docIDHashMap), file, protocol=pickle.HIGHEST_PROTOCOL)

#pickles posting length dictionary for use in query normalization (calculating query df quickly)
def pickle_PostingLength():
    files = retrieveFileList("ALPHABET")
    postingLengthDict = dict()
    for file in files:
        f = open(file, 'r')
        for line in f:
            current_token,current_list = line.split(";", 1)
            postingLengthDict[current_token] = len(ujson.loads(current_list))
        f.close()
    with open('posting_length_dict.pickle', 'wb') as file:
        pickle.dump(postingLengthDict, file, protocol = pickle.HIGHEST_PROTOCOL)
        
#pickles offset values for quick postings retrieval
def pickle_offsetIndex():
    with open ('offset_dict.pickle', 'wb') as file:
        pickle.dump(createOffsetIndex(), file, protocol=pickle.HIGHEST_PROTOCOL)

def get_offsetIndex():
    with open('offset_dict.pickle', 'rb') as file:
        offset_dict = pickle.load(file)
    return offset_dict
    
def get_url_hashmap():
    with open('hash_map.pickle', 'rb') as file:
        hashmap = pickle.load(file)
    return hashmap

def get_posting_length_dict():
    with open('posting_length_dict.pickle', 'rb') as file:
        postingLengthDict = pickle.load(file)
    return postingLengthDict

def getPageRankScores():
    with open('pageRankScores.json', 'rb') as file:
        pageRankScores = ujson.load(file)
    return pageRankScores

#maps all hashed documents to hashes of the urls they point to (maps ints to ints)
def constructPageRankDict():
    global pageRankDict
    for root, _, file in os.walk("DEV"):
        for doc in file:
            try:
                data = json.load(open(root + "/" + doc, 'r'))
                url = normalize(data['url'])
                html_content = data['content']
                if valid_link(url):
                    tree = getTree(html_content)
                    pageRankDict[get_urlhash(url)] = getOutwardLinks(tree, url)
            except UnicodeDecodeError:
                print("ERROR")
    with open('pageRankDict.json', 'w') as f:
        ujson.dump(pageRankDict, f)
        
#uses the page rank algorithm to weight different pages differently
def calculatePageRankScores():
    pageRankDict = ujson.load(open('pageRankDict.json', 'rb'))
    pageRankScores = dict()
    N = len(pageRankDict)
    d = .85
    for doc in pageRankDict:
        pageRankScores[doc] = 1
    print("Beginning iteration 1")
    tmp = dict()
    for i in pageRankDict:
        score_sum= sum([(1/len(pageRankDict[j]) if i in pageRankDict[j] else 0)*pageRankScores[j] for j in pageRankDict if i!=j])
        tmp[i] = (1-d) + d*score_sum
    pageRankScores = tmp
        
    print("Iteration 1 complete")
    iteration = 1
    while(True):
        tmp = dict()
        for i in pageRankDict:
            score_sum= sum([(1/len(pageRankDict[j]) if i in pageRankDict[j] else 0)*pageRankScores[j] for j in pageRankDict if i!=j])
            tmp[i] = (1-d) + d*score_sum
        if tmp == pageRankScores:
            break
        pageRankScores = tmp
        iteration+=1
        print(f'Iteration {iteration} complete')
        if iteration >= 10:
            break
        
    with open('pageRankScores.json', 'w') as f:
        ujson.dump(pageRankScores, f)
    
    
if __name__ == '__main__':
    run()







            
        
        
        




