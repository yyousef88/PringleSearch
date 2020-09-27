from indexer import get_url_hashmap, calculateTF_IDF, getNormalizationValue, get_posting_length_dict, get_offsetIndex
from nltk import bigrams
from nltk.stem import PorterStemmer
from collections import defaultdict
import numpy as np
import ujson

class Controller():
    def __init__(self, query = ""):
        self.query = query
        self.url_hashmap = get_url_hashmap()
        self.tokens = []
        self.postingLengthDict = get_posting_length_dict()
        self.offset_dict = get_offsetIndex()
            
    def setQuery(self, new_query):
        self.query = new_query
    
    #splits query into bigrams and n terms
    def _parse_query(self):
        ps = PorterStemmer()
        self.tokens = [ps.stem(t) for t in self.query.split()]
        self.tokens.extend([",".join(b) for b in bigrams(self.tokens)])
        return self.tokens
    
    #retrieves the postings for a given word (mode is used to
    #differentiate between the high index and the low index
    def _retrievePostings(self, word, mode = ""):
        first_letter = word[0]
        if first_letter.isalpha():
            f = f'index/{first_letter}{mode}.txt'
        else:
            f = f"index/nums{mode}.txt"
        if word not in self.offset_dict[f]:
            return list()
        file = open(f, 'r')
        #uses the offset index to quickly find the word
        file.seek(self.offset_dict[f][word])
        _,l = file.readline().split(";")
        #quickly loads the postings list using ujson
        return ujson.loads(l)
        
    #returns a count of how many times a term appears in the query
    def _queryTermCount(self):
        tmp = defaultdict(int)
        for term in self._parse_query():
            tmp[term] += 1
        return [(term,count) for term,count in tmp.items()]
    
    #calculates the vector for query for use in cosine similary
    def getQueryVector(self):
        queryWeights = dict()
        for term,count in self._queryTermCount():
            postingsLength = self.postingLengthDict[term] if term in self.postingLengthDict else 0
            if postingsLength > 0:
                queryWeights[term] = calculateTF_IDF(count, postingsLength, len(self.url_hashmap))
        queryNormalizationVal = getNormalizationValue(queryWeights)
        for term, weight in queryWeights.items():
            queryWeights[term] = weight/queryNormalizationVal
        return queryWeights
    
    #calculates a score based on the number of tokens in a document that are in a 
    #special field
    def fieldScore(self, field):
        score = 0
        for t in self.tokens:
            if t in field:
                score+= .2 * field[t]['bold'] + .2*field[t]['heading'] + .3 *field[t]['title']
        return score
    
    #creates a vector of every document that contains at least one of the terms
    #calculates the cosine similarity between every document and the query
    #sorts the document list by the cosine similarity score + the field score
    def retrieveUrls(self):
        queryWeights = self.getQueryVector()
        self.docWeights = dict()
        self.docFields = dict()
        documents = [doc + [t] for t in self.tokens for doc in self._retrievePostings(t, mode = "_high")]  
        if len(documents) < 10:
            documents.extend([doc + [t] for t in self.tokens for doc in self._retrievePostings(t)])      
        for doc,_,fields,tfidf,t in documents:
            if doc not in self.docWeights:
                self.docWeights[doc] = dict()
            if doc not in self.docFields:
                self.docFields[doc] = dict()
            self.docFields[doc][t] = fields
            self.docWeights[doc][t] = tfidf
        q = [val for val in queryWeights.values()]
        docList = {doc: np.dot(q,[self.docWeights[doc][term] if term in self.docWeights[doc] else 0 for term in queryWeights]) + self.fieldScore(self.docFields[doc])for doc in self.docWeights}
        return [self.url_hashmap[url] for url,_ in sorted(docList.items(), key = lambda x: -x[1])]
        
    #used for boolean retrieval
    def _retrieveDocumentsBoolean(self):
        posting_sets = []
        for term in self._parse_query():
            posting_sets.append(set(self._retrievePostings(term)))
        if len(posting_sets) == 0:
            return list()
        elif len(posting_sets) == 1:
            return posting_sets[0]
        else:
            return set.intersection(*posting_sets)       
        
    
            
        
            
