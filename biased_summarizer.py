
# coding: utf-8

# In[42]:

import pandas as pd

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import math

try:
    import numpy
except ImportError:
    numpy = None

from sumy.summarizers._summarizer import AbstractSummarizer
from sumy._compat import  to_unicode, unicode, string_types, Counter

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import nltk
from nltk.stem.porter import *

from operator import attrgetter
from collections import namedtuple



# In[6]:

educational_description = pd.read_csv("movie.csv")


# In[44]:

txt_test="Description:     Nemo, a young clownfish, strays from the safety of the Great Barrier Reef and is captured by a diver. Placed in a dentist's aquarium in an office with an ocean view, he finds a group of fish with an escape plan. Meanwhile, Nemo's father searches for his son, meeting a number of ocean creatures along the way. Luck and Disney screenwriting lead to a happy reunion. \Benefits: 'Finding Nemo' can be used to jump-start the natural interest that children have in ocean life, coral reefs, and marine biology. It also teaches lessons about friendship, obeying parents, and avoiding dangerous situations. \This Learning Guide provides information about the animals featured in the movie. The Guide can also be used as the basis for a longer discussion of concepts from biology and coral reefs. Discussion questions focus on the animals shown in the film, biological concepts, and the film's lessons for social-emotional learning. \Possible Problems:  None. \Parenting Points: This film provides an excellent example of what can happen when kids disobey their parents and place themselves at risk. You may confront the issue directly and ask \"How did Nemo get into all that trouble?\" However, since children identify with Nemo, it may be better to approach the question obliquely. Comment about how lucky Nemo was to get out of the dentist's fish tank and how lucky he was that his father survived all the dangers of the long swim when he was searching for Nemo. The kids know very well that Nemo disobeyed his father and, as a result, was captured. " 


# In[51]:

a=[1,1,1,1]
b=[2,2,2,2]
numpy.array(a)*2+numpy.array(b)


# In[60]:




class biased_LexRank():
   

    threshold = 0.1
    epsilon = 0.1
    _stop_words = frozenset()
    words=[]
    sentences=[]
    

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, text, text_bias, bias, sentences_count):
        self._ensure_dependencies_installed()

        
        
        sentences = nltk.sent_tokenize(text) # this gives us a list of sentences
        sentences_words = [self._to_words_set(s) for s in sentences]
        if not sentences_words:
            return tuple()
        tf_metrics = self._compute_tf(sentences_words)
        idf_metrics = self._compute_idf(sentences_words)
        matrix = self._create_matrix(sentences_words, self.threshold, tf_metrics, idf_metrics)
        scores = self.power_method(matrix, self.epsilon)
        
        
        bias_lex_scores=scores[:]
        for ind,sent in enumerate(sentences):
            bias_sentences = nltk.sent_tokenize(text_bias) # this gives us a list of sentences
            bias_sentences.append(sent)
            bias_sentences_words = [self._to_words_set(s) for s in bias_sentences]
            bias_tf_metrics = self._compute_tf(bias_sentences_words)
            bias_idf_metrics = self._compute_idf(bias_sentences_words)
            bias_matrix = self._create_matrix(bias_sentences_words, self.threshold, bias_tf_metrics, bias_idf_metrics)
            bias_scores = self.power_method(bias_matrix, self.epsilon)
            sent_score = bias_scores[-1]
            
            bias_lex_scores[ind]=(numpy.array(bias_lex_scores[ind])*(1-bias))+(numpy.array(sent_score)*(bias))
        #print (scores)
        #print (bias_scores)
        print (bias_lex_scores)
        
        ratings = dict(zip(sentences, scores))
        return self._get_best_sentences(sentences, sentences_count, ratings)

    @staticmethod
    def _ensure_dependencies_installed():
        if numpy is None:
            raise ValueError("LexRank summarizer requires NumPy. Please, install it by command 'pip install numpy'.")

    def _to_words_set(self, sentence):
        
        ps = PorterStemmer()

        words = map(self.normalize_word, nltk.word_tokenize(sentence))
        return [ps.stem(w) for w in words if w not in self._stop_words]

    def _compute_tf(self, sentences):
        tf_values = map(Counter, sentences)

        tf_metrics = []
        for sentence in tf_values:
            metrics = {}
            max_tf = self._find_tf_max(sentence)

            for term, tf in sentence.items():
                metrics[term] = tf / max_tf

            tf_metrics.append(metrics)

        return tf_metrics

    @staticmethod
    def _find_tf_max(terms):
        return max(terms.values()) if terms else 1

    @staticmethod
    def _compute_idf(sentences):
        idf_metrics = {}
        sentences_count = len(sentences)

        for sentence in sentences:
            for term in sentence:
                if term not in idf_metrics:
                    n_j = sum(1 for s in sentences if term in s)
                    idf_metrics[term] = math.log(sentences_count / (1 + n_j))

        return idf_metrics

    def _create_matrix(self, sentences, threshold, tf_metrics, idf_metrics):
        """
        Creates matrix of shape |sentences|×|sentences|.
        """
        # create matrix |sentences|×|sentences| filled with zeroes
        sentences_count = len(sentences)
        matrix = numpy.zeros((sentences_count, sentences_count))
        degrees = numpy.zeros((sentences_count, ))

        for row, (sentence1, tf1) in enumerate(zip(sentences, tf_metrics)):
            for col, (sentence2, tf2) in enumerate(zip(sentences, tf_metrics)):
                matrix[row, col] = self.compute_distance(sentence1, sentence2, tf1, tf2, idf_metrics)

                if matrix[row, col] > threshold:
                    matrix[row, col] = 1.0
                    degrees[row] += 1
                else:
                    matrix[row, col] = 0

        for row in range(sentences_count):
            for col in range(sentences_count):
                if degrees[row] == 0:
                    degrees[row] = 1

                matrix[row][col] = matrix[row][col] / degrees[row]

        return matrix
    def normalize_word(self, word):
        return str(word).lower()
    
    def _get_best_sentences(self, sentences, count, rating, *args, **kwargs):
        rate = rating
        SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",))
        if isinstance(rating, dict):
            assert not args and not kwargs
            rate = lambda s: rating[s]

        infos = (SentenceInfo(s, o, rate(s, *args, **kwargs))
            for o, s in enumerate(sentences))

        # sort sentences by rating in descending order
        infos = sorted(infos, key=attrgetter("rating"), reverse=True)
        # get `count` first best rated sentences
        if not isinstance(count, ItemsCount):
            count = ItemsCount(count)
        infos = count(infos)
        # sort sentences by their order in document
        infos = sorted(infos, key=attrgetter("order"))

        return tuple(i.sentence for i in infos)

    #@staticmethod

    @staticmethod
    def power_method(matrix, epsilon):
        transposed_matrix = matrix.T
        sentences_count = len(matrix)
        p_vector = numpy.array([1.0 / sentences_count] * sentences_count)
        lambda_val = 1.0

        while lambda_val > epsilon:
            next_p = numpy.dot(transposed_matrix, p_vector)
            lambda_val = numpy.linalg.norm(numpy.subtract(next_p, p_vector))
            p_vector = next_p

        return p_vector
    @staticmethod
    def compute_distance(sentence1, sentence2, tf1, tf2, idf_metrics):
        common_words = frozenset(sentence1) & frozenset(sentence2)

        numerator = 0.0
        for term in common_words:
            numerator += tf1[term]*tf2[term] * idf_metrics[term]**2

        denominator1 = sum((tf1[t]*idf_metrics[t])**2 for t in sentence1)
        denominator2 = sum((tf2[t]*idf_metrics[t])**2 for t in sentence2)

        if denominator1 > 0 and denominator2 > 0:
            return numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))
        else:
            return 0.0
        
class ItemsCount(object):
    def __init__(self, value):
        self._value = value

    def __call__(self, sequence):
        if isinstance(self._value, string_types):
            if self._value.endswith("%"):
                total_count = len(sequence)
                percentage = int(self._value[:-1])
                # at least one sentence should be chosen
                count = max(1, total_count*percentage // 100)
                return sequence[:count]
            else:
                return sequence[:int(self._value)]
        elif isinstance(self._value, (int, float)):
            return sequence[:int(self._value)]
        else:
            ValueError("Unsuported value of items count '%s'." % self._value)

    def __repr__(self):
        return to_string("<ItemsCount: %r>" % self._value)


    
    
    
    


# In[62]:


    
txt_bias="natural interest"
summarizer = biased_LexRank()
for sentence in summarizer(txt_test, txt_bias, 0.5, 3):
    print(sentence)

