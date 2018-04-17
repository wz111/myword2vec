import math

from WordFreq import * 
from HuffmanTree import *
from sklearn import preprocessing

WORDVEC_LEN = 500
LEARN_RATE = 0.05
WINDOW_LEN = 5
MODEL = 'skipGram'

class word2vec():
	def __init__(self, vec_len = WORDVEC_LEN, learn_rate = LEARN_RATE, win_len = WINDOW_LEN, model = MODEL):
		self.sents_list = None
		self.vec_len = vec_len
		self.learn_rate = learn_rate
		self.win_len = win_len
		self.model = model
		self.word_dict = None
		self.huffman = None

	def loadWordFreq(self, pretreatWord):
		if self.word_dict is not None:
			raise RuntimeError('the word dict is not empty')
		word_freq = pretreatWord.getWordFreq()

		##remove the stopwords and the punctuations in the sents
		pretreatWord.clearSents()

		self.sents_list = pretreatWord.sents_list
		self.genWordDict(word_freq)

	def genWordDict(self, word_freq):
		word_dict = {}
		if not isinstance(word_freq, dict) and not isinstance(word_freq, list):
			raise ValueError('the word freq info should be a dict or list.')

		if isinstance(word_freq, dict):
			wordSum = sum(word_freq.values())
			for word in word_freq:
				temp_dict = dict(
					word = word,
					freq = word_freq[word],
					possibility = word_freq[word]/wordSum,
					vector = np.random.random([1, self.vec_len]),
					huffmancode = None
				)
				word_dict[word] = temp_dict
		else:
			freq_list = [x[1] for x in word_freq]
			wordSum = sum(freq_list)
			for item in word_freq:
				temp_dict = dict(
					word = item[0],
					freq = item[1],
					possibility = item[1]/wordSum,
					vector = np.random.random([1, self.vec_len]),
					huffmancode = None
				)
				word_dict[item[0]] = temp_dict
		self.word_dict = word_dict

	def trainModel(self):
		if self.word_dict == None:
			raise RuntimeError('loadWordFreq method must be called before train ')
		self.huffman = HuffmanTree(self.word_dict, vec_len = self.vec_len)

		print("huffman tree generated, ready to train word-vector")
		print("")

		#print("visit huffman code: ", self.word_dict['visit']['huffmancode'])
		#begin training
		#window's head and rear
		begin = (self.win_len - 1) >> 1
		#to avoid the 
		#TypeError: slice indices must be integers or None or have an __index__ method
		end = self.win_len - 1 - begin

		if self.model == 'skipGram':
			method = self.skipGram
		else:
			method = self.cbow

		if self.sents_list:
			count = 0
			sents_len = self.sents_list.__len__()
			for sent in self.sents_list:
				sent_len = sent.__len__()
				for i in range(sent_len):
					method(sent[i], sent[max(0, i-begin):i]+sent[(i+1):min(sent_len, i+end+1)])
				count+=1
				print('{c} of {d}'.format(c = count, d = sents_len))
		else:
			raise RuntimeError('file must be cut')

		print('word vector has been generated')
		print("")

	def skipGram(self, midword, word_window):
		##debug
		##print(midword, word_window)
		##
		if not self.word_dict.__contains__(midword):
			return 

		midword_vector = self.word_dict[midword]['vector']
		for i in range(word_window.__len__())[::-1]:
			if not self.word_dict.__contains__(word_window[i]):
				word_window.pop(i)

		if word_window.__len__() == 0:
			return 

		for word in word_window:
			word_huffmancode = self.word_dict[word]['huffmancode']
			e = self.TraverseHuffman(word_huffmancode, midword_vector, self.huffman.root)
			self.word_dict[midword]['vector']+=e
			self.word_dict[midword]['vector'] = preprocessing.normalize(self.word_dict[midword]['vector'])

	def TraverseHuffman(self, word_huffmancode, input_vector, root):
		node = root
		e = np.zeros([1, self.vec_len])
		for i in range(word_huffmancode.__len__()):
			huffmancode_charat = word_huffmancode[i]
			#debug
			#if isinstance(node.value, str):
			#	print(node.value)
			#else:
			q = self.Sigmoid(input_vector.dot(node.value.T))
			#debug
			grad = self.learn_rate * (1 - int(huffmancode_charat) - q)
			e += grad * node.value
			node.value += grad * input_vector
			node.value = preprocessing.normalize(node.value)
			if huffmancode_charat == '0':
				node = node.right
			else:
				node = node.left
		return e

	def Sigmoid(self, value):
		return 1/(1+math.exp(-value))


if __name__ == '__main__':
	wordFreq = WordFreq('austen-emma.txt')
	w2v = word2vec(vec_len = WORDVEC_LEN, learn_rate = LEARN_RATE, win_len = WINDOW_LEN, model = MODEL)
	w2v.loadWordFreq(wordFreq)
	w2v.trainModel()

	standard = np.linalg.norm((w2v.word_dict['man']['vector'] - w2v.word_dict['woman']['vector']) - (w2v.word_dict['gentleman']['vector'] - w2v.word_dict['lady']['vector']))
	print(standard)
