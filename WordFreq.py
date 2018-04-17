import nltk
from nltk.corpus import gutenberg
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk import FreqDist

class WordFreq():
	def __init__(self, filepath):
		#the corpus must be gutenberg
		self.filepath = filepath
		self.words_list = gutenberg.words(filepath)
		self.sents_list = gutenberg.sents(filepath)
		self.english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'']
	def getWordFreq(self):
		lower_wordlist = []
		for word in self.words_list:
			lower_wordlist.append(word.lower())

		filtered_wordlist = [word for word in lower_wordlist if not word in stopwords.words('english')]
		
		filtered_wordlist = [word for word in filtered_wordlist if not word in self.english_punctuations]

		##porter stemmer
		ps = PorterStemmer()
		stemmed_wordlist = [ps.stem(word) for word in filtered_wordlist]

		worDistemp = FreqDist()
		worDist = FreqDist()
		for word in stemmed_wordlist:
			worDist[word]+=1
			worDistemp[word]+=1
		for word in worDistemp:
			if worDistemp[word] == 1:
				del worDist[word]

		#print(worDist.most_common(20))
		return worDist

	def clearSents(self):
		lower_sentslist = []
		for line in self.sents_list:
			lower_sent = []
			for word in line:
				lower_sent.append(word.lower())
			lower_sent = [word for word in lower_sent if not word in stopwords.words('english')]
			lower_sent = [word for word in lower_sent if not word in self.english_punctuations]
			lower_sentslist.append(lower_sent)

		self.sents_list = lower_sentslist
		#print(self.sents_list[0])


#if __name__ == '__main__':
#	wf = WordFreq('shakespeare-hamlet.txt')
#	wf.clearSents()