import numpy as np

class HuffmanTreeNode():
	def __init__(self, value, possibility):
		self.possibility = possibility
		self.left = None
		self.right = None
		self.value = value

		#store the huffman code
		self.huffmancode = ""


class HuffmanTree():
	def __init__(self, word_dict, vec_len = 10000):
		#set the length of the word vector
		self.vec_len = vec_len
		self.root = None

		word_dict_list = list(word_dict.values())
		node_list = [HuffmanTreeNode(x['word'], x['possibility']) for x in word_dict_list]
		self.buildHuffman(node_list)
		self.genHuffmanCode(self.root, word_dict)

	def merge(self, node1, node2):
		top_pos = node1.possibility + node2.possibility

		#np.zeros: generate zero martix 1 row and vec_len column
		top_node = HuffmanTreeNode(np.zeros([1, self.vec_len]), top_pos)
		if node1.possibility > node2.possibility:
			top_node.left = node2
			top_node.right = node1
		else:
			top_node.left = node1
			top_node.right = node2
		return top_node

	def buildHuffman(self, node_list):
		while node_list.__len__() > 1:
			possmin = 0;
			possnextmin = 1;
			if node_list[possnextmin].possibility < node_list[possmin].possibility:
				[possmin, possnextmin] = [possnextmin, possmin]
			for i in range(2, node_list.__len__()):
				if node_list[i].possibility < node_list[possnextmin].possibility:
					possnextmin = i
					if(node_list[possnextmin].possibility < node_list[possmin].possibility):
						[possmin, possnextmin] = [possnextmin, possmin]

			top_node = self.merge(node_list[possmin], node_list[possnextmin])

			#caution the index of the list. Firstly remove the bigger index in order to keep the smaller index stable.
			#[1,2,3,4] pop(1), pop(2) => [1,3]
			#[1,2,3,4] pop(2), pop(1) => [1,4]
			if possmin < possnextmin:
				node_list.pop(possnextmin)
				node_list.pop(possmin)
			elif possmin > possnextmin:
				node_list.pop(possmin)
				node_list.pop(possnextmin)
			else:
				raise RuntimeError('possibility min should not be equal to nextmin')
			node_list.insert(0, top_node)
		self.root = node_list[0]

	def genHuffmanCode(self, node, word_dict):
		stack = [self.root]
		while stack.__len__()>0:
			node = stack.pop()

			while node.left or node.right:
				code = node.huffmancode
				node.left.huffmancode = code + "1"
				node.right.huffmancode = code + "0"
				stack.append(node.right)
				node = node.left
			word = node.value
			code = node.huffmancode
			word_dict[word]['huffmancode'] = code