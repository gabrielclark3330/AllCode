import csv
import os

def isWordInTrie(trie, word)->bool:
	node = trie
	for char in word:
		if char not in node:
			return False
		else:
			node = node[char]
	if "\0" in node:
		return True
	else:
		return False

def isSufixInTrie(trie, word)->bool:
	node = trie
	for char in word:
		if char not in node:
			return False
		else:
			node = node[char]
	return False

def nextPossibleLetters(trie, word)->[]:
	node = trie
	for char in word:
		if char not in node:
			return []
		else:
			node = node[char]
	return list(node.keys())

wordTrie = {"":{}}
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'scrableDictionary.csv')
file = open(filename, "r")
csv_reader = csv.reader(file)
for word in csv_reader:
	node = wordTrie
	for char in word[0]:
		if char in node:
			node = node[char]
		else:
			node[char] = {}
			node = node[char]
	node["\0"] = "\0"


foundWords = set()
boardStr = input()
board = []
boardSize = 4
for row in range(boardSize):
  line = []
  for col in range(boardSize):
    line.append(boardStr[row*4+col])
  board.append(line)


def dfsForWords(position, path, word):
	if isSufixInTrie(wordTrie, word):
		return

	if isWordInTrie(wordTrie, word):
		foundWords.add(word)

	for x in range(-1,2):
		for y in range(-1,2):
			tempPos = (position[0]+x, position[1]+y)
			if tempPos not in path:
				if tempPos[0] >= 0 and tempPos[0] < boardSize and tempPos[1] >= 0 and tempPos[1] < boardSize:
					if (board[tempPos[0]][tempPos[1]]) in nextPossibleLetters(wordTrie, word):
						tempPath = path.copy()
						tempPath.add(tempPos)
						tempWord = word + (board[tempPos[0]][tempPos[1]])
						dfsForWords(tempPos, tempPath, tempWord)


for row in range(boardSize):
	for col in range(boardSize):
		path = set()
		currentWord = ""
		position = (row, col)
		dfsForWords(position, path, currentWord)

displayWords = sorted(list(foundWords), key=lambda x:len(x))
for element in displayWords:
	print(element)
