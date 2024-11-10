import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nlp = spacy.load("en_core_web_md")

def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
	"""
	_______________________________________________________
	Parameters:
	words - 1D Array with 16 shuffled words
	strikes - Integer with number of strikes
	isOneAway - Boolean if your previous guess is one word away from the correct answer
	correctGroups - 2D Array with groups previously guessed correctly
	previousGuesses - 2D Array with previous guesses
	error - String with error message (0 if no error)

	Returns:
	guess - 1D Array with 4 words
	endTurn - Boolean if you want to end the puzzle
	_______________________________________________________
	"""

	# Your Code here
	# Good Luck!
	#Generate word embedding using spaCy
	word_vectors = {word: nlp(word).vector for word in words}

	#clculate pairwise similarity matrix
	word_list = list(word_vectors.keys())
	vectors = np.array([word_vectors[word] for word in word_list])
	similarity_matrix = cosine_similarity(vectors)

	#Initialize groups and guesses
	candidate_groups = []
	used_words = set(word for group in correctGroups for word in group)

	#Find candidate groups using a threshold in similarity
	threshold = 0.75
	for i, word in enumerate(word_list):
		if word not in used_words:
			group = [word]
			for j in range(i + 1, len(word_list)):
				if word_list[j] not in used_words and similarity_matrix[i, j] > threshold:
					group.append(word_list[j])
			if len(group) == 4:
				candidate_groups.append(group)
				used_words.update(group)

	if candidate_groups:
		guess = candidate_groups[0]
	else:
		unused_words = [word for word in word_list if word not in used_words]
		guess = unused_words[:4]
	
	endTurn = (strikes >= 3 or not candidate_groups)

	return guess, endTurn

