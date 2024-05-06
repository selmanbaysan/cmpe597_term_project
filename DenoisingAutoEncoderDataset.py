from torch.utils.data import Dataset
from typing import List
from sentence_transformers.readers.InputExample import InputExample
import numpy as np
from transformers.utils.import_utils import is_nltk_available, NLTK_IMPORT_ERROR


class DenoisingAutoEncoderDataset(Dataset):
    """
    The DenoisingAutoEncoderDataset returns InputExamples in the format: texts=[noise_fn(sentence), sentence]
    It is used in combination with the DenoisingAutoEncoderLoss: Here, a decoder tries to re-construct the
    sentence without noise.

    :param sentences: A list of sentences
    :param noise_fn: A noise function: Given a string, it returns a string with noise, e.g. deleted words
    """

    def __init__(self, sentences: List[str], noise_fn=lambda s: DenoisingAutoEncoderDataset.delete(s)):
        if not is_nltk_available():
            raise ImportError(NLTK_IMPORT_ERROR.format(self.__class__.__name__))

        self.sentences = sentences
        self.noise_fn = noise_fn

    def __getitem__(self, item):
        sent = self.sentences[item]
        return InputExample(texts=[self.noise_fn(sent), sent])

    def __len__(self):
        return len(self.sentences)

    # Deletion noise.
    @staticmethod
    def delete(text, del_ratio=0.6):
        from nltk import word_tokenize, TreebankWordDetokenizer

        words = word_tokenize(text)
        n = len(words)
        if n == 0:
            return text

        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0:
            keep_or_not[np.random.choice(n)] = True  # guarantee that at least one word remains
        words_processed = TreebankWordDetokenizer().detokenize(np.array(words)[keep_or_not])
        return words_processed
    
    # Random swap noise.
    @staticmethod
    def swap(text, swap_ratio=0.1):
        from nltk import word_tokenize, TreebankWordDetokenizer

        words = word_tokenize(text)
        n = len(words)
        if n == 0:
            return text

        swap_idx = np.random.choice(n, int(n * swap_ratio), replace=False)
        words_processed = words.copy()
        for i in swap_idx:
            if i + 1 < n:
                words_processed[i], words_processed[i + 1] = words_processed[i + 1], words_processed[i]
        words_processed = TreebankWordDetokenizer().detokenize(words_processed)
        return words_processed
    
    # Random noun swap noise.
    @staticmethod
    def noun_swap(text, swap_ratio=0.6):
        from nltk import word_tokenize, TreebankWordDetokenizer
        from nltk.tag import pos_tag
        words = word_tokenize(text)
        n = len(words)
        if n == 0:
            return text

        pos_tags = pos_tag(words, lang="eng", tagset="universal")
        noun_idx = [i for i, tag in enumerate(pos_tags) if tag[1] == "NOUN"]
        # check if there are any nouns
        if len(noun_idx) == 0:
            return text
        # create a list of indices to swap
        swap_idx = np.random.choice(noun_idx, int(len(noun_idx) * swap_ratio), replace=False)
        words_processed = words.copy()
        for i in swap_idx:
            swap_candidates = [j for j in noun_idx if j != i]
            if swap_candidates:
                j = np.random.choice(swap_candidates)
                words_processed[i], words_processed[j] = words_processed[j], words_processed[i]
                noun_idx.remove(i)
                noun_idx.remove(j)

        words_processed = TreebankWordDetokenizer().detokenize(words_processed)
        return words_processed

    # Randomly change the word to a synonym using WordNet.
    @staticmethod
    def synonym_replacement(text, replace_ratio=1):
        from nltk import word_tokenize, TreebankWordDetokenizer
        from nltk.corpus import wordnet

        words = word_tokenize(text)
        n = len(words)
        if n == 0:
            return text

        replace_idx = np.random.choice(n, int(n * replace_ratio), replace=False)
        words_processed = words.copy()
        for i in replace_idx:
            synsets = wordnet.synsets(words_processed[i])
            if synsets:
                words_processed[i] = synsets[0].lemmas()[0].name()
        words_processed = TreebankWordDetokenizer().detokenize(words_processed)
        return words_processed