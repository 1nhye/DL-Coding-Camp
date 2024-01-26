from typing import Optional, Union, List, Tuple, Dict
import re
import collections


class BaseTokenizer:
    def __init__(self, corpus: Optional[Union[List[str], str]] = None) -> None:
        self.vocab = {}
        self.merges = {}
        if corpus:
            self.add_corpus(corpus)

    def add_corpus(self, corpus: Union[List[str], str]) -> None:
        raise NotImplementedError(
            "add_corpus method must be implemented in derived classes")

    def get_stats(self) -> Dict[Tuple[str, str], int]:
        raise NotImplementedError(
            "get_stats method must be implemented in derived classes")

    def merge_vocab(self, pair: Tuple[str, str], v_in: Dict[str, int]) -> Dict[str, int]:
        raise NotImplementedError(
            "merge_vocab method must be implemented in derived classes")

    def train(self, n_iter: int) -> None:
        raise NotImplementedError(
            "train method must be implemented in derived classes")

    def tokenize(self, text: Union[List[str], str], padding: bool = False, max_length: Optional[int] = None) -> Union[List[List[int]], List[int]]:
        raise NotImplementedError(
            "tokenize method must be implemented in derived classes")

    def __call__(self, text: Union[List[str], str], padding: bool = False, max_length: Optional[int] = None) -> Union[List[List[int]], List[int]]:
        return self.tokenize(text, padding, max_length)


class BPETokenizer(BaseTokenizer):
    def add_corpus(self, corpus: Union[List[str], str]) -> None:
        if isinstance(corpus, str):
            corpus = [corpus]
        for sentence in corpus:
            symbols = sentence.split()
            for symbol in symbols:
                self.vocab[symbol] += 1

    def get_stats(self) -> Dict[Tuple[str, str], int]:
        pairs = collections.defaultdict(int)

        if not self.vocab or len(self.vocab) == 1:
            return pairs

        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair: Tuple[str, str], v_in: Dict[str, int]) -> Dict[str, int]:
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\\S)' + bigram + r'(?!\\S)')
        for word in self.vocab:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = self.vocab[word]
        return v_out

    def train(self, n_iter: int) -> None:
        for i in range(n_iter):
            pairs = self.get_stats()

            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            self.vocab = self.merge_vocab(best)
            self.merges[i] = best

    def tokenize(self, text: Union[List[str], str], padding: bool = False, max_length: Optional[int] = None) -> Union[List[List[int]], List[int]]:
        if isinstance(text, str):
            text = [text]

        tokenized_text = []
        for sentence in text:
            symbols = sentence.split()
            tokens = []
            for symbol in symbols:
                if symbol in self.vocab:
                    tokens.append(self.vocab[symbol])
                else:
                    tokens.append(self.vocab['<unk>'])
            tokenized_text.append(tokens)

        if padding:
            max_len = max(len(tokens) for tokens in tokenized_text)
            tokenized_text = [tokens + [self.vocab['<pad>']] *
                              (max_len - len(tokens)) for tokens in tokenized_text]

        if max_length:
            tokenized_text = [tokens[:max_length] for tokens in tokenized_text]

        return tokenized_text


class WordTokenizer(BaseTokenizer):
    def add_corpus(self, corpus: Union[List[str], str]) -> None:
        if isinstance(corpus, str):
            corpus = [corpus]
        for sentence in corpus:
            symbols = sentence.split()
            self.vocab.update(symbols)

    def tokenize(self, text: Union[List[str], str], padding: bool = False, max_length: Optional[int] = None) -> Union[List[List[int]], List[int]]:
        if isinstance(text, str):
            text = [text]

        tokenized_text = []
        for sentence in text:
            symbols = sentence.split()
            tokens = [symbol for symbol in symbols if symbol in self.vocab]
            tokenized_text.append(tokens)

        if padding:
            max_len = max(len(tokens) for tokens in tokenized_text)
            tokenized_text = [tokens + ['<pad>'] *
                              (max_len - len(tokens)) for tokens in tokenized_text]

        if max_length:
            tokenized_text = [tokens[:max_length] for tokens in tokenized_text]

        return tokenized_text
