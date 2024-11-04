import spacy
from collections import Counter

class spacy_tokenizer():
    def __init__(self):
        self.spacy_eng = spacy.load("en_core_web_sm")

    def __call__(self, text):
        return [tok.text.lower() for tok in self.spacy_eng.tokenizer(text)]
    

class Vocabulary:
    def __init__(self, callable_tokenizer=None, max_freq=3, unk=True, sos=False, eos=False):

        self.sos = sos
        self.eos = eos
        self.unk = unk
        if callable_tokenizer:
            self.callable_tokenizer = callable_tokenizer
        else:
            self.callable_tokenizer = spacy_tokenizer()

        self.stoi = {"<PAD>": 0}
        if self.unk:
            self.stoi['<UNK>'] = len(self.stoi)
        if self.sos:
            self.stoi['<SOS>'] = len(self.stoi)
        if self.eos:
            self.stoi['<EOS>'] = len(self.stoi)
        
    def __len__(self):
        return len(self.stoi)

    def get_vocabulary(self):
        return self.stoi

    def add_token(self, token_name: str):
        if token_name not in self.stoi:
            self.stoi[token_name] = len(self.stoi)

    def build_vocabulary(self, sentences_list):
        if type(sentences_list[0]) != str:
            ## ex: [['eating', 'apples'], ['eating', 'oranges']]
            sentences_list = [' '.join(sen) for sen in sentences_list]

        word_counts = Counter()
        for sentence in sentences_list:
            tokens = self.callable_tokenizer(sentence)
            word_counts.update(tokens)

        # Filter words with mox_freq or more occurrences
        filtered_words = [word for word, count in word_counts.items() if count >= 3]
        for word in filtered_words:
            if word not in self.stoi:
                self.stoi[word] = len(self.stoi)

    def get_numerical_tokens(self, text: str):
        tokens = self.callable_tokenizer(text)
        # tokens.insert(0, '<SOS>') if self.sos else None
        # tokens.append('<EOS>') if self.eos else None
        unk_id = self.stoi.get('<UNK>', None)
        return [self.stoi.get(word, unk_id) for word in tokens]

    def __call__(self, text: str):
        return self.get_numerical_tokens(text)

    def tokens_to_text(self, tokens_list):
        keys = list(self.stoi.keys())
        values = list(self.stoi.values())

        return ' '.join([keys[values.index(token)] for token in tokens_list])