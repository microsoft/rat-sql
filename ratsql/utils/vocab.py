import collections
import collections.abc
import json
import operator


class Sentinel(object):
    '''Used to represent special values like UNK.'''
    # pylint: disable=too-few-public-methods
    __slots__ = ('name', )

    def __init__(self, name):
        # type: (str) -> None
        self.name = name

    def __repr__(self):
        # type: () -> str
        return '<' + self.name + '>'

    def __lt__(self, other):
        # type: (object) -> bool
        if isinstance(other, IndexedSet.Sentinel):
            return self.name < other.name
        return True


UNK = '<UNK>'
BOS = '<BOS>'
EOS = '<EOS>'


class Vocab(collections.abc.Set):

    def __init__(self, iterable, special_elems=(UNK, BOS, EOS)):
        # type: (Iterable[T]) -> None
        elements = list(special_elems)
        elements.extend(iterable)
        assert len(elements) == len(set(elements))

        self.id_to_elem = {i: elem for i, elem in enumerate(elements)}
        self.elem_to_id = {elem: i for i, elem in enumerate(elements)}

    def __iter__(self):
        # type: () -> Iterator[Union[T, IndexedSet.Sentinel]]
        for i in range(len(self)):
            yield self.id_to_elem[i]

    def __contains__(self, value):
        # type: (object) -> bool
        return value in self.elem_to_id

    def __len__(self):
        # type: () -> int
        return len(self.elem_to_id)

    def __getitem__(self, key):
        # type: (int) -> Union[T, IndexedSet.Sentinel]
        if isinstance(key, slice):
            raise TypeError('Slices not supported.')
        return self.id_to_elem[key]

    def index(self, value):
        # type: (T) -> int
        try:
            return self.elem_to_id[value]
        except KeyError:
            return self.elem_to_id[UNK]

    def indices(self, values):
        # type: (Iterable[T]) -> List[int]
        return [self.index(value) for value in values]

    def __hash__(self):
        # type: () -> int
        return id(self)

    @classmethod
    def load(self, in_path):
        return Vocab(json.load(open(in_path)), special_elems=())

    def save(self, out_path):
        with open(out_path, 'w') as f:
            json.dump([self.id_to_elem[i] for i in range(len(self.id_to_elem))],  f)


class VocabBuilder:
    def __init__(self, min_freq=None, max_count=None):
        self.word_freq = collections.Counter()
        self.min_freq = min_freq
        self.max_count = max_count

    def add_word(self, word, count=1):
        self.word_freq[word] += count

    def finish(self, *args, **kwargs):
        # Select the `max_count` most frequent words. If `max_count` is None, then choose all of the words.
        eligible_words_and_freqs = self.word_freq.most_common(self.max_count)
        if self.min_freq is not None:
            for i, (word, freq) in enumerate(eligible_words_and_freqs):
                if freq < self.min_freq:
                    eligible_words_and_freqs = eligible_words_and_freqs[:i]
                    break

        return Vocab(
                    (word for word, freq in sorted(eligible_words_and_freqs)), 
                    *args, **kwargs)
    
    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.word_freq, f)
    
    def load(self, path):
        with open(path, "r") as f:
            self.word_freq = collections.Counter(json.load(f))
