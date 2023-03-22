import numpy as np
import random
import six


class OneHotEncoder():
    """OneHotEncoder converts the raw DNA sequence to
       onehot embedding of the DNA sequence.
       It also padding the DNA sequence on both side
       to create the embedding of the same length.
        Args:
            dictionary: Onehot encoding formatter. Defaults to {'A':0,'C':1,'G':2,'T':3,'N':[0.]*4}.
            max_length: The max length of the input DNA sequence.
                        Search through the Input DNA sequence
                        and set it to the max DNA sequence in the input
            u_to_t: _description_. Defaults to True.
            pad: _description_. Defaults to 'both'.
            dim: the Dimension of the embedding space for each nucleic acid.
                 Defaults to None.
    """
    def __init__(self,dictionary={'A':0,'C':1,'G':2,'T':3,'N':[0.]*4}, max_length=None,u_to_t=True,pad='both',dim=4,k=4):
        self.dim=dim
        self.dictionary=dictionary
        self.max_length=max_length
        self.u_to_t=u_to_t
        self.pad=pad
        self.k = k
        self.kmers = []
        self.kmer_length = 0
        if self.k > 0:
            self.generate_kmers_up2k()
    def generate_kmers_up2k(self):
        """_summary_: generate all the possible kmers from [1,k]
        """
        nucleotides = ['A', 'C', 'G', 'T']
        def generate_combinations(length, prefix):
            if length == 0:
                self.kmers.append(prefix)
            else:
                for nucleotide in nucleotides:
                    new_prefix = prefix + nucleotide
                    generate_combinations(length-1, new_prefix)
        for length in range(1, self.k+1):
            generate_combinations(length, "")
        self.kmer_length = len(self.kmers)

    def encode(self,sequences):
        if isinstance(sequences,(six.string_types,bytes)):
            sequences = [sequences,]
        elif not isinstance(sequences,(list,tuple,np.ndarray)):
            sequences = list(sequences)
        for i in range(len(sequences)):
            sequences[i] = sequences[i].upper()[:self.max_length]
        if self.u_to_t:
            sequences = [s.replace('U','T') for s in sequences]
        if not self.dictionary:
            self.infer_dictionary(sequences)

        shape = [len(sequences),self.max_length,self.dim or len(self.dictionary)]
        onehot = np.zeros(shape,dtype=np.float16)
        for i, s in enumerate(sequences):
            offset = self.offset(s,self.max_length)
            for j, el in enumerate(s):
                onehot[i,j+offset] = self.dictionary[el]
        if len(sequences)==1:
            onehot = np.squeeze(onehot,axis=0)
        return onehot

    def offset(self,sequence,length):
        if self.pad=='right':
            return 0
        elif self.pad=='left':
            return length-len(sequence)
        elif self.pad=='both':
            return (length-len(sequence))//2

    def kmer_encoder(self, sequences):
        def count_kmers(seq, max_k):
            """Count the number of overlapping k-mers in a DNA sequence."""
            kmers = {}
            for k in range(1, max_k+1):
                for i in range(len(seq) - k + 1):
                    kmer = seq[i:i+k]
                    if kmer in kmers:
                        kmers[kmer] += 1
                    else:
                        kmers[kmer] = 1
            return kmers
        kmers_dict = count_kmers(sequences, self.k)
        emb = np.zeros(self.kmer_length, dtype=np.float16)
        for i, ele in enumerate(self.kmers):
            if ele in kmers_dict:
                emb[i] = kmers_dict[ele]
        return emb

    def infer_dictionary(self,sequences):
        self.dictionary = set(''.join(sequences))

    @property
    def dictionary(self):
        return self._dictionary

    @dictionary.setter
    def dictionary(self,dictionary):
        if isinstance(dictionary,dict):
            new = {}
            for k, v in dictionary.items():
                if isinstance(v,int):
                    el = np.zeros(self.dim or len(dictionary),dtype=np.float16)
                    el[v] = 1
                    new[k]=el
                else:
                    new[k] = np.asarray(v,dtype=np.float16)
            self._dictionary = new
        elif isinstance(dictionary,Iterable):
            dictionary = list(dictionary)
            dictionary.sort()
            new = {}
            for i,k in enumerate(dictionary):
                el = np.zeros(self.dim or len(dictionary),dtype=np.float16)
                el[i] = 1
                new[k]=el
            self._dictionary = new
        else:
            raise TypeError("dictionary must be dict or iterable, got {}".format(type(dictionary)))

    def generateRamdomDNA(self):
        seq = ['A', 'T', 'G', 'C']
        return "".join([seq[random.randint(0, 3)] for i in range(self.max_length)])
