import dawg
import Levenshtein
from operator import itemgetter
import copy
import pdb

'''
DeleteMatchPrefix
Algorithm for fuzzy prefix matching, particularly efficient with short words.

From Marjan Celikik's dissertation (2013)
Albert-Ludwigs-Universität Freiburg im Breisgau
Institut für Informatik

Implemented by
Trevor Sullivan (2016)
for VirtualWorks in the commission of the University of Arizona HLT internship

usage: instantiate DeleteMatch. The constructor takes as an argument a
    dictionary (default W), a distance tolerance, and a prefix cutoff.
DeleteMatch will automatically make an index. Search the index with the
    .search() method.

Changes from whole word method:
In addition to the k-prefixes, index all the i-prefixes for i = m ... k, where
    m is the minimum prefix length
Use Prefix levenshtein distance instead of Word levenshtein distance to verify
    results

'''

# dictionary for testing
W = ['an', 'bat', 'batter', 'batty', 'bitter', 'but', 'butter', 'buttock',
     'jelly', 'nut', 'pea', 'peanut', 'thyme', 'time', 'tome']
test_weights = [1.08, 1.07, 1.06, 1.05, 1.04, 1.03, 1.02, 1.01,
                1.07, 1.06, 1.05, 1.04, 1.03, 1.02, 1.01]


class SearchIndex:
    __instance = None

    @staticmethod
    def get_instance(dictionary=W, weights= test_weights, delta=1, k=4, minimum_prefix=2):
        if SearchIndex.__instance is None:
            SearchIndex(dictionary, weights, delta, k, minimum_prefix)
        return SearchIndex.__instance


    def __init__(self, dictionary = W, weights = test_weights, delta = 1, k = 4,
                 minimum_prefix = None, max_returns = 20, verbose=False):
        ''' This is a private constructor, do not call it '''
        if SearchIndex.__instance is not None:
            raise Exception("Attempted to construct a duplicate singleton!")
        else:
            SearchIndex.__instance = self

        '''
        Weighted version of the truncated deletion neighborhood method described
             by Celikik

        :dictionary: the list of entries/phrases to be indexed for searching
        :weights: the list of weights associated with the words in the
                dictionary. Should have same length.
        :delta:  the tolerance for errors. default: 1
        :k: truncation prefix length. experiment this to see what works best for your application. Should be siginificantly
            shorter than the average word length default: 4
        :minimum_prefix: the smallest searchable prefix. default: 2
        :max returns: the largest number of desired return values. Necessary because short prefixes will take a very long time to complete with
            a large dictionary. Default: 20

        '''
        if minimum_prefix is None:
            self.minimum_prefix = 2 + delta
        else:
            self.minimum_prefix = minimum_prefix

        #instance variables:
        self.verbose = verbose
        self.index = None

        #this dictionary takes the form of a trie
        #   keys: strings from n-deletion-neighborhood
        #   values: tuples of the form (rangebegin, rangesize)
        #it can also be compressed (dissertation p 27)
        self.ranges = dict()
        #this dictionary takes the form {rangeid:[start_of_range,size_of_range]}
        #in production, the range set should be encoded as a hash table. I think.
        #TODO: assess if this is even necessary after the trie implementation
        dictionary, self.weights = self.sort_dict(dictionary, weights)
        self.dictionary = []
        self.lcp = []
        #set the input data to lower case and then put together the lcp(longest common prefix) array
        for i in range(0, len(dictionary)):
            self.dictionary.append(dictionary[i].lower())
            self.lcp.append(self.get_lcp(dictionary[i], dictionary[i+1]) if i < len(dictionary) - 1 else 0)
        self.delta = delta
        self.k = k
        self.max_returns = max_returns
        self.createIndex()



    def createIndex(self):
        '''
        Truncated Deletion Neighborhoods.
        Like the deletion neighborhood technique above, except each dictionary entry is truncated to k characters
        Every entry in the index therefore represents a range of dictionary words with that prefix.
        k = 7 and delta = 3 gets a index size roughly the same size as the dictionary widh identical running time to full-size deletion neighborhood.
        k = 6 and delta = 3 gets an index size less than the size of dictionary, at the cost of apprx double runtime
        Investigate: using different Ks for different words

        Changes from full word algorithm: make a trie instead. to search for substrings of length i, we do a traversal of the last trie node
            (the trie tool handles this automatically)
        '''
        prefixes = self.getRanges() #takes the form {prefix:rangeid}

        #each range is an array of length 2, [index, size]
        #turn the arrays into tuples, and use them as the values for the trie
        #first: get all of the deletion neighborhood for the keys, arrange into two arrays for feeding into the trie

        keys = []
        values = []
        fmt = "<HH" #little-endian two unsigned short tuple Max number and size of ranges, 65000
        #if that's not enough, use I for unsigned int

        i = 0
        for prefix in prefixes.keys():
            substringset = n_deletion_neighborhood(prefix, self.delta) #get deletion neighborhood
            for sub in substringset:
                #lemma 3.2.5 in order to index we only need sequences of length k-ð
                #TODO: decide whether or not to make this line != or >
                #   > should result in a bigger index but somehow doesn't? I'm confused.
                #   != causes loss of recall for words shorter than k-ð
                if len(sub) > self.k-self.delta or len(sub) < self.minimum_prefix:
                    continue
                keys.append(sub)
                values.append(tuple(self.ranges[prefixes[prefix]]))
                i += 1

        try:
            self.index = dawg.RecordDAWG(fmt, zip(keys, values))
        except:
            if self.verbose:
                print("Big dictionary, trying to index with 4 bytes rather than 2")
            self.index = dawg.RecordDAWG("<II", zip(keys, values))
        del prefixes



    def getRanges(self):
        '''
        populates the ranges dictionary based on the dictionary and cutoff
        returns a dictionary of prefixes and the associated rangeid
        '''
        prefixes = dict()

        self.dictionary.sort()

        rangeid = 0
        for i in range(0, len(self.dictionary)):
            prefix = self.dictionary[i][:self.k]
            if prefix not in prefixes.keys(): #if the prefix isn't already associated with a range, make a new one
                prefixes[prefix] = rangeid
                self.ranges[rangeid] = [i, 1]
                rangeid += 1
            else:
                self.ranges[prefixes[prefix]][1] += 1 #if it is, increase the range size

        return prefixes



    def search(self, query):
        '''
        TODO: There is an error that occurs here when k=delta+3
        '''
        if len(query) < self.minimum_prefix:
            return []
        query = query.lower()
        candidate_ids_short = []
        candidate_ids = []
        substringset = n_deletion_neighborhood(query[:self.k], self.delta)

        #intersect = set(self.index.keys()).intersection(substringset)
        #rather than set intersection of the full keyset, we want to search for common prefixes,
        #but only common prefixes of sufficient length, lenq-ð
        intersect = set()
        # pdb.set_trace()
        for sub in substringset:
            # if len(sub) < len(query) - self.delta:
            #   continue
            # Not sure why this was here, but it was breaking searches with long queries. ndelnei already makes sure they aren't too short
            intersect = intersect.union(set(self.index.keys(sub)))


        if len(intersect) != 0:
            duplicate = []
            for match in intersect:
                for rangeid in self.index[match]:
                    if rangeid[0] in duplicate:
                        continue
                    for i in range(rangeid[0], rangeid[0]+rangeid[1]): #for each index in the dictionary range
                        if len(candidate_ids_short) < self.max_returns:
                            #if the return list isn't full yet, just add to it
                            candidate_ids_short.append([i, self.weights[i]])
                        else:
                            if self.weights[i] > candidate_ids_short[0][1]:
                                #if the present word is higher weighted than the lowest one in the list, pop the lowest and add it
                                candidate_ids_short.pop(0)
                                candidate_ids_short.append([i, self.weights[i]])
                        candidate_ids_short = sorted(candidate_ids_short, key=itemgetter(1))


                        #if pld(self.dictionary[i], query) <= self.delta: # uncomment when not using suffix filtering
                        #   candidate_ids.append(i)
                        candidate_ids.append(i)
                    duplicate.append(rangeid[0]) #don't look at this range again

        # pdb.set_trace()

        if len(query) > self.k:
            # use big list and do suffix filter for long queries
            candidates = self.prefixFilter(candidate_ids, query[:self.k])
            candidates = self.suffixFilter(candidates, query)
        else:
            # use short list and don't suffix filter for short queries
            candidates = self.prefixFilter([row[0] for row in candidate_ids_short], query[:self.k])

        # sort based on
        candidates = sorted(candidates, key=itemgetter(1))
        final_out = candidates[-(self.max_returns):]

        final_out.reverse()

        return final_out



    def prefixFilter(self, candidate_ids, query):
        #If the longest common prefix between a candidate word and the query is greater than
        #   the length of the query, you can automatically approve it

        output = []
        already_tested = [] #list of indices already filtered
        l = len(query)
        candidate_ids = sorted(candidate_ids)
        for i in candidate_ids:
            if i in already_tested:
                continue
            already_tested.append(i)
            if pld(self.dictionary[i], query) > self.delta:
                continue
            else:
                output.append([self.dictionary[i], self.weights[i]])
            while self.lcp[i] >= l:
                #as long as the longest common prefix is longer than the query, it's still a match
                output.append([self.dictionary[i+1], self.weights[i]])
                already_tested.append(i+1)
                i += 1


        return output





    def suffixFilter(self, prefix_matches, query):
        '''
        adapted from pseudocode on page 29 of M. Celikik's dissertation
        this will filter out words that aren't within the levenshtein distance post-prefix without having to compute full word levenshtein distance
        '''

        freqs = self.getFreqs(query)

        saved_freqs = copy.deepcopy(freqs)

        # pdb.set_trace()

        output = []
        for j in range(len(prefix_matches)):
            word = prefix_matches[j][0]
            # pdb.set_trace()
            count = self.k
            i = 0
            for i in range(self.k, min(len(word), len(query)+self.delta)):
                if word[i] in freqs.keys():
                    if freqs[word[i]] > 0:
                        count += 1
                    freqs[word[i]] -= 1

                if count < i - self.delta:
                    #WLD(w,q) > ð
                    break
            if count < len(query) - self.delta:
                #WLD(w,q) > ð
                pass
            else:
                if pld(word, query) <= self.delta:
                    # the filter removes the easy ones, where the length or unigram frequency is radically different, but only on the suffix
                    # we still need to use levenshtein to check for correctness.
                    output.append([word, prefix_matches[j][1]])


            # this occasionally misbehaves
            # for j in range(self.k, i): #reset frequency vector
            #   if word[j] in freqs.keys():
            #       freqs[word[j]] += 1
            freqs = copy.deepcopy(saved_freqs)



        return output




    def getFreqs(self, query):
        '''
        returns how often each character appears (used for verification/filtering)
        '''
        freqs = dict()

        for char in query:
            if char in freqs.keys():
                freqs[char] += 1
            else:
                freqs[char] = 1

        return freqs


    def get_lcp(self, rot1, rot2):
        '''
        returns the length of the longest common prefix
        '''
        output = 0
        for i in range(0, min(len(rot1), len(rot2))):
            if rot1[i] != rot2[i]:
                return output
            else:
                output += 1
        return output


    def sort_dict(self, dictionary, weights):
        '''
        sorting alphebetically for weighted version
        '''
        temp_list = [ [x,y] for x, y in zip(dictionary, weights)]

        temp_list = sorted(temp_list, key=itemgetter(0))

        return [row[0] for row in temp_list], [row[1] for row in temp_list]

    def __str__(self):
        return "Weighted DeleteMatchPrefix Index: 𝛿={:n}, k={:n}, ".format(self.delta, self.k)


    def __repr__(self):
        return str(self)








def levenshtein(s1, s2):
    '''
    from wikipedia
    '''
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            # j+1 instead of j since previous_row and current_row are
            #   one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

    return previous_row[-1]


def lev(a, b):
    # cheating
    return Levenshtein.distance(a, b)
    '''
    also from wikipedia, because the above didn't work for some reason.
    this one's allegedly slower
    '''

    if not a:
        return len(b)
    if not b:
        return len(a)
    return min(lev(a[1:], b[1:])+(a[0] != b[0]), lev(a[1:], b)+1,
               lev(a, b[1:])+1)


def pld(a, b):
    if len(a) == len(b):
        return lev(a, b)

    smaller_one = a if min(len(a), len(b)) == len(a) else b
    bigger_one = a if max(len(a), len(b)) == len(a) else b
    distances = []
    for i in range(0, len(bigger_one)):
        distances.append(lev(smaller_one, bigger_one[:i]))
    return min(distances)
    # calculate minimum levenshtein distance on the prefixes


def subsequence(word, positions):
    '''
    deletes the characters in positions from the word
    word = string to be subsequenced
    positions = integer array of positions to be removed
    '''
    for pos in positions[::-1]:
        word = word[:pos] + word[pos+1:]

    return word


def n_deletion_neighborhood(word, delta):
    '''
    Ud(w,delta) = w if delta=0
              U(0->w) {Ud(s(w, i), delta - 1)}
    returns a set containing the union of all possible subsequences of
        word with delta deletions
    word = string in question
    delta = maximum number of deletions
    '''
    output = set()

    if delta == 0:
        output.add(word)
        return output

    for i in range(0, len(word)+1):
        # Union(0->w) {Ud(s(w, i), delta - 1)}
        output = output.union(n_deletion_neighborhood(subsequence(word, [i]),
                                                      delta-1))

    return output


def n_deletion_neighborhood_all_sizes(word, delta):
    '''
    same as above, except returns all subsequences with 0...delta deletions
    '''
    output = set()

    for i in range(0, delta+1):
        output = output.union(n_deletion_neighborhood(word, i))

    return output


def DNindex(dictionary=W, delta=1):
    '''
     Deletion Neighborhoods.
    Rather than store all the information about the dictionary word in
        the index, store only the substring and the word id
    iterate over W and generate a set of substrings for each w.
        add the substrings to the index dict with the wordid(s) as the data.
    '''
    index = dict()

    for i in range(0, len(dictionary)):
        substringset = n_deletion_neighborhood(dictionary[i], delta)
        for sub in substringset:
            if sub in index.keys():
                index[sub].append(i)
            else:
                index[sub] = [i]

    return index


def DNsearch(query, index, dictionary=W, delta=1):
    output = []
    substringset = n_deletion_neighborhood(query, delta)

    if len(set(index.keys()).intersection(substringset)) != 0:
        duplicate = []
        for match in set(index.keys()).intersection(substringset):
            for i in index[match]:
                if i in duplicate:
                    continue
                if lev(dictionary[i], query) <= delta:
                    output.append(dictionary[i])
                duplicate.append(i)  # don't look at this word again

    return output


class SearchIndexUnweighted:
    __instance = None

    @staticmethod
    def get_instance(dictionary=W, delta=1, k=4, minimum_prefix=2):
        if SearchIndexUnweighted.__instance is None:
            SearchIndexUnweighted(dictionary, delta, k, minimum_prefix)
        return SearchIndexUnweighted.__instance

    def __init__(self, dictionary=W, delta=1, k=4, minimum_prefix=2):
        ''' This is a private constructor, do not call it '''
        if SearchIndexUnweighted.__instance is not None:
            raise Exception("Attempted to construct a duplicate singleton!")
        else:
            SearchIndexUnweighted.__instance = self
        # instance variables:
        self.index = None
        # this dictionary takes the form of a trie
        #   keys: strings from n-deletion-neighborhood
        #   values: tuples of the form (rangebegin, rangesize)
        # it can also be compressed (dissertation p 27)
        self.ranges = dict()
        # this dictionary takes the form
        #   {rangeid:[start_of_range,size_of_range]}
        # in production, the range set should be encoded as a hash table.
        #   I think.
        # TODO: assess if this is even necessary after the trie implementation
        self.minimum_prefix = minimum_prefix
        dictionary = sorted(dictionary)
        self.dictionary = []
        self.lcp = []
        for i in range(0, len(dictionary)):
            self.dictionary.append(dictionary[i].lower())
            self.lcp.append(self.get_lcp(dictionary[i], dictionary[i+1])
                            if i < len(dictionary) - 1 else 0)

        self.delta = delta
        self.k = k
        self.createIndex()

    def createIndex(self):
        '''
        Truncated Deletion Neighborhoods.
        Like the deletion neighborhood technique above, except each dictionary
             entry is truncated to k characters
        Every entry in the index therefore represents a range of dictionary
            words with that prefix.
        k = 7 and delta = 3 gets a index size roughly the same size as the
            dictionary widh identical running time to full-size deletion
            neighborhood.
        k = 6 and delta = 3 gets an index size less than the size of dictionary
            at the cost of apprx double runtime
        Investigate: using different Ks for different words

        Changes from full word algorithm: make a trie instead. to search for
            substrings of length i, we do a traversal of the last trie node
            (the trie tool handles this automatically)
        '''
        prefixes = self.getRanges()  # takes the form {prefix:rangeid}

        # each range is an array of length 2, [index, size]
        # turn the arrays into tuples, and use them as the values for the trie
        # first: get all of the deletion neighborhood for the keys, arrange
        #    into two arrays for feeding into the trie

        keys = []
        values = []
        fmt = "<HH"
        # little-endian two unsigned short tuple Max number and
        # size of ranges, 65000
        # if that's not enough, use I for unsigned int

        i = 0
        for prefix in prefixes.keys():
            substringset = n_deletion_neighborhood(prefix, self.delta)
            # get deletion neighborhood
            for sub in substringset:
                # lemma 3.2.5 in order to index we only need sequences of
                #   length k-ð
                # TODO: decide whether or not to make this line != or >
                #   > should result in a bigger index but somehow doesn't?
                #       I'm confused.
                #   != causes loss of recall for words shorter than k-ð
                if len(sub) > self.k-self.delta or (len(sub) <
                                                    self.minimum_prefix):
                    continue
                keys.append(sub)
                values.append(tuple(self.ranges[prefixes[prefix]]))
                i += 1
        try:
            self.index = dawg.RecordDAWG(fmt, zip(keys, values))
        except:
            print("Big dictionary, trying to index with 4 bytes rather than 2")
            self.index = dawg.RecordDAWG("<II", zip(keys, values))

        del prefixes

    def getRanges(self):
        '''
        populates the ranges dictionary based on the dictionary and cutoff
        returns a dictionary of prefixes and the associated rangeid
        '''
        prefixes = dict()

        self.dictionary.sort()

        rangeid = 0
        for i in range(0, len(self.dictionary)):
            prefix = self.dictionary[i][:self.k]
            if prefix not in prefixes.keys():
                # if the prefix isn't already associated with a range,
                # make a new one
                prefixes[prefix] = rangeid
                self.ranges[rangeid] = [i, 1]
                rangeid += 1
            else:
                self.ranges[prefixes[prefix]][1] += 1
                # if it is, increase the range size

        return prefixes

    def search(self, query):
        query = query.lower()
        candidate_ids = []
        substringset = n_deletion_neighborhood(query[:self.k], self.delta)

        # intersect = set(self.index.keys()).intersection(substringset)
        # rather than set intersection of the full keyset, we want to search
        #  for common prefixes,
        # but only common prefixes of sufficient length, lenq-ð
        intersect = set()
        for sub in substringset:
            if len(sub) < len(query) - self.delta:

                continue
            intersect = intersect.union(set(self.index.keys(sub)))

        # pdb.set_trace()
        if len(intersect) != 0:
            duplicate = []
            for match in intersect:
                for rangeid in self.index[match]:
                    if rangeid[0] in duplicate:
                        continue
                    for i in range(rangeid[0], rangeid[0]+rangeid[1]): #for each index in the dictionary range
                        #if pld(self.dictionary[i], query) <= self.delta: # uncomment when not using suffix filtering
                        #   candidate_ids.append(i)
                        candidate_ids.append(i)
                    duplicate.append(rangeid[0])
                    # don't look at this range again

        candidates = self.prefixFilter(candidate_ids, query[:self.k])

        if len(query) >= self.k:
            return self.suffixFilter(candidates, query)
        else:
            return candidates



    def prefixFilter(self, candidate_ids, query):
        #If the longest common prefix between a candidate word and the query is greater than
        #   the length of the query, you can automatically approve it

        output = []
        already_tested = [] #list of indices already filtered
        l = len(query)
        candidate_ids = sorted(candidate_ids)
        for i in candidate_ids:
            if i in already_tested:
                continue
            already_tested.append(i)
            # if pld(self.dictionary[i], query) > self.delta:
            #   continue
            # else:
            #   output.append(self.dictionary[i])
            while self.lcp[i] >= l:
                #as long as the longest common prefix is longer than the query, it's still a match
                output.append(self.dictionary[i+1])
                already_tested.append(i+1)

                i += 1

        return output





    def suffixFilter(self, prefix_matches, query):
        '''
        adapted from pseudocode on page 29 of M. Celikik's dissertation
        this will filter out words that aren't within the levenshtein distance post-prefix without having to compute full word levenshtein distance
        '''

        freqs = self.getFreqs(query)

        output = []
        for word in prefix_matches:
            # pdb.set_trace()
            count = self.k
            i = 0
            for i in range(self.k, len(word)):
                if word[i] in freqs.keys():
                    if freqs[word[i]] > 0:
                        count += 1
                    freqs[word[i]] -= 1

                if count < i - self.delta:
                    #WLD(w,q) > ð
                    break
            if count < len(query) - self.delta:
                #WLD(w,q) > ð
                pass
            else:
                if pld(word, query) <= self.delta:  # the filter removes the easy ones, where the length or unigram frequency is radically different, but only on the suffix
                    output.append(word)                         # we still need to use levenshtein to check for correctness.


            for j in range(self.k, i): #reset frequency vector
                if word[j] in freqs.keys():
                    freqs[word[j]] += 1


        return output




    def getFreqs(self, query):
        freqs = dict()

        for char in query:
            if char in freqs.keys():
                freqs[char] += 1
            else:
                freqs[char] = 1

        return freqs


    def get_lcp(self, rot1, rot2):
        output = 0
        for i in range(0, min(len(rot1), len(rot2))):
            if rot1[i] != rot2[i]:
                return output
            else:
                output += 1
        return output

    def __str__(self):
        return "Unweighted DeleteMatchPrefix Index: 𝛿={:n}, k={:n}, ".format(self.delta, self.k)


    def __repr__(self):
        return str(self)

