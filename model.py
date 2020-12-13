import numpy as np
import math
import argparse

def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums) == np.shape(row_sums)[0])  # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix


class Corpus(object):
    """
    A collection of documents.
    """
    document_theme_prob: None
    topic_prob: None

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None
        self.document_theme_prob = None  # P(z | d)
        self.theme_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)
        self.background_model = None #P(w | B)
        self.cluster_theme_word_prob = None # P(w | z (i,j) )
        self.bg_prob = None # P(d, Ci, w = B)
        self.common_topic_prob = None #P(d, Ci, z, w = C)
        self.collection_set = None #collection of each document [0, 0, 1, 1, 2, ... ]

        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]

        Update self.number_of_documents
        """

        # opening the text file
        ret = []
        collection = []
        with open(self.documents_path, 'r') as file:

            # reading each line
            i = 0
            for line in file:
                new_doc = []
                # reading each word
                doc = line.split()
                collection.append(int(doc[0]))
                for k in range(1, len(doc)):
                    new_doc.append(doc[k])
                ret.append(new_doc)
                i += 1

        print(collection)

        self.documents = ret
        self.number_of_documents = len(self.documents)
        print(self.number_of_documents)
        # !!!!!!change if ci is not certain
        self.collection_set = collection

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus: ["rain", "the", ...]

        Update self.vocabulary_size
        """

        voc = set()

        for doc in self.documents:
            for word in doc:
                if word not in voc:
                    voc.add(word)

        #create a list of vocabulary words and sort the list
        voc_lst = list(voc)
        voc_lst.sort()

        self.vocabulary = voc_lst
        self.vocabulary_size = len(voc_lst)

    def build_background_model(self):
        '''
        construct the background model P(w|B), put in self.background
        :return:
        '''

        ret = [0 for i in range(self.vocabulary_size)]
        d = {}
        for i in range(self.vocabulary_size):
            d[self.vocabulary[i]] = i

        ct = 0
        for i in range(self.number_of_documents):
            for j in range(len(self.documents[i])):
                cur_w = self.documents[i][j]
                w_idx = d[cur_w]
                ret[w_idx] += 1
                ct += 1

        ret = np.array(ret) / ct
        self.background_model = ret


    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document,
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        doc_num = self.number_of_documents
        voc_size = self.vocabulary_size
        d = {}
        for i in range(voc_size):
            d[self.vocabulary[i]] = i

        mat = [[0 for i in range(voc_size)] for j in range(doc_num)]
        for i in range(doc_num):
            for j in range(len(self.documents[i])):
                cur_w = self.documents[i][j]
                w_idx = d[cur_w]
                mat[i][w_idx] += 1

        self.term_doc_matrix = mat

    def initialize_randomly(self, number_of_topics, number_of_clusters):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob
        """

        mat = [[np.random.random_sample() for i in range(number_of_topics)] for j in range(self.number_of_documents)]

        mat2 = [[np.random.random_sample() for i in range(self.vocabulary_size)] for j in range(number_of_topics)]

        mat3 = [[[np.random.random_sample() for i in range(self.vocabulary_size)] for j in range(number_of_topics)] for k in range(number_of_clusters)]

        self.document_theme_prob = normalize(np.array(mat))
        self.theme_word_prob = normalize(np.array(mat2))
        mat3 = np.array(mat3)
        for k in range(len(mat3)):
            mat3[k] = normalize(mat3[k])

        self.cluster_theme_word_prob = mat3

        '''
        print('doc-theme-prob')
        print(self.document_theme_prob)
        print('theme-word-prob')
        print(self.theme_word_prob)
        print('cluster-theme-word-prob')
        print(self.cluster_theme_word_prob)
        '''




    def initialize_uniformly(self, number_of_topics, number_of_clusters):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, self.vocabulary_size))
        self.topic_word_prob = normalize(self.topic_word_prob)

        self.cluster_theme_word_prob = np.ones((number_of_clusters, number_of_topics, self.vocabulary_size))
        self.cluster_theme_word_prob = normalize(self.cluster_theme_word_prob)


    def initialize(self, number_of_topics, number_of_clusters, random=True):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics, number_of_clusters)
        else:
            self.initialize_uniformly(number_of_topics, number_of_clusters)

    def expectation_step(self, number_of_topics,number_of_clusters,lambda_b, lambda_c):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")

        #expect to have a (ci, d, w) 3d array for demonimator of P(d, ci, w = j)
        denominators = [np.dot(self.document_theme_prob, lambda_c * self.theme_word_prob + (1 - lambda_c) * self.cluster_theme_word_prob[i]) for i in range(number_of_clusters)]
        denominators = np.array(denominators)

        for d in range(self.number_of_documents):
            for t in range(number_of_topics):
                for w in range(self.vocabulary_size):
                    i = self.collection_set[d]
                    denominator = denominators[i][d][w]
                    result = self.document_theme_prob[d][t] * (
                                self.theme_word_prob[t][w] * lambda_c + self.cluster_theme_word_prob[i][t][w] * (
                                    1 - lambda_c)) / denominator
                    # print('P(d ci w = j)',self.topic_prob[d][t][w])
                    self.topic_prob[d][t][w] = result

        for d in range(self.number_of_documents):
            for w in range(self.vocabulary_size):
                i = self.collection_set[d]
                denominator2 = lambda_b * self.background_model[w] + (1 - lambda_b) * denominators[i][d][w]
                self.bg_prob[d][w] = lambda_b * self.background_model[w] / denominator2

        for d in range(self.number_of_documents):
            for t in range(number_of_topics):
                for w in range(self.vocabulary_size):
                    i = self.collection_set[d]
                    denominator3 = lambda_c * self.theme_word_prob[t][w] + (1 - lambda_c) * \
                                   self.cluster_theme_word_prob[i][t][w]
                    result = lambda_c * self.theme_word_prob[t][w] / denominator3
                    self.common_topic_prob[d][t][w] = result

        '''
        print('-----------topic prob------------')
        print(self.topic_prob)
        print('-----------bg prob------------')
        print(self.bg_prob)
        print('-----------common topic prob------------')
        print(self.common_topic_prob)
        '''



    def maximization_step(self, number_of_topics,number_of_clusters):
        """ The M-step updates P(w | z)
        """
        print("M step:")


        for d in range(self.number_of_documents):
            cur_denom = 0
            #i = self.collection_set[d]
            for t in range(number_of_topics):
                for w in range(self.vocabulary_size):
                    cur_denom += self.term_doc_matrix[d][w]*self.topic_prob[d][t][w]

            for t in range(number_of_topics):
                numerator = 0
                for w in range(self.vocabulary_size):
                    numerator += self.term_doc_matrix[d][w]*self.topic_prob[d][t][w]
                self.document_theme_prob[d][t] = numerator/cur_denom


        for t in range(number_of_topics):
            denom2 = 0
            for d in range(self.number_of_documents):
                for w in range(self.vocabulary_size):
                    x1 = self.term_doc_matrix[d][w]
                    x2 = 1 - self.bg_prob[d][w]
                    x3 = self.topic_prob[d][t][w]
                    x4 = self.common_topic_prob[d][t][w]
                    add = x1 * x2 * x3 * x4
                    denom2 += add

            for w in range(self.vocabulary_size):
                numerator2 = 0
                for d in range(self.number_of_documents):
                    x1 = self.term_doc_matrix[d][w]
                    x2 = 1 - self.bg_prob[d][w]
                    x3 = self.topic_prob[d][t][w]
                    x4 = self.common_topic_prob[d][t][w]
                    add = x1 * x2 * x3 * x4
                    numerator2 += add
                self.theme_word_prob[t][w] = numerator2 / denom2

        for i in range(number_of_clusters):
            for t in range(number_of_topics):
                denom3 = 0.0
                for d in range(self.number_of_documents):
                    #------------------Nov22
                    if self.collection_set[d] != i:
                        continue
                    #------------------Nov22
                    for w in range(self.vocabulary_size):
                        x1 = self.term_doc_matrix[d][w]
                        x2 = 1 - self.bg_prob[d][w]
                        x3 = self.topic_prob[d][t][w]
                        x4 = 1 - self.common_topic_prob[d][t][w]
                        add = x1 * x2 * x3 * x4
                        denom3 += add

                for w in range(self.vocabulary_size):
                    numerator3 = 0
                    for d in range(self.number_of_documents):
                        # ------------------Nov22
                        if self.collection_set[d] != i:
                            continue
                        # ------------------Nov22
                        x1 = self.term_doc_matrix[d][w]
                        x2 = 1 - self.bg_prob[d][w]
                        x3 = self.topic_prob[d][t][w]
                        x4 = 1 - self.common_topic_prob[d][t][w]
                        add = x1 * x2 * x3 * x4
                        numerator3 += add
                    self.cluster_theme_word_prob[i][t][w] = numerator3 / denom3

        '''
        print('----------pi d,j ----------')
        print(self.document_theme_prob)
        print('-----------theme word prob-----------')
        print(self.theme_word_prob)
        print('-----------cluster theme word prob ---------')
        print(self.cluster_theme_word_prob)
        '''


    def calculate_likelihood(self, number_of_topics,lambda_b, lambda_c):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices

        """
        lh = 0
        for d in range(self.number_of_documents):
            i = self.collection_set[d]
            second_log_sum = 0
            for w in range(self.vocabulary_size):
                count = self.term_doc_matrix[d][w]
                base1 = lambda_b * self.background_model[w]
                base2 = 0
                for t in range(number_of_topics):
                    add = lambda_c * self.theme_word_prob[t][w] + (1 - lambda_c) * self.cluster_theme_word_prob[i][t][w]
                    add = add * self.document_theme_prob[d][t]
                    base2 += add
                base2 = base2 * (1 - lambda_b)
                log_sum = math.log(base1 + base2)
                log_sum = count * log_sum
                second_log_sum += log_sum
            lh += second_log_sum

        self.likelihoods.append(lh)
        # print(self.likelihoods)
        return lh

    def top_k(self,matrix, K, axis=1):
        if axis == 0:
            row_index = np.arange(matrix.shape[1 - axis])
            topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
            topk_data = matrix[topk_index, row_index]
            topk_index_sort = np.argsort(-topk_data, axis=axis)
            topk_data_sort = topk_data[topk_index_sort, row_index]
            topk_index_sort = topk_index[0:K, :][topk_index_sort, row_index]
        else:
            column_index = np.arange(matrix.shape[1 - axis])[:, None]
            topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
            topk_data = matrix[column_index, topk_index]
            topk_index_sort = np.argsort(-topk_data, axis=axis)
            topk_data_sort = topk_data[column_index, topk_index_sort]
            topk_index_sort = topk_index[:, 0:K][column_index, topk_index_sort]
        return topk_data_sort, topk_index_sort



    def ccmm(self, number_of_topics, number_of_clusters,max_iter,lambda_b, lambda_c, epsilon):

        """
        Model topics.
        """
        print("EM iteration begins...")


        # build term-doc matrix
        self.build_term_doc_matrix()

        # Create the counter arrays.

        # P(d, Ci, w = j)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        #P(d, Ci, w = B)
        self.bg_prob = np.zeros([self.number_of_documents, self.vocabulary_size], dtype=np.float)

        #P(d, Ci, j, w = C)
        self.common_topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)


        # P(z | d) P(w | z)
        self.initialize(number_of_topics, number_of_clusters, random=True)

        # Run the EM algorithm
        current_likelihood = self.calculate_likelihood(number_of_topics,lambda_b, lambda_c)
        print('init likelihood')
        print(current_likelihood)

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")

            prev_likelihood = current_likelihood
            self.expectation_step(number_of_topics,number_of_clusters,lambda_b, lambda_c)
            self.maximization_step(number_of_topics,number_of_clusters)
            current_likelihood = self.calculate_likelihood(number_of_topics,lambda_b, lambda_c)
            print(current_likelihood - prev_likelihood)
            if abs(current_likelihood - prev_likelihood) < epsilon or current_likelihood - prev_likelihood < 0:
                #print('Last likelihood: ', current_likelihood)
                break


        f = open('results.txt', 'w')
        common_model = [[0 for i in range(number_of_topics)] for j in range(8)]
        common_values, common_idx = self.top_k(self.theme_word_prob, 8)
        for i in range(number_of_topics):
            for k in range(8):
                common_model[k][i] = (common_values[i][k], self.vocabulary[common_idx[i][k]])
        print('Below is the common theme model')
        print(common_model)
        f.write('Below is the common theme model \n')
        f.write(str(common_model)+'\n\n')

        collection_specific_model = [[[0 for i in range(number_of_topics)] for j in range(5)] for k in range(number_of_clusters)]

        for c in range(number_of_clusters):
            collection_specific_values, collection_specific_idx = self.top_k(self.cluster_theme_word_prob[c],5)
            for i in range(number_of_topics):
                for k in range(5):
                    collection_specific_model[c][k][i] = (collection_specific_values[i][k], self.vocabulary[collection_specific_idx[i][k]])
            print('Below is the collection-specific theme model of collection ', c)
            print(collection_specific_model[c])
            f.write('Below is the collection-specific theme model of collection '+str(c)+'\n')
            f.write(str(collection_specific_model[c])+'\n\n')
        #print(self.document_theme_prob)
        f.close()

def main(args):
    documents_path = args.document_path
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    corpus.build_background_model()
    print('Background model',corpus.background_model)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = args.number_of_topics
    max_iterations = 30
    epsilon = 2
    lambda_c = args.lambda_c
    lambda_b = args.lambda_b
    number_of_clusters = args.number_of_cols
    corpus.ccmm(number_of_topics,number_of_clusters, max_iterations, lambda_b, lambda_c, epsilon)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS410 Course Project by Yutong Lin')
    parser.add_argument('--document', dest='document_path', type=str,
                        help='the document path for contextual text mining')
    parser.add_argument('--clusterNumber', dest='number_of_topics', type=int, default=5,
                        help='number of clusters in mining')
    parser.add_argument('--collectionNumber', dest='number_of_cols', type=int, default=2,
                        help='number of collections in mining')
    parser.add_argument('--c', dest='lambda_c', type=float, default = 0.25,
                        help='lambda_c in the mixture model')
    parser.add_argument('--b', dest='lambda_b', type=float, default=0.91,
                        help='lambda_b in the mixture model')
    args = parser.parse_args()
    main(args)
