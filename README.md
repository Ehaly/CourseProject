# CourseProject

Please fork this repository and paste the github link of your fork on Microsoft CMT. Detailed instructions are on Coursera under Week 1: Course Project Overview/Week 9 Activities.

Please read Project_Documentation for more details.

To run the scraper code, run “jupyter notebook”.

The code could be run by command “python model.py -h”.

To run the first example, “python model.py --document wars_news.txt --clusterNumber 5 --collectionNumber 2 --c 0.25 --b 0.91”

To run the second example, “python model.py --document laptop_reviews.txt --clusterNumber 4 --collectionNumber 3 --c 0.7 --b 0.96”

The class Corpus consists of the following functions:
__init__(document_path): initialize a Corpus object.

build_corpus(): read the document in document path, store the collection number and the document in self.collection and self.documents.

build_vocabulary(): read the documents and build the vocabulary for the whole dataset.

build_background_model(): build the background model from the whole dataset.

build_term_doc_matrix(): Construct the term-document matrix where each row represents a document, each column represents a vocabulary term.self.term_doc_matrix[i][j] is the count of term j in document i.

initialize(self, number_of_collections, number_of_clusters, random=True):  initialize the matrices document_topic_prob , topic_word_prob and collection_topic_word_prob.

expectation_step(number_of_collections,number_of_clusters,lambda_b, lambda_c): the E-step updates the P(zd ci w = j), i.e. the topic_prob matrix, p(zd,Ci,w = B) i.e. the bg_prob matrix, and p(zd,Ci ,j,w = C ), i.e. the common_topic_prob matrix.

maximization_step(number_of_collections, number_of_clusters ): the M-step updates the matrices document_topic_prob , topic_word_prob and collection_topic_word_prob

calculate_likelihood(number_of_collections,lambda_b, lambda_c): Calculate the current log-likelihood of the model using the model's updated probability matrices.

ccmm(number_of_collections, number_of_clusters, max_iter, lambda_b, lambda_c, epsilon): execute the text mining on the document passed in in max_iter times of iteration.