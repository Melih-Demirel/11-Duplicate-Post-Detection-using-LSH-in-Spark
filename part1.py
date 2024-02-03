from lxml import etree
from typing import NamedTuple, OrderedDict
import bleach
import sys
import html
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import profile

N_SHINGLES = 5
SAMPLES = 1000
XML_FILE = sys.argv[1]
LOWER_STRINGS = True
DEBUG_MODE = False
STOPWORDS = set(stopwords.words('english'))
random.seed(42)

posts = []
posts_shingles = []
posts_similarities = []

class post(NamedTuple):
    id: int
    body: str

class post_shingles(NamedTuple):
    id: int
    shingles: frozenset[tuple]

class post_similarity(NamedTuple):
    id1: int
    id2: int
    similarity: float

def refineBody(body):
    unespaced = html.unescape(body)
    bleached = bleach.clean(unespaced, tags=[], attributes={}, strip=True)
    refined_body = bleached.replace('\n', '')
    return refined_body
    
# Source: https://github.com/Networks-Learning/stackexchange-dump-to-postgres/blob/master/row_processor.py
def parse():
    context = etree.iterparse(XML_FILE, events=("end",))
    for _, elem in context:
        if elem.tag == "row":
            assert elem.text is None, "The row wasn't empty"
            yield post(int(elem.attrib['Id']), refineBody(elem.attrib['Body']))
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

def readData():
    posts_list = parse()
    for row_attribs in posts_list:
        posts.append(row_attribs)

    print(f'Read {XML_FILE}: {len(posts)} posts found!')
    if DEBUG_MODE:
        print(posts[:2])

def createShingles(post):
    words = word_tokenize(post.body)
    if LOWER_STRINGS:
        filtered_words = [word.lower() for word in words if word.lower() not in STOPWORDS]
    else:
        filtered_words = [word for word in words if word.lower() not in STOPWORDS]
    shingles = frozenset([' '.join(gram) for gram in list(ngrams(filtered_words, N_SHINGLES))])
    return post_shingles(id=post.id, shingles=shingles)

def dataPreperation():
    readData()
    if DEBUG_MODE:
        print(posts_shingles[0].shingles)


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0

def compute_jaccard_similarity(posts_shingles):
    post_pairs = list(combinations(posts_shingles, 2))
    for pair in post_pairs:
        posts_similarities.append(post_similarity(
            id1=pair[0].id,
            id2=pair[1].id,
            similarity=jaccard_similarity(pair[0].shingles, pair[1].shingles)
        ))

def plot():
    similarities = [entry.similarity for entry in posts_similarities]
    bin_edges = np.arange(0, 1.02, 0.02)
    counts, bin_edges = np.histogram(similarities, bins=bin_edges)
    for i, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        print(f"Bin {i + 1}: {start:.2f} - {end:.2f} | Count: {counts[i]}")

    plt.hist(similarities, bins=bin_edges, log=True, edgecolor='black')
    plt.title('Jaccard Similarity Distribution')
    plt.xlabel('Jaccard Similarity')
    plt.ylabel('Number of Pairs (log scale)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def bruteForce():
    random_indices = random.sample(range(len(posts)), SAMPLES)
    random_posts = [posts[index] for index in random_indices]
    for random_post in random_posts:
        posts_shingles.append(createShingles(random_post))
    
    compute_jaccard_similarity(posts_shingles)
    plot()

def main():
    '''
        Only downloading once is enough!
    '''
    # nltk.download('punkt')
    # nltk.download('stopwords')

    dataPreperation()
    bruteForce()

    return 1

main()