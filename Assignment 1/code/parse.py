import sys
import nltk
nltk.download('averaged_perceptron_tagger')
from providedcode.transitionparser import TransitionParser
from providedcode.dependencygraph import DependencyGraph


def parser(sentences, model):
    tp = TransitionParser.load(model)
    for sentence in sentences:
        dgraph = DependencyGraph.from_sentence(sentence)
        parsed = tp.parse([dgraph])
        print parsed[0].to_conll(10).encode('utf-8')


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Usage: parse.py path_to.model'
    parser(sys.stdin, sys.argv[1])