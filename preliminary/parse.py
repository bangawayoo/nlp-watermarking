if __name__ == "__main__":
    NUM_SAMPLE = 1
    import sys
    sys.path.append("/workspace/ilm/")

    import pickle
    from ilm.datasets import arxiv_cs_abstracts, roc_stories

    corpus = arxiv_cs_abstracts("train", attrs=['title', 'abstract'])
    corpus = [c.replace("\n", " :") for c in corpus]
    corpus = corpus[:NUM_SAMPLE]

    from supar import Parser
    parser = Parser.load('biaffine-dep-en')
    result = []
    for idx, c in enumerate(corpus):
        dataset = parser.predict(c, lang='en', prob=True, verbose=False)
        predicted = dataset[0].values
        tokens, tags, arcs = predicted[1], predicted[-3], predicted[-4]
        result.append([idx, tokens, tags, arcs])

    with open(f"./abs_parsed-{NUM_SAMPLE}.pkl", 'wb') as f:
        pickle.dump(result, f)

