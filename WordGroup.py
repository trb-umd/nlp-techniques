from gensim import corpora

def group_words(wc_data):

    #divide data into 3 different groups based on Mindset

    grouped = wc_data.groupby(wc_data.Mindset)
    fixed_data = grouped.get_group("Fixed")
    middle_data = grouped.get_group("Middle")
    growth_data = grouped.get_group("Growth")

    #get all the mission statement words from each group into their own list

    fx_words = fixed_data['Mission Statement'].values.tolist()
    md_words = middle_data['Mission Statement'].values.tolist()
    gr_words = growth_data['Mission Statement'].values.tolist()

    # Create Dictionary

    fx_id2word = corpora.Dictionary(fx_words)
    md_id2word = corpora.Dictionary(md_words)
    gr_id2word = corpora.Dictionary(gr_words)

    # Term Document Frequency, word id and word frequency, bag-of-words

    fx_corpus = [fx_id2word.doc2bow(text) for text in fx_words]
    md_corpus = [md_id2word.doc2bow(text) for text in md_words]
    gr_corpus = [gr_id2word.doc2bow(text) for text in gr_words]

    return fixed_data, middle_data, growth_data, fx_words, md_words, gr_words, fx_id2word, md_id2word, gr_id2word, \
           fx_corpus, md_corpus, gr_corpus
