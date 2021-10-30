import gensim

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from wordcloud import WordCloud

def apply_lda(state_data, words, id2word, corpus, stop_words):

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=4, random_state=100,
                                                update_every=1, chunksize=100, passes=10, alpha='auto',
                                                per_word_topics=True)

    print("\nLDA model topics:\n")
    print(lda_model.print_topics())

    # extract dominant topics for each document, weight, and keywords

    def format_topics_sentences(ldamodel=None, corpus=corpus, texts=state_data):

        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document

        for i, row_list in enumerate(ldamodel[corpus]):
            row = row_list[0] if ldamodel.per_word_topics else row_list
            row = sorted(row, key=lambda x: (x[1]), reverse=True)

            # Get the Dominant topic, Perc Contribution and Keywords for each document

            for j, (topic_num, prop_topic) in enumerate(row):

                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4),
                                                                      topic_keywords]), ignore_index=True)

                else:
                    break

        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output

        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

        return(sent_topics_df)

    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=words)

    # Format

    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # word cloud

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = lda_model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()