import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import os
import math
np.random.seed(0)



if __name__ == "__main__":
     #文章データに対する下処理（英単語を全て小文字に，数字をゼロに置換）
    def normalize(text):

        text = str(text[0])
        text = text.lower()
        text = re.sub(r'\d+', '0', text)
        text = text.split('\n')
        return text


    #MeCabで整理
    import MeCab
    
    def analyzer(text, mecab, stopwords=[], target_part_of_speech=['proper_noun', 'noun', 'verb', 'adjective']):
        
        text = str(text[0])
        mecab.parse('')
        node = mecab.parseToNode(text)
        words = []
        
        while node:
            
            features = node.feature.split(',')
            surface = features[6]
            
            if (surface == '*') or (len(surface) < 2) or (surface in stopwords):
                node = node.next
                continue
                
            noun_flag = (features[0] == '名詞')
            proper_noun_flag = (features[0] == '名詞') & (features[1] == '固有名詞')
            verb_flag = (features[0] == '動詞') & (features[1] == '自立')
            adjective_flag = (features[0] == '形容詞') & (features[1] == '自立')
            
    
            if ('proper_noun' in target_part_of_speech) & proper_noun_flag:
                words.append(surface)
            elif ('noun' in target_part_of_speech) & noun_flag:
                words.append(surface)
            elif ('verb' in target_part_of_speech) & verb_flag:
                words.append(surface)
            elif ('adjective' in target_part_of_speech) & adjective_flag:
                words.append(surface)
            
            node = node.next
            
        return words

   

    #ストップワードの排除
    with open(r"C:\Users\KANAME\Documents\sample_1\master\airpods\stop_word.txt", 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
        stopwords = [stopword.strip() for stopword in stopwords]

    while '' in stopwords:
        stopwords.remove('')


    #LDA
    import gensim
    
    mecab = MeCab.Tagger(r'-Ochasen -d C:\neologd')
    mecab.parse('')

    texts = []
    original_texts = []
    with open(r"C:\Users\KANAME\Documents\sample_1\master\airpods\oginal_text.txt", "r", encoding="utf-8_sig") as file:
        line = file.readlines()
        for text in line:
            text = str(text)
            original_texts.append(text)          
            text = text.split('\n')        
            text = normalize(text)                          
            words = analyzer(text, mecab, stopwords=stopwords, target_part_of_speech=['noun', 'proper_noun', 'verb', 'adjective'])
            texts.append(words)      

    #辞書の作成
    dictionary = gensim.corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=10, no_above=0.4) #n回未満の単語とnn割以上の文章で出現する単語を削除
    corpus = [dictionary.doc2bow(t) for t in texts]

    #Topic数の指定
    NUM_TOPICS = 4

    #LDAモデルの作成
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=NUM_TOPICS, random_state=0)





    # Visualize
    # wordcloud
    from wordcloud import WordCloud
    from PIL import Image
    import matplotlib
    import matplotlib.pylab as plt
    #font = {'family': 'TakaoGothic'}
    #matplotlib.rc('font', **font)
    fpath = "C:\Windows\Fonts\msgothic.ttc"

    num_topics = NUM_TOPICS
    ncols = math.ceil(num_topics/2)
    nrows = math.ceil(lda_model.num_topics/ncols)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows,figsize=(30,20))
    axs = axs.flatten()

    def color_func(word, font_size, position, orientation, random_state, font_path):
        return 'darkturquoise'

    for i, t in enumerate(range(lda_model.num_topics)):

        x = dict(lda_model.show_topic(t, 25))
        im = WordCloud(
            font_path=fpath,
            width = 600,
            height = 400,
            prefer_horizontal = 1, #文字を横向きに
            background_color='white', #背景色
            #color_func=color_func,
            colormap = 'tab10', #文字色の数
            random_state=0
        ).generate_from_frequencies(x)
        axs[i].imshow(im)
        axs[i].axis('off')
        axs[i].set_title('Topic ' +str(t))
        axs[i].title.set_size(100)
        

    plt.tight_layout()
    plt.savefig("./WordCloud2.png")
    plt.show()