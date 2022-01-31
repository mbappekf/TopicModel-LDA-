import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import os
import neologdn
import unicodedata
import MeCab
import gensim
import matplotlib
import matplotlib.pylab as plt
from wordcloud import WordCloud
from PIL import Image
import matplotlib
import matplotlib.pylab as plt
import math
np.random.seed(0)
 
mecab = MeCab.Tagger(r'-Ochasen -d C:\neologd')
mecab.parse('')


if __name__ == "__main__":
    
    #小文字大文字の統一，正規化，数字の削除
    def normalize(text):

        text = str(text[0])
        text = text.lower()
        text = neologdn.normalize(text)
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\d+', '0', text)
        text = text.split('\n')
        return text
    
    #品詞分解（フラグを選んで判定する品詞の選択）
    def analyzer(text, mecab, stopwords=[], target_part_of_speech=['proper_noun', 'noun', 'verb', 'adjective']):
        
        text = str(text[0])
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

   

    #ストップワードの定義
    with open(r"C:\Users\KANAME\Documents\sample_1\master\airpods\stop_word.txt", 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
        stopwords = [stopword.strip() for stopword in stopwords]
    while '' in stopwords:
        stopwords.remove('')
    
    #使用するテキストのダウンロード
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


    #LDAモデルの作成
    dictionary = gensim.corpora.Dictionary(texts)

    #単語フィルター：no_below がn数以下の文章に出現する単語の消去， no_above が全体のn割以上に出現する単語の消去
    dictionary.filter_extremes(no_below=20, no_above=0.6) 
    corpus = [dictionary.doc2bow(t) for t in texts]


    
    #トピック数の決定
    #トピック数のグラフ表示

    #コヒーレンスのみ対象
    font = {'family': 'TakaoGothic','size': 22}
    matplotlib.rc('font', **font)
    
    start = 2
    limit = 8
    step = 1
    
    coherence_vals = []
    perplexity_vals = []
    
    for n_topic in tqdm(range(start, limit, step)):
    
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topic, random_state=0)
        #perplexity_vals.append(np.exp2(-lda_model.log_perplexity(corpus)))
        coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_vals.append(coherence_model_lda.get_coherence())
    
    x = range(start, limit, step)
    
    fig, ax1 = plt.subplots(figsize=(15,10))
    
    c1 = 'darkturquoise'
    ax1.plot(x, coherence_vals, 'o-', color=c1, linestyle='None')
    ax1.set_xlabel('Num Topics')
    ax1.set_ylabel('Coherence', color=c1); ax1.tick_params('y', colors=c1)
    
    
    #c2 = 'slategray'
    #ax2 = ax1.twinx()
    #ax2.plot(x, perplexity_vals, 'o-', color=c2, linestyle='None')
    #ax2.set_ylabel('Perplexity', color=c2); ax2.tick_params('y', colors=c2)
    
    
    
    ax1.set_xticks(x)
    fig.tight_layout()
    plt.savefig("./topic_num.png")
    plt.show()




    #wordcloud
    #トピック数の設定
    NUM_TOPICS = 6
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=NUM_TOPICS, random_state=0)
    
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
    plt.savefig("./WordCloud.png")
    