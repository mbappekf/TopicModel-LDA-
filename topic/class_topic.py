import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import os
import neologdn
import unicodedata
import math

np.random.seed(0)
 
#globでの読み込みは一行ごとの読み込みはできない． 
#テキストデータのパス glob.glob('text/**/*.txt')
#text_paths = glob.glob('airpods/*.txt')←*の数に注意


#文章の読み込み
with open(r"C:\\Users\KANAME\Documents\sample_1\various\LDA\airpods\star1.txt", "r", encoding="utf-8" ) as file:
    line = file.readlines()

    for text_paths in line:
        print(text_paths.rstrip())


 
#len(text_paths)  #文章データ数
#print(text_paths)
file.close()
print(type(text_paths))
#print(open(text_paths[0], 'r',encoding="utf-8").read())


if __name__ == "__main__":
     #文章データに対する下処理（英単語を全て小文字に，数字をゼロに置換）
    def normalize(text):

        text = str(text[0])
        text = text.lower()
        text = neologdn.normalize(text)
        text = unicodedata.normalize('NFKC', text)
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
    



    #Topic数の指定
    NUM_TOPICS = 6

    #LDA
    import gensim
    
    mecab = MeCab.Tagger(r'-Ochasen -d C:\neologd')
    mecab.parse('')

    texts = []
    original_texts = []
    #使用するテキストの選択（話題比率検出側も設定）
    with open(r"C:\Users\KANAME\Documents\sample_1\master\airpods\oginal_text.txt", "r", encoding="utf-8_sig") as file:
        line = file.readlines()
        for text in line:
            text = str(text)
            original_texts.append(text)          
            text = text.split('\n')        
            text = normalize(text)                          
            words = analyzer(text, mecab, stopwords=stopwords, target_part_of_speech=['noun', 'proper_noun', 'verb', 'adjective'])
            texts.append(words)      

    #LDA，コーパスの設定
    dictionary = gensim.corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=40, no_above=0.6)#n回未満の単語とnn割以上の文章で出現する単語を削除
    corpus = [dictionary.doc2bow(t) for t in texts]
    
    #クラスタリング結果表示準備
    from collections import defaultdict
    score_by_topic = defaultdict(int)

    #corpusとtextsの数合わせ
    
    new_corpus = []
    for document_index, document_list in enumerate(texts):
        target_corpus = corpus[document_index]
        already_exists = []
        partial_corpus = []
        original_corpus_index = 0
        for word in document_list:
            if not word in already_exists:
                already_exists.append(word)
                partial_corpus.append(target_corpus[original_corpus_index])
                original_corpus_index += 1
            else:
                additional_corpus_index = document_list.index(word)
                partial_corpus.append(target_corpus[additional_corpus_index])
        new_corpus.append(partial_corpus)
    

    #LDAモデルの作成
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=NUM_TOPICS, random_state=0)


    
    #オプションでワードクラウドの準備
    from wordcloud import WordCloud
    from PIL import Image
    import matplotlib
    import matplotlib.pylab as plt
    import math
    font = {'family': 'TakaoGothic', 'size': 20 }
    matplotlib.rc('font', **font)
    fpath = "C:\Windows\Fonts\msgothic.ttc"

    num_topics = NUM_TOPICS
    ncols = math.ceil(num_topics/2)
    nrows = math.ceil(lda_model.num_topics/ncols)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows,figsize=(15,10))
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
    plt.show()
    #plt.savefig("./WordCloud_allstar.png")
    

        

    #クラスタリング結果の表示
    target_topic = 3 #<<<<0始まりであることに注意
    target_probability = 0.5
    document_index = 0

    #delete_flag: 同じファイルが存在していると上書きモードになってしまうのでだるい->1回1回消してしまう方が楽．delte_flag=Trueだと，既に存在しているファイルを消去する．基本Trueでいい．
    delete_flag = True
    #text_file_name: 解析するファイルの名前があった方が便利と思ってとりあえず書いてみた
    text_file_name = "original_text"
    #save_file_name: 解析結果の保存先ファイル名．
    save_file_name = "topic_" + str(target_topic) + "_prob_" + str(target_probability) + "_" + text_file_name + "_text.txt"
    #save_file_exists_flag: save_file_nameで保存されているファイルが既に存在するかどうかのフラグ
    save_file_exists_flag = os.path.exists(save_file_name)

    os.makedirs(r"C:\Users\KANAME\Documents\sample_1\master\airpods\topic", exist_ok=True)

    if save_file_exists_flag == True and delete_flag == True:
        os.remove(save_file_name)

    for unseen_doc, text in zip(corpus, texts):
        #print("unseen_doc: ", unseen_doc)
        #print("lda_model: ", lda_model[unseen_doc])
        for topic, score in lda_model[unseen_doc]:
            #print("document_index: " + str(document_index) + ", topic_index: " + str(topic) + ", score: " + str(score))
            score_by_topic[int(topic)] = float(score)
            if score >= target_probability and topic == target_topic:
                with open(save_file_name, "a", encoding="utf-8") as f:
                    #print("document index: ", document_index)
                    #print("original_texts[document_index]: ", original_texts[document_index])
                    #print("Section1 worked.")
                    f.write(original_texts[document_index])
            elif len(lda_model[unseen_doc]) == 1 and topic == target_topic and lda_model[0][0] >= target_probability:
                with open(save_file_name, "a", encoding="utf-8") as f:
                    #print("document index: ", document_index)
                    #print("original_texts[document_index]: ", original_texts[document_index])
                    #print("Section2 worked.")
                    f.write(original_texts[document_index])
        #for i in range(NUM_TOPICS):
            #print('{:.2f}'.format(score_by_topic[i]), end='\t')
            #words = ','.join([features[i] for i in test_texts()[:-7-1:-1]]) 
            #print(words,"\n")
        document_index += 1