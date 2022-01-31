#coding: UTF-8
"""
テキストの前処理・ヒストグラムの作成・データフレームの作成・スコアの抽出
"""

"""
顔文字，絵文字，記号（句点と読点以外）の削除
文章の長さでフィルター
１文章をリストの１要素で返す．
"""

"""
＜作成されるもの＞
極性値ヒストグラム・スコア抽出ファイル
"""

from icecream import ic
import neologdn
import unicodedata
import re
import emoji
import itertools
import MeCab
import nagisa
import oseti
import pandas as pd
import itertools
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams



#mecabオブジェクトの作成
mecab = MeCab.Tagger(r'-d C:\neologd')
mecab.parse('')



#顔文字の定義・作成
def extract_kaomoji(text):

    KAOMOJI_LEN = 3

    #顔文字の定義
    results = nagisa.extract(text, extract_postags=['補助記号'])
    words = results.words
    kaomoji_words = []
    kaomoji_idx = [i for i, w in enumerate(words) if len(w) >= KAOMOJI_LEN]
    kaomoji_hands = ['ノ', 'ヽ', '∑', 'm', 'O', 'o', '┐', '/', '\\', '┌', 'w', 'v', 'b', 'V' '!', '！']

    #顔文字の検索
    for i in kaomoji_idx:
        kaomoji = words[i]
        try:
            #顔文字の右手
            if words[i-1] in kaomoji_hands and 0 < i:
                kaomoji = words[i-1] + kaomoji
            
            #顔文字の左手
            if words[i+1] in kaomoji_hands:
                kaomoji = kaomoji + words[i+1]
        
        except IndexError:
            pass
        finally:
            kaomoji_words.append(kaomoji)
    
    return kaomoji_words

#顔文字変換用関数
KAOMOJI_PH = "。"
def replace_kaomoji(text, target_list, PH):

    for trg in target_list:
        text = text.replace(trg, PH)

    return text

#対象となる文字の保護,整形
def select_symbols(text):
    
    #テキストの整形
    text = neologdn.normalize(text)
    text = text.lower()
    text = unicodedata.normalize('NFKC', text)

    """
    記号の整形
    """
    #文末記号（ピリオド，！,？）を「。」に変換(半角・全角に注意)
    text = text.replace('．','。').replace('.','。').replace('!','。').replace('！','。').replace('?','。').replace('？','。').replace('\u3000', '')

    #連続する句点の削除
    text = re.sub(r'[。]+', '。', text)


    #句点の統一
    text = text.replace('、','，').replace(',', '，')

    #連続するコンマの削除
    text = re.sub(r'[，]+', '，', text)


    return text


#各種リストの作成
def create_list(text, mecab, target_part_of_speech = ['symbol', 'one_letter_noum', 'period_symbol', 'comma_symbol']):

    node = mecab.parseToNode(text)
    delete_symbols = []
    wakachi = []

    while node:
        features = node.feature.split(',')
        surface = node.surface

        #分かち書きしたものをリストに格納
        wakachi.append(surface)

        #各種フラグの作成
        symbol_flag = (features[0] == '記号')
        period_symbol_flag = (features[0] == '記号') & (features[1] == '句点')
        comma_symbol_flag = (features[0] == '記号') & (features[1] == '読点')
        one_letter_noum_flag = (features[0] == '名詞')

        if ('symbol' in target_part_of_speech) & (symbol_flag) & (not period_symbol_flag) & (not comma_symbol_flag):
            delete_symbols.append(surface)
        
        elif ('one_letter_noum' in target_part_of_speech) & one_letter_noum_flag:
            if surface == '*':
                delete_symbols.append(surface)
        
        node = node.next

        wakachi = list(filter(None, wakachi))

    return wakachi, delete_symbols


#不要な記号の削除用
remove_PATH = ""
def remove_symbols(text, target_list, PH):

    for trg in target_list:
        text = text.replace(trg, PH)
    
    return text

#文章の長さでフィルター
def len_filter(text):
    result = []
    target = ''

    #リスト内の要素の文字数でフィルター
    #空のリストも削除
    for item in text:
        len_flag = len(item) < 4
        if (item != target) & (not len_flag):
            result.append(item)
    cleaned_text = result

    return cleaned_text

"""
＊＊＊データフレームの作成＊＊＊
original_txtリストには整形済み文章データ
kyoku_scoreリストには各文章の点数を記入
"""

def create_df(cleaned_text):
    original_txt = cleaned_text
    analyzer = oseti.Analyzer()
    kyoku_score = []

    for item in cleaned_text:
        #ic(item)
        score = analyzer.analyze(item)
        kyoku_score.append(score)
    
    #kyoku_score = list(itertools.chain.from_iterable(kyoku_score))

    #DataFrameの作成
    kyoku_df = pd.DataFrame({'元の文章': original_txt, '極性値': kyoku_score})

    return original_txt, kyoku_score, kyoku_df


#極性値（score）のヒストグラムの作成
def create_hist(kyoku_df):
    x = kyoku_df['極性値']
    
    fig = plt.figure(figsize =(20,20))
    ax = fig.add_subplot(1,1,1)
    ax.hist(x, bins=20, range=(-1.0, 1.0))
    ax.set_xlabel('極性値', fontsize=40)
    ax.set_ylabel('文章数', fontsize=40)
    ax.grid(True)
    ax.tick_params(labelsize=40)

    #rcParams['figure.figsize'] = 40,40
    plt.savefig("./hist.png")
    plt.show()

    return score_hist


"""
極性データフレームの中から取り出したい点数の文章のみをテキストファイルに起こす．
ファイル名やデータの保存場所は随時更新
"""
def extraction_score(kyoku_df):

    #ファイル既存フラグ
    delete_flag = True

    #取り出す極性値の設定
    target_score = 1.0

    #保存先ファイルの選択
    document_index = 0

    text_file_name = "top3pro0.5"
    save_file_name = "score" + str(target_score) + "_" + text_file_name + ".txt"
    save_file_exists_flag = os.path.exists(save_file_name)
    os.makedirs(r"C:\Users\KANAME\Documents\sample_1\master\airpods\kyokusei", exist_ok=True)
    if (save_file_exists_flag == True) & (delete_flag == True):
        os.remove(save_file_name)
    for text_idx, score in enumerate(kyoku_df['極性値']):
        if score == target_score:
            with open(save_file_name, "a", encoding="utf-8") as file:
                file.write(kyoku_df["元の文章"][text_idx])
                if text_idx == (len(kyoku_df.index)-1):
                    pass
                else:
                    file.write("\n")
    
    return save_file_name


#使用するデータの読み込み
with open(r"C:\Users\KANAME\Documents\sample_1\master\airpods\topic\topic_3_prob_0.5_original_text_text.txt", "r", encoding="utf-8") as file:
    text = file.read()
    #前処理
    kaomoji_list = extract_kaomoji(text)
    text = replace_kaomoji(text, kaomoji_list, KAOMOJI_PH)
    text = select_symbols(text)
    text = ''.join(['' if c in emoji.UNICODE_EMOJI else c for c in text])
    wakachi, delete_symbols = create_list(text, mecab, target_part_of_speech=['symbol', 'one_letter_noum'])
    text = remove_symbols(text, delete_symbols, remove_PATH)
    text = re.split('\n|。', text)
    cleaned_text = len_filter(text)
    #DataFrameの作成
    original_txt, kyoku_score, kyoku_df = create_df(cleaned_text)
    #DataFrameのDebug用
    #original_txt, kyoku_score = create_df(cleaned_text)
    #ヒストグラムの作成
    #score_hist = create_hist(kyoku_df)
    #scoreの抽出→ファイルに保存
    #save_file_name = extraction_score(kyoku_df)

    #debug用
    #ic(len(original_txt))
    #ic(len(kyoku_score))
    #ic(cleaned_text)
    #ic(original_txt)
    #ic(kyoku_score)
    ic(kyoku_df)
    #ic(delete_symbols)
    #print(kyoku_score)

    """
    csvファイルに保存
    """
    kyoku_df.to_csv('kyokusei.csv', encoding="shift_jis")