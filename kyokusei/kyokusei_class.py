"""
データフレーム内のデータから所望のスコアのデータのみテキストファイルに抽出
"""
import pandas as pd
import os 
from icecream import ic

kyoku_df = pd.read_csv(r'C:\Users\KANAME\Documents\sample_1\master\airpods\kyokusei\kyokusei.csv', encoding="cp932")

#debug用
#ic(kyoku_df)


"""
書き込み用ファイルの作成
"""


#フラグの作成
target_score = 


#delete_flag: 同じファイルが存在していると上書きモードになってしまうのでだるい->1回1回消してしまう方が楽．delte_flag=Trueだと，既に存在しているファイルを消去する．基本Trueでいい．
delete_flag = True
#save_file_name: 解析結果の保存先ファイル名．
save_file_name = "_kyoku_score_" + str(target_score) + "_.txt"
#save_file_exists_flag: save_file_nameで保存されているファイルが既に存在するかどうかのフラグ
save_file_exists_flag = os.path.exists(save_file_name)

os.makedirs(r"C:\Users\KANAME\Documents\sample_1\master\airpods\kyokusei", exist_ok=True)

if (save_file_exists_flag == True) & (delete_flag == True):
    os .remove(save_file_name)
