import oseti
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams


#対象のテキストデータは解析ごとに確認
with open(r"C:\Users\KANAME\Documents\sample_1\master\airpods\topic\topic_3_prob_0.5_original_text_text.txt", "r", encoding="utf-8") as file:
    text = file.read()
    text = text.replace("!","。").replace("！","。").replace("?","。").replace("？","。").replace("\n","。")
    text = text.replace("。。","。")
    analyzer = oseti.Analyzer()
    texts = analyzer.analyze(text)
    text_detail = analyzer.analyze_detail(text)

    #print(texts)
    #print("----------------")
    #print(詳細↓)
    #print(text_detail)


#ヒストグラム作成

x = texts

fig = plt.figure(figsize =(20,20))
ax = fig.add_subplot(1,1,1)


ax.hist(x, bins=20, range=(-1.0, 1.0))
ax.set_title('極性値：ヒストグラム', fontsize=40)
ax.set_xlabel('極性値', fontsize=40)
ax.set_ylabel('文章数', fontsize=40)
ax.grid(True)
ax.tick_params(labelsize=40)

#rcParams['figure.figsize'] = 40,40
plt.savefig("./hist.png")
plt.show()
