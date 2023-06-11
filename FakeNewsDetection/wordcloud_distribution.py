import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

data = pd.read_csv("data/train.csv")

sns.set()
plt.figure(figsize=(15, 8))
plt.grid(None)
sns.countplot(x='label', data=data)
plt.title("Distribution of Fake News")
plt.savefig("distribution.png")
plt.show()

plt.figure(figsize=(10, 10))
plt.grid(None)
news = " ".join(data["text"][data["label"] == 1].astype(str))  # Convert to string and combine all text of fake news
WC = WordCloud(width=1000, height=500, max_words=500, min_font_size=5)
news_words = WC.generate(news)
plt.imshow(news_words, interpolation='bilinear')
plt.savefig("word_cloud.png")
plt.show()

plt.figure(figsize=(10, 10))
plt.grid(None)
news = " ".join(data["text"][data["label"] == 0].astype(str))  # Convert to string and combine all text of real news
WC = WordCloud(width=1000, height=500, max_words=500, min_font_size=5)
news_words = WC.generate(news)
plt.imshow(news_words, interpolation='bilinear')
plt.savefig("word_cloud1.png")
plt.show()
