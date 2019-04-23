import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# loading Emma by Jane Austen
from nltk.corpus import gutenberg
text = gutenberg.raw('austen-emma.txt')

# Generate a word cloud image
wordcloud = WordCloud(max_font_size=72).generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



# loading course syllabus
f = open('Analytics_Syllabus_v2.txt','r', encoding = "ISO-8859-1")
text = f.read()
f.close()

# Generate a word cloud image
wordcloud = WordCloud(max_font_size=72, background_color='white').generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

