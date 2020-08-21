# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]

# # DESAFIO 1
# 
# Use o NLTK para criar um pipeline que realize as seguintes tarefas, nesta ordem: 
# Tokenization, Sentence Splitting, Lemmatization, Stemming e POS tagging 
# 
# Em seguida gere as seguintes informações estatísticas e gráficos de barras em relação ao texto em inglês task1.txt: 
# Quantas palavras temos em todo o texto? 
# Quantos radicais (stemming) diferentes existem?
# Qual o número de sentenças e a média de tokens por sentença?
# Gere um gráfico de barra do conjunto de POS tags de todas as palavras do texto. Ordene os resultados e responda: quais classes gramaticais correspondem a mais de 70 ou 80% do total?

# %%
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from string import punctuation
from collections import Counter

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()


# %%
def read_file(filename: str) -> str:  
  with open(filename, 'r', encoding='utf-8') as file:
    return file.read()

def lematize_tokens(tokens: list) -> list:
    return [lemmatizer.lemmatize(token) for token in tokens]

def stem_tokens(tokens: list) -> list:
    return [stemmer.stem(token) for token in tokens]

def get_onlywords_from_tokens(tokens: list) -> list:
    # usa-se token[0] pois caracteres como ´´ não são pegos na validação. Então nesse caso fica apenas ` que contém no punctuation
    return [token for token in tokens if token[0] not in punctuation and not token.isdigit()]

def get_qtd_tokens_for_sentence(sentence: str) -> int:
  return len([word for word in nltk.word_tokenize(sentence)])

def get_average_tokens_for_sentence(sentence_list: list) -> int:
  return sum(sentence_list) / len(sentence_list)

# %% [markdown]
# ## Tokenization, Sentence Splitting, Lemmatization, Stemming e POS tagging

# %%
text_1 = read_file('task_1.txt')
text_1_word_tokens = nltk.word_tokenize(text_1)
text_1_sent_tokens = nltk.sent_tokenize(text_1)
text_1_lemmas = lematize_tokens(text_1_word_tokens)
text_1_stems = stem_tokens(text_1_word_tokens)
text_1_taggs = nltk.pos_tag(text_1_word_tokens)

# %% [markdown]
# ## Quantas palavras temos em todo o texto? (também exluindo números)

# %%
text_1_onlywords = get_onlywords_from_tokens(text_1_word_tokens)

print(f"No text_1 existem {len(text_1_onlywords)} palavras.")

# %% [markdown]
# ## Quantos radicais (stemming) diferentes existem? (contando apenas com palavras)

# %%
text_1_onlyword_stemms = stem_tokens(text_1_onlywords)

different_stemms = set(text_1_onlyword_stemms)
print(f'No text_1 exitem {len(different_stemms)} radicais diferentes.')

# %% [markdown]
# ## Qual o número de sentenças e a média de tokens por sentença?

# %%


tokens_for_sentence = { sentence: get_qtd_tokens_for_sentence(sentence) for sentence in text_1_sent_tokens}

for sentence, qtd in tokens_for_sentence.items():
    print(f'A sentença "{sentence}" contém {qtd} tokens.\n')



# %%
average = get_average_tokens_for_sentence(list(tokens_for_sentence.values()))
print(f'Média de *tokens* por sentença é {average:.2f}.')

# %% [markdown]
# ## Gere um gráfico de barra do conjunto de POS tags de todas as palavras do texto. Ordene os resultados e responda: quais classes gramaticais correspondem a mais de 70 ou 80% do total?

# %%
taggs = nltk.pos_tag(text_1_onlywords)
counts = Counter( tag for (word, tag) in taggs)
print(counts)
fig = plt.figure()
ax = fig.add_axes([1,1,2,2])

ax.bar(counts.keys(), counts.values(), width=0.5, color="red")
ax.legend(labels=['Count'])

plt.show()


# %%
print(f"Os 3 radicais mais comuns são: {counts.most_common(3)}.")

# %% [markdown]
# 
# # DESAFIO 2
# 
# 
# Crie um programa em python que tokenize o arquivo em português task2.txt, a imagem "pseudo-código" nesse arquivo tem o pseudo-código do algoritmo de tokenização. 

# %%





# %%



