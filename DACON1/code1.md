## code1 은 코랩파일입니다.

```python
import re
import pandas as pd
import numpy as np
import random
!pip install sentence-transformers datasets
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

SEED = 0

np.random.seed(SEED)
random.seed(SEED)

from google.colab import drive
drive.mount('/content/gdrive')

df = pd.read_csv('/content/gdrive/My Drive/DACON1/news.csv')
df.head()

# 제목 + 내용
df['text'] = df['title'] + ' : ' + df['contents']
df.head()

def preprocess_text(text):
    # URL 제거
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 해시태그 제거
    text = re.sub(r'#\w+', '', text)
    
    # 멘션 제거
    text = re.sub(r'@\w+', '', text)
    
    # 이모지 제거
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # 공백 및 특수문자 제거
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 숫자 제거
    text = re.sub(r'\d+', '', text)
    
    return text.lower()
df['processed_text'] = df['text'].apply(preprocess_text)
# Sentence BERT 모델 로드
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

# 텍스트 feature 추출
sentence_embeddings = model.encode(df['text'].tolist())

# 추출한 feature를 데이터프레임에 저장
df_embeddings = pd.DataFrame(sentence_embeddings)

# Sentence BERT 임베딩을 사용하여 군집화 수행
kmeans = KMeans(n_clusters=6, random_state=SEED)

df['kmeans_cluster'] = kmeans.fit_predict(sentence_embeddings)

df[df['kmeans_cluster'] == 0]['text'].head(3)
df[df['kmeans_cluster'] == 1]['text'].head(3)
df[df['kmeans_cluster'] == 2]['text'].head(3)
df[df['kmeans_cluster'] == 3]['text'].head(3)
df[df['kmeans_cluster'] == 4]['text'].head(3)
df[df['kmeans_cluster'] == 5]['text'].head(3)

mapping_dict = {
    0: 1,
    1: 3,
    2: 2,
    3: 0,
    4: 4,
    5: 5
}
df['mapping'] = df['kmeans_cluster'].apply(lambda x: mapping_dict[x])
sample = pd.read_csv('/content/gdrive/My Drive/DACON1/sample_submission.csv')
sample['category'] = df['mapping'].values
sample['category'].head()
sample.to_csv('baseline_submit.csv', index=False)
```
