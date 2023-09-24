## code1 은 코랩파일입니다.
- 쉽게 말해서 뉴스 기사 데이터를 전처리하고, 기사들의 군집을 찾아내기 위한 작업을 수행합니다.
- 구체적인 과정은 다음과 같습니다:
  
필요한 라이브러리와 모듈을 임포트합니다. 특히 sentence-transformers는 문장 임베딩을 위한 라이브러리이며, datasets는 다양한 데이터셋을 다루기 위한 라이브러리입니다.
시드값을 설정하여 연산 결과의 재현성을 보장합니다.
Google Colab 환경에서 Google Drive를 마운트합니다. 이를 통해 Google Drive에 저장된 파일을 직접 불러올 수 있습니다.
news.csv라는 뉴스 기사 데이터를 불러와 데이터프레임 df로 저장합니다.
기사의 제목과 내용을 합쳐 text라는 새로운 열을 생성합니다.
preprocess_text 함수를 정의합니다. 이 함수는 주어진 텍스트 데이터에 대한 전처리 작업을 수행합니다. 전처리 작업은 URL, 해시태그, 멘션, 이모지, 특수문자, 숫자 제거 등이 포함됩니다.
preprocess_text 함수를 이용하여 text 열의 텍스트를 전처리하고 그 결과를 processed_text 열에 저장합니다.
Sentence BERT 모델을 로드하고, 이 모델을 사용하여 text 열의 텍스트 데이터를 임베딩합니다. 이 임베딩 결과는 sentence_embeddings에 저장됩니다.
sentence_embeddings를 기반으로 KMeans 알고리즘을 사용하여 군집화를 수행합니다. 군집의 개수는 6개로 설정되어 있습니다.
각 군집에서 속하는 기사의 제목과 내용을 출력합니다.
군집 번호를 재매핑하는 mapping_dict 딕셔너리를 정의하고, 이를 사용하여 군집 번호를 재매핑합니다.
sample_submission.csv라는 제출 예시 파일을 불러오고, 해당 파일의 category 열에 재매핑된 군집 번호를 저장합니다.
결과를 baseline_submit.csv 파일로 저장합니다.

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
