# -*- coding: utf-8 -*-
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, save_model, Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate, Dropout
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import cv2
import argparse
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import sys

user_image_path = 'C:/Users/disse/OneDrive/Desktop/styling/CODY_Model/picture/winter_espa.jpg'
age_model_path = 'C:/Users/disse/OneDrive/Desktop/styling/CODY_Model/model/age_model_test.h5'
gender_model_path = 'C:/Users/disse/OneDrive/Desktop/styling/CODY_Model/model/gender_model_test.h5'
clothes_data_path = 'C:/Users/disse/OneDrive/Desktop/styling/CODY_Model/_data/styles.csv'
user_model_path = 'C:/Users/disse/OneDrive/Desktop/styling/CODY_Model/_data/user.csv'

def generate_user_info(age_model_path, gender_model_path, user_image_path):
    # 커스텀 객체를 등록
    custom_objects = {'mse': MeanSquaredError}

    # 모델 로드
    age_model = load_model(age_model_path, custom_objects=custom_objects)
    gender_model = load_model(gender_model_path, custom_objects=custom_objects)

    # 이미지 로드 및 전처리
    img = cv2.imread(user_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # 예측
    age_preds = age_model.predict(img)
    gender_preds = gender_model.predict(img)

    # 결과 저장
    results = {
        'age': int(age_preds[0]),
        'gender': 'female' if gender_preds[0] > 0.5 else 'male'
    }

    print(age_preds[0])
    print(gender_preds[0])

    with open('user_info.json', 'w') as fp:
        json.dump(results, fp)

    return results

print(generate_user_info(age_model_path, gender_model_path, user_image_path))

# 최종 후보 선정모델 - NCF

a = 2 * (0.43810000000000003 * 0.5162228407177347144 / (0.43810000000000003 + 0.5162228407177347144))
print(a)

# DataLoader - 데이터 로드 및 전처리

class DataLoader:
    def __init__(self, user_data_path, item_data_path):
        self.user_data = pd.read_csv(user_data_path)
        self.item_data = pd.read_csv(item_data_path)

    def load_data(self):
        # 사용자 및 아이템 인덱싱
        self.user_data['user_index'] = self.user_data['user_id'].astype('category').cat.codes
        self.item_data['item_index'] = self.item_data['id'].astype('category').cat.codes

        # 사용자 데이터와 아이템 데이터 결합
        user_item_interaction = pd.merge(self.user_data, self.item_data, how='cross')

        # 상호작용 확률을 조정하기 위한 가중치 설정
        user_item_interaction['interaction_probability'] = np.random.rand(user_item_interaction.shape[0])
        user_item_interaction['interaction'] = np.where(user_item_interaction['interaction_probability'] > 0.9, 1, 0)

        # 훈련 데이터와 테스트 데이터 분리
        train, test = train_test_split(user_item_interaction, test_size=0.2, random_state=42)
        return train, test

# Metric - top@K 메트릭 계산

class Metric:
    def __init__(self, test_positives):
        self.test_positives = test_positives

    def hit_rate(self, top_k_recommended, actual_items):
        if len(top_k_recommended) == 0:
            return 0
        is_hit = np.isin(top_k_recommended, actual_items)
        return np.mean(is_hit)

    def ndcg(self, top_k_recommended, actual_items):
        if len(top_k_recommended) == 0 or len(actual_items) == 0:
            return 0
        dcg = 0.0
        for i, item in enumerate(top_k_recommended):
            if item in actual_items:
                dcg += 1.0 / np.log2(i + 2)
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(len(actual_items)))
        return dcg / ideal_dcg if ideal_dcg > 0 else 0

    def evaluate_model(self, ncf_model, num_users, num_items, top_k=5, batch_size=256):
        hits = []
        ndcgs = []
        precisions = []
        recalls = []
        f1_scores = []

        for user_id in range(0, num_users, batch_size):
            end = min(user_id + batch_size, num_users)
            user_input = np.array([i for i in range(user_id, end) for _ in range(num_items)])
            item_input = np.array(list(range(num_items)) * (end - user_id))

            predictions = ncf_model.predict(user_input, item_input)
            predictions = predictions.reshape((end - user_id), num_items)  # reshape to (number of users, number of items)

            for i in range(end - user_id):
                user_idx = user_id + i
                top_k_indices = predictions[i].argsort()[-top_k:][::-1]
                top_k_items = top_k_indices

                actual_items = self.test_positives.get(user_idx, [])
                hr = self.hit_rate(top_k_items, actual_items)
                ng = self.ndcg(top_k_items, actual_items)

                # Calculate binary vectors for precision, recall, and F1-score
                binary_predictions = np.isin(range(num_items), top_k_items).astype(int)
                binary_true = np.isin(range(num_items), actual_items).astype(int)

                precision = precision_score(binary_true, binary_predictions, zero_division=1)
                recall = recall_score(binary_true, binary_predictions, zero_division=1)
                f1 = f1_score(binary_true, binary_predictions, zero_division=1)

                hits.append(hr)
                ndcgs.append(ng)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)

        return np.mean(hits), np.mean(ndcgs), np.mean(precisions), np.mean(recalls), np.mean(f1_scores)

# NCF Model

# 모델 결합 및 실행 프로세스

class NCFModel:
    def __init__(self, num_users, num_items, num_features=10):
        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self.model = self.create_model()

    def create_model(self):
        user_input = Input(shape=(1,), name='user_input')
        item_input = Input(shape=(1,), name='item_input')

        user_embedding = Embedding(input_dim=self.num_users, output_dim=self.num_features, name='user_embedding')(user_input)
        item_embedding = Embedding(input_dim=self.num_items, output_dim=self.num_features, name='item_embedding')(item_input)

        user_vector = Flatten()(user_embedding)
        item_vector = Flatten()(item_embedding)

        concat = Concatenate()([user_vector, item_vector])

        dense = Dense(256, activation='relu')(concat)
        dropout = Dropout(0.3)(dense)
        final_layer = Dense(128, activation='relu')(dropout)
        output = Dense(1, activation='sigmoid')(final_layer)

        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, train_samples, batch_size=256, epochs=10):
        user_input = np.array(train_samples['user_index'])
        item_input = np.array(train_samples['item_index'])
        labels = np.array(train_samples['interaction'])
        self.model.fit([user_input, item_input], labels, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])

    def predict(self, user_input, item_input, batch_size=256):
        predictions = self.model.predict([np.array(user_input), np.array(item_input)], batch_size=batch_size)
        return predictions.flatten()

    def save(self, filepath):
        save_model(self.model, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        print(f"Loading model from {filepath}")
        model = load_model(filepath, custom_objects={'mse': MeanSquaredError})
        
        # 임베딩 레이어의 임베딩 차원 추출
        num_users = model.input[0].shape[1]
        num_items = model.input[1].shape[1]
        user_embedding_layer = model.get_layer('user_embedding')
        num_features = user_embedding_layer.output_dim  # or you can set it directly if known

        instance = cls(num_users, num_items, num_features)
        instance.model = model
        return instance

def find_clothes(df, gender, season, usage, color, model, user_index, top_n=3):
    filtered_clothes = df[(df['gender'] == gender) &
                          (df['season'] == season) &
                          (df['usage'] == usage) &
                          (df['baseColour'] == color)]

    if filtered_clothes.empty:
        return "No clothes found matching the criteria."

    item_indices = filtered_clothes['item_index'].tolist()
    user_indices = [user_index] * len(item_indices)
    predictions = model.predict(user_indices, item_indices, batch_size=512)
    
    # Copy the DataFrame to avoid SettingWithCopyWarning
    filtered_clothes = filtered_clothes.copy()
    filtered_clothes.loc[:, 'prediction'] = predictions  # Use .loc to set the values

    filtered_clothes = filtered_clothes.sort_values('prediction', ascending=False)
    top_clothes = filtered_clothes.head(top_n)

    count_vect = CountVectorizer(min_df=1, ngram_range=(1, 2))
    product_mat = count_vect.fit_transform(top_clothes['productDisplayName'])
    product_sim = cosine_similarity(product_mat)

    final_recommendations = []
    for sim_idx, row in enumerate(top_clothes.itertuples()):
        max_index = min(len(top_clothes), top_n + 1)
        similar_indices = product_sim[sim_idx].argsort()[::-1][1:max_index]
        similar_items = top_clothes.iloc[similar_indices]
        recommendation = {
            'product': row.productDisplayName,
            'most_similar': similar_items['productDisplayName'].tolist(),
            'prediction_score': row.prediction
        }
        final_recommendations.append(recommendation)

    return final_recommendations

def main():
    if len(sys.argv) > 1:  # sys.argv[0]는 스크립트의 파일명입니다.
        gender = sys.argv[1]
        season = sys.argv[2]
        usage = sys.argv[3]
        color = sys.argv[4]
    # else:
    #     gender = 'Women'
    #     season = 'Winter'
    #     usage = 'Casual'
    #     color = 'White'

    data_loader = DataLoader(user_model_path, clothes_data_path)
    train_data, test_data = data_loader.load_data()
    num_users = train_data['user_index'].nunique()
    num_items = train_data['item_index'].nunique()

    ncf = NCFModel(num_users, num_items)

    model_path = "C:/Users/disse/OneDrive/Desktop/styling/CODY_Model/model/ncf_model.h5"
    ncf.save(model_path)
    loaded_model = NCFModel.load(model_path)

    test_positives = {i: test_data[test_data['user_index'] == i]['item_index'].unique().tolist() for i in range(num_users)}
    metric = Metric(test_positives)
    # hit_rate, ndcg, precision, recall, f1score = metric.evaluate_model(loaded_model, num_users, num_items, top_k=10, batch_size=512)

    # print(f"Hit Rate: {hit_rate}, NDCG: {ndcg}")
    # print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1score}")
    print("모델 훈련 및 평가 완료")

    clothes = data_loader.item_data
    final_recommendations = find_clothes(clothes, gender, season, usage, color, model=loaded_model, user_index=0, top_n=3)
    # final_recommendations = find_clothes(clothes, gender = 'Women', season = 'Winter', usage = 'Casual', color = 'White', model=loaded_model, user_index=0, top_n=3)

    if isinstance(final_recommendations, str):
        print(final_recommendations)
    else:
        df = pd.DataFrame(final_recommendations)
        json_data = df.to_json(orient='records')
        with open('C:/Users/disse/OneDrive/Desktop/StylingByte/Assets/recommended_clothes_with_data.json', 'w') as f:
            f.write(json_data)

if __name__ == "__main__":
    main()
