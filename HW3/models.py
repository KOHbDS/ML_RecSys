import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
import sys

warnings.filterwarnings('ignore')

class BaseRecommender:
    """Базовый класс для всех рекомендательных моделей"""
    
    def __init__(self, name="BaseRecommender"):
        self.name = name
        self.is_fitted = False
    
    def fit(self, train_data):
        """Обучение модели"""
        raise NotImplementedError
    
    def predict_score(self, user_id, item_id):
        """Предсказание скора для пары user-item"""
        raise NotImplementedError
    
    def predict_top_k(self, user_id, k=10, items_pool=None):
        """Предсказание top-k айтемов для пользователя"""
        raise NotImplementedError
    
    def batch_predict_top_k(self, user_ids, k=10, items_pool=None):
        """Пакетное предсказание top-k айтемов для списка пользователей"""
        result = {}
        for user_id in tqdm(user_ids, desc=f"{self.name} batch predict"):
            result[user_id] = self.predict_top_k(user_id, k, items_pool)
        return result

class PopularityRecommender(BaseRecommender):
    """Рекомендации на основе популярности айтемов"""
    
    def __init__(self, name="PopularityRecommender", weighted=True):
        super().__init__(name)
        self.weighted = weighted  # Взвешивать ли по проценту просмотра
        self.item_popularity = None
        self.all_items = None
        self.default_score = 0.0
    
    def fit(self, train_data):
        """
        Обучение модели популярности
        
        Parameters:
        -----------
        train_data : pandas.DataFrame
            Датафрейм с колонками user_id, item_id, watched_pct
        """
        print(f"Fitting {self.name}...")
        
        if self.weighted and 'watched_pct' in train_data.columns:
            popularity = train_data.groupby('item_id')['watched_pct'].agg(['count', 'mean'])
            popularity['score'] = popularity['count'] * popularity['mean'] / 100
        else:
            popularity = train_data.groupby('item_id').size().reset_index(name='score')
            popularity = popularity.set_index('item_id')

        max_score = popularity['score'].max()
        if max_score > 0:
            popularity['score'] = popularity['score'] / max_score
        
        self.item_popularity = popularity['score'].to_dict()
        self.all_items = list(self.item_popularity.keys())
        self.default_score = 1 / (max(self.item_popularity.values()) * 100) if self.item_popularity else 0.0
        self.is_fitted = True
        
        print(f"{self.name} fitted successfully.")
        return self
    
    def predict_score(self, user_id, item_id):
        """Предсказание скора для пары user-item"""
        if not self.is_fitted:
            raise Exception("Model is not fitted yet")
        
        return self.item_popularity.get(item_id, self.default_score)
    
    def predict_top_k(self, user_id, k=10, items_pool=None):
        """Предсказание top-k айтемов для пользователя"""
        if not self.is_fitted:
            raise Exception("Model is not fitted yet")
        
        items_to_consider = items_pool if items_pool is not None else self.all_items

        valid_items = [item for item in items_to_consider if item in self.item_popularity]

        top_items = sorted(valid_items, 
                           key=lambda x: self.item_popularity.get(x, self.default_score), 
                           reverse=True)[:k]
        
        return top_items

class MatrixFactorizationRecommender(BaseRecommender):
    """Рекомендации на основе матричной факторизации (SVD)"""
    
    def __init__(self, n_factors=100, name="MatrixFactorizationRecommender"):
        super().__init__(name)
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.global_mean = 0
        self.all_items = None
    
    def fit(self, train_data):
        """
        Обучение модели матричной факторизации
        
        Parameters:
        -----------
        train_data : pandas.DataFrame
            Датафрейм с колонками user_id, item_id, watched_pct
        """
        print(f"Fitting {self.name}...")

        unique_users = train_data['user_id'].unique()
        unique_items = train_data['item_id'].unique()
        
        self.user_mapping = {user: i for i, user in enumerate(unique_users)}
        self.item_mapping = {item: i for i, item in enumerate(unique_items)}
        self.reverse_user_mapping = {i: user for user, i in self.user_mapping.items()}
        self.reverse_item_mapping = {i: item for item, i in self.item_mapping.items()}
        self.all_items = list(unique_items)

        if 'watched_pct' in train_data.columns:
            ratings = train_data['watched_pct'].values / 100.0
        else:
            ratings = np.ones(len(train_data))
            
        self.global_mean = np.mean(ratings)
        
        user_indices = [self.user_mapping[user] for user in train_data['user_id']]
        item_indices = [self.item_mapping[item] for item in train_data['item_id']]
    
        matrix = csr_matrix((ratings, (user_indices, item_indices)), 
                            shape=(len(self.user_mapping), len(self.item_mapping)))

        svd = TruncatedSVD(n_components=self.n_factors, random_state=42)
        item_factors = svd.fit_transform(matrix.T)

        user_factors = matrix.dot(item_factors).dot(np.linalg.pinv(np.diag(svd.singular_values_)))

        self.user_factors = normalize(user_factors)
        self.item_factors = normalize(item_factors)
        
        self.is_fitted = True
        print(f"{self.name} fitted successfully.")
        return self
    
    def predict_score(self, user_id, item_id):
        """Предсказание скора для пары user-item"""
        if not self.is_fitted:
            raise Exception("Model is not fitted yet")

        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return self.global_mean
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[item_id]
        
        user_vec = self.user_factors[user_idx].reshape(1, -1)
        item_vec = self.item_factors[item_idx].reshape(1, -1)

        similarity = cosine_similarity(user_vec, item_vec)[0][0]

        score = (similarity + 1) / 2
        
        return score
    
    def predict_top_k(self, user_id, k=10, items_pool=None):
        """Предсказание top-k айтемов для пользователя"""
        if not self.is_fitted:
            raise Exception("Model is not fitted yet")

        if user_id not in self.user_mapping:
            if items_pool is not None:
                return items_pool[:min(k, len(items_pool))]
            return self.all_items[:min(k, len(self.all_items))]
        
        items_to_consider = items_pool if items_pool is not None else self.all_items

        user_idx = self.user_mapping[user_id]
        user_vec = self.user_factors[user_idx].reshape(1, -1)

        item_scores = {}
        for item_id in items_to_consider:
            if item_id in self.item_mapping:
                item_idx = self.item_mapping[item_id]
                item_vec = self.item_factors[item_idx].reshape(1, -1)
                similarity = cosine_similarity(user_vec, item_vec)[0][0]
                score = (similarity + 1) / 2 
                item_scores[item_id] = score
            else:
                item_scores[item_id] = self.global_mean

        top_items = sorted(item_scores.keys(), key=lambda x: item_scores[x], reverse=True)[:k]
        
        return top_items

class NeuralRecommender(BaseRecommender):
    """Нейросетевая модель для рекомендаций"""
    
    class RecommenderDataset(Dataset):
        """Датасет для обучения нейросетевой модели"""
        def __init__(self, interactions, user_mapping, item_mapping, n_users, n_items):
            self.interactions = interactions
            self.user_mapping = user_mapping
            self.item_mapping = item_mapping
            self.n_users = n_users
            self.n_items = n_items
        
        def __len__(self):
            return len(self.interactions)
        
        def __getitem__(self, idx):
            interaction = self.interactions.iloc[idx]
            user_id = interaction['user_id']
            item_id = interaction['item_id']

            user_idx = self.user_mapping.get(user_id, 0)
            item_idx = self.item_mapping.get(item_id, 0)

            if 'watched_pct' in interaction:
                rating = min(interaction['watched_pct'] / 100.0, 1.0)
            else:
                rating = 1.0 
                
            return user_idx, item_idx, float(rating)
    
    class NeuralCF(nn.Module):
        """модель для коллаборативной фильтрации"""
        def __init__(self, n_users, n_items, embedding_dim=64, hidden_layers=[128, 64]):
            super().__init__()

            self.user_embedding = nn.Embedding(n_users, embedding_dim)
            self.item_embedding = nn.Embedding(n_items, embedding_dim)

            layers = []
            input_dim = 2 * embedding_dim 
            
            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.Dropout(0.2))
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, 1))
            layers.append(nn.Sigmoid()) 
            self.layers = nn.Sequential(*layers)
            self._init_weights()
        
        def _init_weights(self):
            """Инициализация весов модели"""
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
        
        def forward(self, user_indices, item_indices):
            user_embeds = self.user_embedding(user_indices)
            item_embeds = self.item_embedding(item_indices)
            x = torch.cat([user_embeds, item_embeds], dim=1)
            output = self.layers(x)
            
            return output.squeeze()
    
    def __init__(self, embedding_dim=64, hidden_layers=[128, 64], batch_size=1024, 
                 epochs=5, learning_rate=0.001, name="NeuralRecommender"):
        super().__init__(name)
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.model = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.all_items = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_mean = 0.0
    
    def fit(self, train_data):
        """
        Обучение модели
        
        Parameters:
        -----------
        train_data : pandas.DataFrame
            Датафрейм с колонками user_id, item_id, watched_pct
        """
        print(f"Fitting {self.name}...")
        print(f"Using device: {self.device}")

        unique_users = list(train_data['user_id'].unique())
        unique_items = list(train_data['item_id'].unique())
        
        self.user_mapping = {user: i+1 for i, user in enumerate(unique_users)}
        self.item_mapping = {item: i+1 for i, item in enumerate(unique_items)}
        self.reverse_user_mapping = {i: user for user, i in self.user_mapping.items()}
        self.reverse_item_mapping = {i: item for item, i in self.item_mapping.items()}
        self.all_items = unique_items
        
        if 'watched_pct' in train_data.columns:
            self.global_mean = train_data['watched_pct'].mean() / 100.0
        else:
            self.global_mean = 0.5

        n_users = len(self.user_mapping) + 1
        n_items = len(self.item_mapping) + 1
        
        dataset = self.RecommenderDataset(
            train_data, self.user_mapping, self.item_mapping, n_users, n_items
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0 if 'ipykernel' in sys.modules else 4,
            persistent_workers=False
        )

        self.model = self.NeuralCF(
            n_users, n_items, self.embedding_dim, self.hidden_layers
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                user_indices, item_indices, ratings = batch
                user_indices = user_indices.to(self.device)
                item_indices = item_indices.to(self.device)
                ratings = ratings.to(self.device).float()
                optimizer.zero_grad()

                outputs = self.model(user_indices, item_indices)
                loss = criterion(outputs, ratings)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        print(f"{self.name} fitted successfully.")
        return self
    
    def predict_score(self, user_id, item_id):
        """Предсказание скора для пары user-item"""
        if not self.is_fitted:
            raise Exception("Model is not fitted yet")
        self.model.eval()
        user_idx = self.user_mapping.get(user_id, 0)
        item_idx = self.item_mapping.get(item_id, 0)

        if user_idx == 0 or item_idx == 0:
            return self.global_mean

        user_tensor = torch.tensor([user_idx]).to(self.device)
        item_tensor = torch.tensor([item_idx]).to(self.device)

        with torch.no_grad():
            score = self.model(user_tensor, item_tensor).item()
        
        return score
    
    def predict_top_k(self, user_id, k=10, items_pool=None):
        """Предсказание top-k айтемов для пользователя"""
        if not self.is_fitted:
            raise Exception("Model is not fitted yet")

        self.model.eval()
        user_idx = self.user_mapping.get(user_id, 0)

        if user_idx == 0:
            items_to_return = items_pool if items_pool is not None else self.all_items
            return items_to_return[:min(k, len(items_to_return))]
        
        items_to_consider = items_pool if items_pool is not None else self.all_items

        user_indices = torch.tensor([user_idx] * len(items_to_consider)).to(self.device)
        item_indices = torch.tensor([self.item_mapping.get(item, 0) for item in items_to_consider]).to(self.device)

        with torch.no_grad():
            scores = self.model(user_indices, item_indices).cpu().numpy()

        item_scores = {item: score for item, score in zip(items_to_consider, scores)}
        top_items = sorted(item_scores.keys(), key=lambda x: item_scores[x], reverse=True)[:k]
        
        return top_items


def calculate_diversity(recommendations, item_features, feature_column='genres'):
    """
    Рассчитывает разнообразие рекомендаций на основе указанной характеристики айтемов.
    
    Parameters:
    -----------
    recommendations : list
        Список рекомендованных айтемов
    item_features : pandas.DataFrame
        Датафрейм с характеристиками айтемов
    feature_column : str
        Название колонки с характеристикой для оценки разнообразия
    
    Returns:
    --------
    float
        Значение разнообразия от 0 до 1
    """
    if not recommendations or len(recommendations) < 2:
        return 0.0

    features = []
    for item_id in recommendations:
        if item_id in item_features.index:
            feature_value = item_features.loc[item_id, feature_column]
            if isinstance(feature_value, str):
                item_features_list = [f.strip() for f in feature_value.split(',')]
                features.extend(item_features_list)
    
    if not features:
        return 0.0
    unique_features = set(features)
    diversity = len(unique_features) / len(features)
    
    return diversity

def calculate_rank_based_overlap(list1, list2, p=0.9):
    """
    Рассчитывает метрику Rank-Based Overlap между двумя списками рекомендаций.
    
    Parameters:
    -----------
    list1, list2 : list
        Списки рекомендованных айтемов
    p : float
        Параметр, контролирующий вес позиций в ранжировании (обычно от 0.9 до 1.0)
    
    Returns:
    --------
    float
        Значение RBO от 0 до 1, где 1 означает полное совпадение
    """
    if not list1 or not list2:
        return 0.0

    min_length = min(len(list1), len(list2))
    list1 = list1[:min_length]
    list2 = list2[:min_length]

    sum_overlap = 0.0
    for d in range(1, min_length + 1):
        set1 = set(list1[:d])
        set2 = set(list2[:d])
        overlap = len(set1.intersection(set2)) / d
        sum_overlap += p**(d-1) * overlap
    rbo = (1 - p) * sum_overlap
    
    return rbo
