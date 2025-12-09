import pandas as pd
import numpy as np

class ManualPCA:
    def __init__(self, ratings_path, movies_path):
        # Carrega dados
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)
        
        # Matriz de Utilidade (User x Movie)
        self.user_movie_matrix = self.ratings.pivot(
            index='userId', columns='movieId', values='rating'
        ).fillna(0)
        
        self.user_ids = list(self.user_movie_matrix.index)
        self.movie_ids = list(self.user_movie_matrix.columns)
        
        # Variáveis do Modelo
        self.U_k = None 
        self.V_k = None 
        self.mean_ratings = None

    def fit(self, k=20):
        print("⚙️ Treinando PCA")
        A = self.user_movie_matrix.values
        
        # 1. Centraliza dados
        self.mean_ratings = np.mean(A, axis=0)
        A_centered = A - self.mean_ratings
        
        # 2. Covariância (A^T * A)
        C = np.dot(A_centered.T, A_centered)
        
        # 3. Autovalores/vetores (eigh para matriz simétrica)
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        
        # 4. Ordena e seleciona Top K
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.V_k = eigenvectors[:, sorted_indices][:, :k]
        
        # 5. Projeta usuários (Coeficientes)
        self.U_k = np.dot(A_centered, self.V_k)
        print(f"✅ Modelo treinado (k={k}).")

    def get_user_raw_data(self, user_id, limit=100):
        # Retorna dados brutos para análise do Agente
        if user_id not in self.user_ids: return None
        
        user_data = self.ratings[self.ratings['userId'] == user_id]
        merged = user_data.merge(self.movies, on='movieId')
        
        total = len(merged)
        media = merged['rating'].mean()
        # Ordena por nota para pegar os extremos
        amostra = merged.sort_values(by='rating', ascending=False).head(limit)
        
        lista = [f"| {row['title']} | Nota: {row['rating']} | {row['genres']} |" for _, row in amostra.iterrows()]
        return f"User {user_id} | Total: {total} | Média: {media:.2f}\n" + "\n".join(lista)

    def recommend(self, user_id, top_n=5):
        # Gera apenas a lista de filmes
        if user_id not in self.user_ids: return "Usuário não encontrado."
        
        user_idx = self.user_ids.index(user_id)
        c_vector = self.U_k[user_idx]
        
        # 1. Reconstrói notas (A ~ c * V^T + média)
        predicted_ratings = np.dot(c_vector, self.V_k.T) + self.mean_ratings
        
        # 2. Filtra filmes já vistos (seta para -1)
        user_original = self.user_movie_matrix.iloc[user_idx].values
        predicted_ratings[user_original > 0] = -1 
        
        # 3. Pega índices dos Top N
        top_indices = np.argsort(predicted_ratings)[::-1][:top_n]
        
        recommendations = []
        for idx in top_indices:
            real_id = self.movie_ids[idx]
            row = self.movies[self.movies['movieId'] == real_id]
            if not row.empty:
                recommendations.append(f"🎬 **{row['title'].values[0]}** (Score: {predicted_ratings[idx]:.2f})")
            
        return "\n".join(recommendations)