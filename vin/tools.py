import numpy as np
import pandas as pd 
import surprise as sp
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
from surprise import SVD, NMF
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import cross_validate
import pandas as pd
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV






def predic_user_vin_tool(id_user,id_vin):
    # nettoyage des données 
    #On prend que les colonnes qui nous intéresse
    parsed_data = pd.read_csv("static/vin/wine-reviews/winemag-data-130k-v2.csv")
    filtered_data = parsed_data[['country','province','region_1','variety','price','taster_name','points']]
    cleaned_data = filtered_data.rename(columns={'region_1': 'region'}).dropna(subset=['country','province','region','variety','taster_name','points'])

    # On trie les vins par pays,puis région,puis ville puis variété et enfin prix
    #On rajoute une colonne Id
    wines = cleaned_data.groupby(['country', 'province', 'region', 'variety']).agg({'price': 'mean'}).reset_index()
    wines = wines.assign(id=pd.Series(range(1, wines.shape[0]+1), dtype=int, index=wines.index))
    wines = wines[['id', 'country', 'province', 'region', 'variety', 'price']]

    users_all = cleaned_data.groupby('taster_name').count().reset_index()[['taster_name']]
    users_all = users_all.assign(id=pd.Series(range(1, users_all.shape[0]+1), dtype=int, index=users_all.index))

    # On lie les testeurs aux différents vins avec id
    wine_id_translator = {(row['country'], row['province'], row['region'], row['variety']): row['id'] for index, row in wines.iterrows()}
    user_id_translator = {row['taster_name']: row['id'] for index, row in users_all.iterrows()}

    def get_wine_id_series(data_frame):
        return pd.Series((wine_id_translator[(row['country'], row['province'], row['region'], row['variety'])] for _, row in data_frame.iterrows()), index=data_frame.index)

    def get_user_id_series(data_frame):
        return pd.Series((user_id_translator[row['taster_name']] for _, row in data_frame.iterrows()), index=data_frame.index)

    # On fait la moyenne des notes des utilisateurs qui ont jugés plusieurs fois le même vin
    ratings_all = cleaned_data.assign(wine_id=get_wine_id_series, user_id=get_user_id_series)[['taster_name', 'user_id', 'wine_id', 'points']].groupby(['user_id', 'taster_name', 'wine_id']).mean().reset_index()

    # classements des vins avec au moins 3 avis
    most_rated_wines = list(ratings_all.groupby(['wine_id']).count()[lambda x: x['points'] >= 3].reset_index()['wine_id'].values)

    ratings = ratings_all.loc[ratings_all['wine_id'].isin(most_rated_wines)].astype({'wine_id': int, 'user_id': int}).reset_index(drop=True)
    wines = wines.loc[wines['id'].isin(most_rated_wines)].astype({'id': int}).reset_index(drop=True)
    users = users_all.loc[users_all['id'].isin(ratings['user_id'].values)].astype({'id': int}).reset_index(drop=True)
    #entrer les données
    user=int(id_user)
    vin=int(id_vin)
    
    df3 = ratings[(ratings['user_id'] == user) & (ratings['wine_id'] == vin)]
    username=(df3["taster_name"].to_string(index=False))
    username=username[1:]
    print(username)
    user2=user-1
    
    # approche collaborative

    liste_result = []
    
    def predict_collab(ratings, taster_name, wine_id):
        
        is_target = (ratings['taster_name'] == taster_name) & (ratings['wine_id'] == wine_id)
        target = ratings[is_target].reset_index(drop=True)
        target = target.iloc[0]
        train_set = sp.Dataset.load_from_df(
            ratings[~is_target][['user_id', 'wine_id', 'points']], 
            sp.Reader(rating_scale=(0, 100))
        ).build_full_trainset()
    
        algo = sp.KNNBasic(verbose=False)
        algo.fit(train_set)
        prediction = algo.predict(target['user_id'], target['wine_id'], verbose=False)
        return prediction.est, prediction.est - target['points'], target['points']
    
    # Pour l'algorythme on ne prend que les données des 3 premières colonnes 
    
    reader = Reader(rating_scale=(1, 100))
    data = Dataset.load_from_df(ratings[['user_id', 'wine_id', 'points']], reader)
    algo=SVD()
    trainset = data.build_full_trainset()  
    algo.fit(trainset)
    testset = trainset.build_anti_testset()
    
    #On récupère les données dans ratings pour caculer l'erreur
    predictions = algo.test(testset)
    erreur=accuracy.rmse(predictions)
    tableau=pd.DataFrame(predictions)
    df2 = tableau[(tableau['uid'] == user2) & (tableau['iid'] == vin)]
    predratings=ratings[(ratings['user_id'] == user) & (ratings['wine_id'] == vin)]
    predratingsfloat=float(predratings["points"].to_string(index=False))
    pred_SVD=float(df2["est"].to_string(index=False))
    error_SVD = predratingsfloat-pred_SVD
    liste_result.append("{:.5f}".format(error_SVD))
    """
    #fonction de prédiction
    def get_all_predictions(prediction,n):
     
        
            # première prédiction pour chaque utilisateur
        similar_n = defaultdict(list)    
        for uid, iid, true_r, est, _ in prediction:
            similar_n[uid].append((iid, est))
    
        # prédictions général pour tout les utilisateurs 
        for uid, user_ratings in similar_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            similar_n[uid]=user_ratings[:n]
        return similar_n
    n=4
    pred_user=get_all_predictions(predictions,n)
    tmp = pd.DataFrame.from_dict(pred_user)
    tmp_transpose = tmp.transpose()
    
    #On choisit le pseudo à recommander 
    Pseudo= 1
    results_pred = tmp_transpose.loc[1]
    """
    # approche basé sur le contenu
    
    def predict_content(ratings, wines, taster_name, wine_id):
        
        user_ratings = ratings[ratings['taster_name'] == taster_name].join(wines.set_index('id'), on='wine_id')
        is_target = (user_ratings['wine_id'] == wine_id)
        
        features = pd.get_dummies(user_ratings.drop(columns=['points']))
        train_features = features[~is_target]
        target_features = features[is_target]
        
        encoder = LabelEncoder()
        train_labels = encoder.fit_transform(user_ratings[~is_target]['points'])
        target_label = user_ratings[is_target]['points'].iloc[0]
    
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(train_features, train_labels)
        prediction = encoder.inverse_transform(clf.predict(target_features))[0]
        return prediction, prediction - target_label, target_label
    
    #Fonction test avec le le nom du testeur et l'id du vin
    def test_classifier(taster_name, wine_id):
        
        pred_cf, error_cf, truth = predict_collab(ratings, taster_name, wine_id)
        pred_cn, error_cn, truth = predict_content(ratings, wines, taster_name, wine_id)
        liste_result.append("Résultat pour {} sur le vin numéro {}:".format(taster_name, wine_id))
        liste_result.append("{:.5f}".format(error_cf))
        liste_result.append("{:.5f}".format(error_cn))
        
    test_classifier(taster_name=username , wine_id=vin)


    

        
    def predict_hybrid(ratings, wines, taster_name, wine_id):
        
    #Si le vin possède plus de 3 notes on utilise la méthode collaborative
    #sinon la méthode sur le contenu 
        num_ratings = len(ratings[ratings['wine_id'] == wine_id])
        if num_ratings > 3:
            liste_result.append('approche collaborative utilisée')
            return predict_collab(ratings, taster_name, wine_id)
        else:
            liste_result.append('approche basé sur le contenu utilisée')
            return predict_content(ratings, wines, taster_name, wine_id)
    
    
    #test avec le le nom du testeur et l'id du vin 
    #affichage méthode hybride 
    #pred, error, truth = predict_hybrid(ratings, wines, taster_name=username, wine_id=vin)
    #print("approche hybride: \t prediction: {:.5f} \t erreur: {:.5f}\n".format(pred, error))
    
    pred, error, truth = predict_hybrid(ratings, wines, taster_name=username, wine_id=vin)

    liste_result.append("{:.5f}".format(error))

    return liste_result


def predic_user_top_tool(user_id):
    # nettoyage des données 
    #On prend que les colonnes qui nous intéresse
    parsed_data = pd.read_csv("static/vin/wine-reviews/winemag-data-130k-v2.csv")
    filtered_data = parsed_data[['country','province','region_1','variety','price','taster_name','points']]
    cleaned_data = filtered_data.rename(columns={'region_1': 'region'}).dropna(subset=['country','province','region','variety','taster_name','points'])

    # On trie les vins par pays,puis région,puis ville puis variété et enfin prix
    #On rajoute une colonne Id
    wines = cleaned_data.groupby(['country', 'province', 'region', 'variety']).agg({'price': 'mean'}).reset_index()
    wines = wines.assign(id=pd.Series(range(1, wines.shape[0]+1), dtype=int, index=wines.index))
    wines = wines[['id', 'country', 'province', 'region', 'variety', 'price']]

    users_all = cleaned_data.groupby('taster_name').count().reset_index()[['taster_name']]
    users_all = users_all.assign(id=pd.Series(range(1, users_all.shape[0]+1), dtype=int, index=users_all.index))

    # On lie les testeurs aux différents vins avec id
    wine_id_translator = {(row['country'], row['province'], row['region'], row['variety']): row['id'] for index, row in wines.iterrows()}
    user_id_translator = {row['taster_name']: row['id'] for index, row in users_all.iterrows()}

    def get_wine_id_series(data_frame):
        return pd.Series((wine_id_translator[(row['country'], row['province'], row['region'], row['variety'])] for _, row in data_frame.iterrows()), index=data_frame.index)

    def get_user_id_series(data_frame):
        return pd.Series((user_id_translator[row['taster_name']] for _, row in data_frame.iterrows()), index=data_frame.index)

    # On fait la moyenne des notes des utilisateurs qui ont jugés plusieurs fois le même vin
    ratings_all = cleaned_data.assign(wine_id=get_wine_id_series, user_id=get_user_id_series)[['taster_name', 'user_id', 'wine_id', 'points']].groupby(['user_id', 'taster_name', 'wine_id']).mean().reset_index()

    # classements des vins avec au moins 3 avis
    most_rated_wines = list(ratings_all.groupby(['wine_id']).count()[lambda x: x['points'] >= 3].reset_index()['wine_id'].values)

    ratings = ratings_all.loc[ratings_all['wine_id'].isin(most_rated_wines)].astype({'wine_id': int, 'user_id': int}).reset_index(drop=True)
    wines = wines.loc[wines['id'].isin(most_rated_wines)].astype({'id': int}).reset_index(drop=True)
    users = users_all.loc[users_all['id'].isin(ratings['user_id'].values)].astype({'id': int}).reset_index(drop=True)
    user=int(user_id)
    reader = Reader(rating_scale=(1, 100))
    data = Dataset.load_from_df(ratings[['user_id', 'wine_id', 'points']], reader)
    algo=SVD()
    trainset = data.build_full_trainset()  
    algo.fit(trainset)
    testset = trainset.build_anti_testset()
    
    
    #Predicting the ratings for testset
    predictions = algo.test(testset)
    erreur=accuracy.rmse(predictions)
      #fonction de prédiction
    def get_all_predictions(prediction,n):
     
        
            # première prédiction pour chaque utilisateur
        similar_n = defaultdict(list)    
        for uid, iid, true_r, est, _ in prediction:
            similar_n[uid].append((iid, est))
    
        # prédictions général pour tout les utilisateurs 
        for uid, user_ratings in similar_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            similar_n[uid]=user_ratings[:n]
        return similar_n
    n=5
    pred_user=get_all_predictions(predictions,n)
    tmp = pd.DataFrame.from_dict(pred_user)
    tmp_transpose = tmp.transpose()
    
    #On choisit le pseudo à recommander 
    results_pred = tmp_transpose.loc[user]
    return results_pred
    
