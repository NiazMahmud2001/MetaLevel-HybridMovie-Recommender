import pandas as pd
import numpy as np 
import joblib
import os 
import gradio as gd
import urllib.request


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# DataFramePreprocessor class is for loading the dataset and returning the rated and non-rated movies.
class DataFramePreprocessor():
    def __init__(self): 
        self.moviesDF = pd.read_csv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml-1m/movies.dat"), 
            delimiter="::", 
            encoding="latin-1" ,
            names=["MovieID", "Title", "Genres"],
            engine='python'
        )
        self.ratingsDF = pd.read_csv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml-1m/ratings.dat"), 
            delimiter="::", 
            encoding="latin-1" ,
            names=["UserID", "MovieID", "Rating", "Timestamp"],
            engine='python'
        )
        self.usersDF = pd.read_csv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),  "ml-1m/users.dat"), 
            delimiter="::", 
            encoding="latin-1" ,
            names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
            engine='python'
        )
        
    def get_AllDF(self):
        return {
            "moviesDF": self.moviesDF, 
            "ratingsDF": self.ratingsDF, 
            "usersDF": self.usersDF
        }
    
    def get_number_of_rated_movies(self):
        return len(self.ratingsDF.merge(self.moviesDF, on="MovieID"))

    def get_Rated_NonRated_Movies(self): 
        rated_movies = self.ratingsDF.merge(self.moviesDF, on="MovieID")
        rated_movies = self.moviesDF.loc[self.moviesDF["MovieID"].isin(rated_movies["MovieID"].unique())] 

        non_rated_movies = self.moviesDF.loc[~self.moviesDF["MovieID"].isin(rated_movies["MovieID"].unique())] 
        rated_movies["Title"] = rated_movies["Title"].str.lower()
        
        merged_movies = self.ratingsDF.merge(self.moviesDF, on="MovieID", how="left")
        
        return {
            "rated_movies": rated_movies, 
            "non_rated_movies": non_rated_movies,
            "merged_movies": merged_movies
        }
        
    def addNewRating(self, uid, movID, mov_rating): 
        checker = self.ratingsDF.loc[self.ratingsDF["UserID"]==uid]
        if len(checker.loc[checker["MovieID"]==movID])>0: 
            return False # means user has already 
        else: 
            newRating = pd.DataFrame({
                "UserID": [uid], 
                "MovieID": [movID], 
                "Rating": [mov_rating], 
                "Timestamp": [909999]
            })
            self.ratingsDF = pd.concat([self.ratingsDF, newRating], ignore_index=True)
            return True # means user has added new rating successfully




# LoadModels class is for loading the pre-trained models and returning them.
class LoadModels(): 
    def __init__(self): 
        self.svdModel = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainedModel/svd_model.pkl"))
        self.gradientModel = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainedModel/gradient_model.pkl"))
        self.svd_reader_trainset = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainedModel/svd_reader_trainset.pkl"))
    
    def getSVDModel(self):
        return {
            "svdModel": self.svdModel, 
            "svd_reader_trainset": self.svd_reader_trainset
        }
        
    def getGradientModel(self):
        return {
            "gradientModel": self.gradientModel
        }



class SVDModel(): 
    def __init__(self, svdModel, svd_reader_trainset, ratingMovie_MergedDF, non_rated_movies):
        # get vdModel, svd_reader_trainset -> from class LoadModels
        # get rated_movies -> from class DataFramePreprocessor
        self.svdModel = svdModel
        self.svd_reader_trainset = svd_reader_trainset
        self.ratingMovie_MergedDF = ratingMovie_MergedDF
        self.non_rated_movies = non_rated_movies
        
        
    def __private_get_Bias_feature_matrix(self):
        user_feature = self.svdModel.pu 
        item_feature = self.svdModel.qi
        user_bias = self.svdModel.bu
        item_bias = self.svdModel.bi
        
        return {
            "user_feature": user_feature, 
            "user_bias": user_bias,
            "item_feature": item_feature,
            "item_bias": item_bias  
        }
    
    
    def __private_calculate_Bias_feature_matrix(self): 
        modelFeatureBias = self.__private_get_Bias_feature_matrix()
        
        svd_userBias = []
        svd_itemBias = []
        user_vector = []
        item_vector = []
        
        # fill svd_userBias and user_vector
        for x in sorted(self.ratingMovie_MergedDF["UserID"].unique(), reverse=False):
            try: 
                user_inner_id = self.svd_reader_trainset.to_inner_uid(x)
                svd_userBias.append(modelFeatureBias["user_bias"][user_inner_id])
                user_vector.append(modelFeatureBias["user_feature"][user_inner_id])
            except ValueError as e: 
                svd_userBias.append(0.0)
                user_vector.append(np.zeros(self.svdModel.pu.shape[1]))
            
        # fill svd_itemBias and item_vector
        for x in sorted(self.ratingMovie_MergedDF["MovieID"].unique(), reverse=False): 
            try:
                movie_inner_id = self.svd_reader_trainset.to_inner_iid(x)
                svd_itemBias.append(modelFeatureBias["item_bias"][movie_inner_id])
                item_vector.append(modelFeatureBias["item_feature"][movie_inner_id])
            except ValueError as e: 
                svd_itemBias.append(0.0)
                item_vector.append(np.zeros(self.svdModel.qi.shape[1]))
            
        
        return {
            "svd_userBias": np.array(svd_userBias), 
            "svd_itemBias": np.array(svd_itemBias), 
            "user_vector": np.array(user_vector), 
            "item_vector": np.array(item_vector)
        }
        
    def final_SVD_Vector_Table_Calculator(self): 
        svdModelAllInfo = self.__private_calculate_Bias_feature_matrix()
        
        SVD_Final_Vector_table = svdModelAllInfo['user_vector'] @ svdModelAllInfo['item_vector'].T + svdModelAllInfo['svd_userBias'].reshape(-1, 1) + svdModelAllInfo['svd_itemBias'].reshape(1, -1) + self.svd_reader_trainset.global_mean
        SVD_rated_Vector_table = pd.DataFrame(SVD_Final_Vector_table , columns=sorted(self.ratingMovie_MergedDF["MovieID"].unique(), reverse=False)) 
        
        SVD_non_rated_Vector_table = np.full((len(svdModelAllInfo['user_vector']), len(self.non_rated_movies["MovieID"].unique())), self.svd_reader_trainset.global_mean)
        SVD_non_rated_Vector_table = pd.DataFrame(SVD_non_rated_Vector_table, columns=self.non_rated_movies["MovieID"].unique())

        SVD_Final_Vector_table = pd.concat([SVD_rated_Vector_table, SVD_non_rated_Vector_table], axis=1)
        
        return SVD_Final_Vector_table[sorted(SVD_Final_Vector_table.columns)]
    



class TFIDFModel(): 
    def __init__(self, moviesDF, ratingsDF, rated_movies, non_rated_movies): 
        self.moviesDF = moviesDF
        self.ratingsDF = ratingsDF
        self.rated_movies = rated_movies
        self.non_rated_movies = non_rated_movies
        
        self.prep_all_movies = ColumnTransformer(
            transformers=[
                ('genresTFIDF',  TfidfVectorizer(
                    lowercase=True,
                    stop_words='english',
                    ngram_range=(1,18),
                    max_df=0.9,
                    min_df=1,
                    max_features=500,
                    analyzer="word", 
                    token_pattern=r'[^|]+'
                ), 'Genres')
            ],
            remainder='drop'
        )
        self.prep_all_movies.fit(self.moviesDF)
        self.rated_tfidfTable = self.prep_all_movies.transform(self.rated_movies)
        self.non_rated_tfidfTable = self.prep_all_movies.transform(self.non_rated_movies)

        
    def __private_rated_Movies_TfIdf_Vectorizer(self): 
        rated_cosineSim = cosine_similarity(self.rated_tfidfTable, self.rated_tfidfTable) # value range: 0-1
        rated_tfidf_df =pd.DataFrame(rated_cosineSim, columns=self.rated_movies["MovieID"].to_numpy()) # shape -> (number of user , number of rated movies)
        return rated_tfidf_df
    
    
    def __private_nonRated_Movies_TfIdf_Vectorizer(self):         
        non_rated_cosineSim = cosine_similarity(self.non_rated_tfidfTable, self.rated_tfidfTable).T
        non_rated_tfidf_df = pd.DataFrame(non_rated_cosineSim, columns=self.non_rated_movies["MovieID"].to_numpy()) # shape -> (number of user , number of rated movies)
        return non_rated_tfidf_df
        
        
    def __private_combined_ratedNonRatedTFIDF_Vectorizer(self): 
        combinedPivoted = MinMaxScaler(feature_range=(0, 5))
        rated_tfidf_df = self.__private_rated_Movies_TfIdf_Vectorizer()
        non_rated_tfidf_df = self.__private_nonRated_Movies_TfIdf_Vectorizer()
        
        combined_tfidf_df = pd.concat([rated_tfidf_df, non_rated_tfidf_df] , axis=1)
        combined_tfidf_df = combined_tfidf_df[sorted(combined_tfidf_df.columns)]
        combined_tfidf_df_t = combinedPivoted.fit_transform(combined_tfidf_df)
        combined_tfidf_df = pd.DataFrame(combined_tfidf_df_t , columns=combined_tfidf_df.columns)
        return combined_tfidf_df
    
    def get_combined_tfidf_df(self): 
        return self.__private_combined_ratedNonRatedTFIDF_Vectorizer()
    
    def final_combnedTFIDF_Vectorizer(self): 
        combined_tfidf_df = self.__private_combined_ratedNonRatedTFIDF_Vectorizer()
        ratingsDFPivoted = self.ratingsDF.pivot(index="UserID", columns="MovieID", values="Rating").fillna(0)
        ratingsDFPivoted = (ratingsDFPivoted != 0).astype(int) # convert all rated movies to 1 and non-rated movies to 0

        comb_svd_tfIdf1 = pd.DataFrame(np.dot(ratingsDFPivoted, combined_tfidf_df), columns=combined_tfidf_df.columns)
        return comb_svd_tfIdf1





class MetaModelPrediction(): 
    def __init__(self, gradientModel, SVDTable, Tf_IdfTable, pure_tf_idf_table, ratingsDF, movieDF, numbOfRatedMovies): 
        self.metaModel = gradientModel  
        self.SVDArray = SVDTable
        self.Tf_IdfArray = Tf_IdfTable
        self.pure_tf_idf_table = pure_tf_idf_table
        self.ratingsDF = ratingsDF
        self.movieDF = movieDF
        self.numbOfRatedMovies = numbOfRatedMovies
        
        
        
    def __private_getTopN_Movie(self, UID, movieid, ret_n_movie): 
        ret_n_movie = ret_n_movie*10 if ret_n_movie<1000 else ret_n_movie
        
        rated_mov_by_uid = self.ratingsDF.loc[self.ratingsDF["UserID"]==UID]
        unrated_mov_by_uid = self.ratingsDF.loc[~self.ratingsDF["MovieID"].isin(rated_mov_by_uid["MovieID"])].drop(["Timestamp"], axis=1)["MovieID"].unique()    
        unrated_mov_by_uid = pd.DataFrame({
            "MovieID": unrated_mov_by_uid
        })
        
    
        svd_user_all_rated_movie = self.SVDArray.loc[UID-1].to_numpy()
        tfIdf_all_movies = self.Tf_IdfArray.loc[UID-1].to_numpy()
        svd_user_all_rated_movie= pd.DataFrame({
            "MovieID": [x for x in range(1, len(svd_user_all_rated_movie)+1)], 
            "Rating": svd_user_all_rated_movie
        })
        tfIdf_all_movies = pd.DataFrame({
            "MovieID": [x for x in range(1, len(tfIdf_all_movies)+1)], 
            "Rating": tfIdf_all_movies
        })
        
        pure_tfIdf = list(self.pure_tf_idf_table.loc[movieid])
        pure_tfIdf = pd.DataFrame({
            "MovieID": self.pure_tf_idf_table.columns, 
            "Pure_TfIdf_Rating": pure_tfIdf
        })
        
        # now ccmbine  "unrated_mov_by_uid", "svd_user_all_rated_movie" and "tfIdf_all_movies"
        merged_all = unrated_mov_by_uid.merge(svd_user_all_rated_movie , on=["MovieID"])
        merged_all = merged_all.merge(tfIdf_all_movies, on=["MovieID"])
        merged_all = merged_all.merge(pure_tfIdf, on=["MovieID"])
        merged_all = merged_all.rename(
            columns={"Rating_x": "SVD_Rating" , "Rating_y":"TfIdf_Rating"}
        )
        
        combinedScale = MinMaxScaler(feature_range=(0, 5))
        merged_all["TfIdf_Rating"] = combinedScale.fit_transform(merged_all[["TfIdf_Rating"]])
        
        if self.numbOfRatedMovies-len(unrated_mov_by_uid)>30: 
            # if user rated less that 30 movies then we mostly relay on Tf-IDF
            merged_all["Combined_Ratinng"] =  (0.2*merged_all["SVD_Rating"]) + (0.6*merged_all["TfIdf_Rating"]) + (0.2*merged_all["Pure_TfIdf_Rating"])
        else: 
            # now user has enough rated movies ... we ralay in mostly SVD
            merged_all["Combined_Ratinng"] =  (0.4*merged_all["SVD_Rating"]) + (0.2*merged_all["TfIdf_Rating"])+ (0.4*merged_all["Pure_TfIdf_Rating"])

        merged_all = merged_all.sort_values(["Combined_Ratinng"], ascending=False)
        return merged_all.iloc[0:ret_n_movie, :]
    
    
    def metaPredicton(self, UID, movieid, ret_n_movie):
        topN_df = self.__private_getTopN_Movie(UID, movieid, ret_n_movie)
        
        metaModel_pred = self.metaModel.predict(topN_df[["TfIdf_Rating", "SVD_Rating"]])
        
        topN_df["Meta_Model_Rating"] = metaModel_pred
        topN_df = topN_df.sort_values(["Meta_Model_Rating"], ascending=False)
        
        return topN_df.iloc[0:ret_n_movie, :].merge(self.movieDF, on="MovieID")
    
    

class GradioUi(): 
    def __init__(self):        
        self.allDFInit = DataFramePreprocessor()
        self.loadedModels = LoadModels()
        self.svdModel = SVDModel(
            svdModel=self.loadedModels.getSVDModel()["svdModel"],
            svd_reader_trainset=self.loadedModels.getSVDModel()["svd_reader_trainset"],
            ratingMovie_MergedDF = self.allDFInit.get_Rated_NonRated_Movies()["merged_movies"],
            non_rated_movies = self.allDFInit.get_Rated_NonRated_Movies()["non_rated_movies"]
        )
        self.ifidfModel = TFIDFModel(
            moviesDF=self.allDFInit.get_AllDF()["moviesDF"],
            ratingsDF=self.allDFInit.get_AllDF()["ratingsDF"],
            rated_movies=self.allDFInit.get_Rated_NonRated_Movies()["rated_movies"],
            non_rated_movies=self.allDFInit.get_Rated_NonRated_Movies()["non_rated_movies"]
        )    
        self.metaLevelModel = MetaModelPrediction(
            gradientModel = self.loadedModels.getGradientModel()["gradientModel"],
            SVDTable = self.svdModel.final_SVD_Vector_Table_Calculator(),
            Tf_IdfTable = self.ifidfModel.final_combnedTFIDF_Vectorizer(),
            pure_tf_idf_table= self.ifidfModel.get_combined_tfidf_df(),
            ratingsDF = self.allDFInit.get_AllDF()["ratingsDF"],
            movieDF = self.allDFInit.get_AllDF()["moviesDF"],
            numbOfRatedMovies = self.allDFInit.get_number_of_rated_movies()
        )
        
        self.moviesInfoDF = self.allDFInit.get_AllDF()["moviesDF"]
    
    def readMovieDF(self): 
        self.moviesInfoDF["Movie_Info"] = self.moviesInfoDF["MovieID"].astype(str) + " - " + self.moviesInfoDF["Title"] + " - " + self.moviesInfoDF["Genres"]
        return self.moviesInfoDF["Movie_Info"].tolist() 
    
    def processorFunc(self, uid, movieInfo, rating, num_rec_movies): 
        movieInfo = movieInfo.split(" - ")
        rating = int(rating)
        uid = int(uid)
        num_rec_movies = int(num_rec_movies)
        
        if uid == None or movieInfo == None: 
            return "Please enter a valid UserID and select a movie."
        else: 
            ratingResponse = "You have not added any rating yet"
            
            if int(rating)>0: 
                newRating = self.allDFInit.addNewRating(
                    uid= uid, 
                    movID= int(movieInfo[0].strip()), 
                    mov_rating = rating
                )
                if newRating: 
                    self.svdModel = SVDModel(
                        svdModel=self.loadedModels.getSVDModel()["svdModel"],
                        svd_reader_trainset=self.loadedModels.getSVDModel()["svd_reader_trainset"],
                        ratingMovie_MergedDF = self.allDFInit.get_Rated_NonRated_Movies()["merged_movies"],
                        non_rated_movies = self.allDFInit.get_Rated_NonRated_Movies()["non_rated_movies"]
                    )
                    self.ifidfModel = TFIDFModel(
                        moviesDF=self.allDFInit.get_AllDF()["moviesDF"],
                        ratingsDF=self.allDFInit.get_AllDF()["ratingsDF"],
                        rated_movies=self.allDFInit.get_Rated_NonRated_Movies()["rated_movies"],
                        non_rated_movies=self.allDFInit.get_Rated_NonRated_Movies()["non_rated_movies"]
                    )    
                    self.metaLevelModel = MetaModelPrediction(
                        gradientModel = self.loadedModels.getGradientModel()["gradientModel"],
                        SVDTable = self.svdModel.final_SVD_Vector_Table_Calculator(),
                        Tf_IdfTable = self.ifidfModel.final_combnedTFIDF_Vectorizer(),
                        pure_tf_idf_table= self.ifidfModel.get_combined_tfidf_df(),
                        ratingsDF = self.allDFInit.get_AllDF()["ratingsDF"],
                        movieDF = self.allDFInit.get_AllDF()["moviesDF"],
                        numbOfRatedMovies = self.allDFInit.get_number_of_rated_movies()
                    )
                    ratingResponse = "You have added new rating successfully"
                else: 
                    ratingResponse = "You have already rated this movie"
            
            predData = self.metaLevelModel.metaPredicton(UID=uid, movieid=int(movieInfo[0].strip()) , ret_n_movie=num_rec_movies)
            recommended_Movie = predData["MovieID"].tolist()
            meta_Model_Rating = predData["Meta_Model_Rating"].tolist()
            rec_movie_Title = predData["Title"].tolist()
            rec_movie_Genres = predData["Genres"].tolist()
            
            respp = ratingResponse + "\n\n"
            for movie, title, genre, expectedRating in zip(recommended_Movie, rec_movie_Title, rec_movie_Genres, meta_Model_Rating):
                respp += f"🎬 Movie: {movie}||  Title: {title}||  Genre: {genre}||  Expected Rating: {round(expectedRating)}\n\n"
                
            return respp
        
    def launchUI(self):
        demoUI = gd.Interface(
            fn=self.processorFunc,
            inputs=[
                gd.Textbox(label="UserID:", placeholder="Please Enter your User ID"),
                gd.Dropdown(label="Select a Movie", choices= self.readMovieDF()),
                gd.Slider(label="Give a Rating to Selected Movie:", minimum=0, maximum=5, step=1, value=0),
                gd.Slider(label="Number of Recommended movies: ", minimum=10, maximum=100, step=1, value=10)
            ],
            outputs=[gd.Textbox(label="Recommended Movies:", placeholder="recommended movies will be displayed here")]

        )
        # demoUI.launch()
        demoUI.launch(share=True, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__": 
    GradioUi().launchUI()




























