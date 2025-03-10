import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

Dense = layers.Dense
BatchNormalization = layers.BatchNormalization
Dropout = layers.Dropout
import matplotlib.pyplot as plt

def home():
    st.title("Recommend Games Model Description")

    st.subheader("Purpose")
    st.write("The purpose of this project is to build a model that can recommend games to users based on their preferences.")

    st.subheader("Problem Type")
    st.write("This is a recommendation problem.")

    st.subheader("Source Data")
    st.write("this dataset come from https://www.kaggle.com/datasets/anandshaw2001/video-game-sales")

    st.subheader("Data")
    st.write("The dataset contains the following columns:")
    st.write("1. Name: The name of the game.")
    st.write("2. Platform: The platform of the game.")
    st.write("3. Year: The year the game was released.")
    st.write("4. Genre: The genre of the game.")
    st.write("5. Publisher: The publisher of the game.")
    st.write("6. NA_Sales: The sales of the game in North America.")
    st.write("7. EU_Sales: The sales of the game in Europe.")
    st.write("8. JP_Sales: The sales of the game in Japan.")
    st.write("9. Other_Sales: The sales of the game in other regions.")
    st.write("10. Global_Sales: The global sales of the game.")
    
    st.subheader("Data Cleaning")
    st.write("We will be cleaning the data by removing missing values and duplicates.")
    st.write("We will also be removing the columns that are not needed for the model.")

    st.subheader("Model")
    st.write("We will be building a Content-Based Filtering model and a Collaborative Filtering model.")
    st.write("The Content-Based Filtering model will recommend games based on the similarity of their features.")
    st.write("The Collaborative Filtering model will recommend games based on the ratings given by users.")
    st.write("We will also be building a Hybrid model that combines the scores of the Content-Based Filtering model and the Collaborative Filtering model.")

    st.subheader("Evaluation")
    st.write("We will be evaluating the models using Mean Cosine Similarity for the Content-Based Filtering model and Explained Variance for the Collaborative Filtering model.")

    st.subheader("Deployment")
    st.write("We will be deploying the model using Streamlit.")
    st.write("The user will be able to select a game and the model will recommend games based on the selected game.")    

def recommend():
    st.title("🎮Game Recommendation System")

    # โหลดข้อมูล
    df = pd.read_csv("vgsales.csv")

    # ทำความสะอาดข้อมูล
    df.dropna(subset=['Publisher'], inplace=True)
    df['Year'] = df['Year'].interpolate(method='linear')
    df["Year"] = df["Year"].astype(int)
    df = df.reset_index(drop=True)

    # สร้าง widget สำหรับเลือกเกม
    game = st.selectbox("🎯 Select a game", df["Name"].unique())

    # โมเดล Content-Based Filtering
    df['Combined_Features'] = df['Genre'] + " " + df['Platform'] + " " + df['Publisher']
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Combined_Features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    game_index = df[df['Name'] == game].index[0]
    similarity_scores = list(enumerate(cosine_sim[game_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]
    recommended_games_content = df.iloc[[i[0] for i in similarity_scores]][['Name', 'Platform', 'Genre', 'Publisher']]
    
    # โมเดล Collaborative Filtering (SVD)
    ratings = pd.DataFrame({
        'User': np.random.randint(1, 1000, size=len(df)),
        'Game': df['Name'],
        'Rating': np.random.randint(1, 6, size=len(df))
    })
    ratings = ratings.groupby(['User', 'Game']).mean().reset_index()
    ratings_pivot = ratings.pivot_table(index='User', columns='Game', values='Rating', aggfunc='mean').fillna(0)
    matrix = ratings_pivot.values
    user_means = np.mean(matrix, axis=1)
    matrix_demeaned = matrix - user_means.reshape(-1, 1)
    
    U, sigma, Vt = svds(matrix_demeaned, k=50)
    sigma = np.diag(sigma)
    predictions = np.dot(np.dot(U, sigma), Vt) + user_means.reshape(-1, 1)
    predictions_df = pd.DataFrame(predictions, index=ratings_pivot.index, columns=ratings_pivot.columns)
    
    if game in predictions_df.columns:
        recommended_games_svd = predictions_df[game].sort_values(ascending=False).index[:5].tolist()
        recommended_games_svd_df = df[df['Name'].isin(recommended_games_svd)][['Name', 'Platform', 'Genre', 'Publisher']]
    else:
        recommended_games_svd_df = pd.DataFrame()
    
    # Hybrid Model (Combine Scores)
    st.subheader("📌 Recommended Games")
    
    hybrid_scores = {}
    for i, row in recommended_games_content.iterrows():
        hybrid_scores[row['Name']] = hybrid_scores.get(row['Name'], 0) + 0.5  # 50% weight for Content-Based
    
    for i, row in recommended_games_svd_df.iterrows():
        hybrid_scores[row['Name']] = hybrid_scores.get(row['Name'], 0) + 0.5  # 50% weight for SVD
    
    hybrid_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    hybrid_recommendations_df = df[df['Name'].isin([game[0] for game in hybrid_recommendations])][['Name', 'Platform', 'Genre', 'Publisher']]
    st.dataframe(hybrid_recommendations_df)    
    
    st.write('---')

    # ประเมินโมเดล
    st.subheader("📊 Model Evaluation")
    mean_cosine_sim = np.mean([i[1] for i in similarity_scores])
    explained_variance = np.sum(sigma**2) / np.sum(np.var(matrix_demeaned, axis=0))
    
    st.write(f"✔️ Mean Cosine Similarity (Content-Based): {mean_cosine_sim:.4f}")
    st.write(f"✔️ Explained Variance (SVD): {explained_variance:.4f}")

    st.write('---')

def home2():
    st.title("Lung Disease Prediction Model Description")
    st.header("Purpose")
    st.write("""
    The purpose of this project is to build a model that can predict the likelihood of recovery from lung disease based on user data. 
    This model will help healthcare professionals assess the recovery chances of a patient based on a variety of features like age, 
    gender, lung capacity, smoking status, disease type, treatment type, and hospital visits.
    """)

    st.header("Problem Type")
    st.write("""
    This is a **binary classification** problem, where the model predicts the likelihood of recovery (Recovered or Not Recovered) 
    from lung disease based on the input data.
    """)

    st.header("Source Data")
    st.write("""
    The dataset used in this project comes from **Kaggle**:  
    [Lung Diseases Dataset](https://www.kaggle.com/datasets/samikshadalvi/lungs-diseases-dataset). 
    It contains several important attributes related to lung diseases and recovery outcomes, including:
    """)
    st.write("""
    - **Age**: The age of the person.
    - **Gender**: The male or female sex of the person.
    - **Lung Capacity**: The lung capacity of the person.
    - **Smoking Status**: Whether the person is a smoker or non-smoker.
    - **Disease Type**: The type of lung disease.
    - **Treatment Type**: The type of treatment received by the person.
    - **Hospital Visits**: The number of hospital visits made by the person.
    - **Recovered**: Whether the person recovered or not.
    """)

    st.header("Model")
    st.write("""
    A **Neural Network** will be built to predict the likelihood of recovery from lung disease based on the cleaned and processed user data. 
    This model is designed to handle complex relationships in the data and perform well with classification tasks.

    - **Neural Network Architecture**: 
      The model consists of multiple layers:
      - **Dense Layers**: Fully connected layers of neurons.
      - **Activation Functions**: **ReLU** is used in hidden layers, and **Sigmoid** is used in the output layer for probabilities between 0 and 1.
      - **Batch Normalization**: To ensure stable training and faster convergence.
      - **Dropout**: To prevent overfitting by randomly setting a fraction of the input units to zero during training.

    - **Optimizer**: The **Adam optimizer** is used, as it adapts the learning rate during training.

    - **Loss Function**: **Binary Cross-Entropy** loss function is used, as this is a binary classification task.
    """)


    st.header("Data Preparation")
    st.write("""
    Before using the data to train the model, we perform several steps to prepare and clean the data:

    1. **Missing Values**:
       - **Numerical Features** (e.g., Age, Lung Capacity, Hospital Visits): Missing values are replaced with the median of the respective columns.
       - **Categorical Features** (e.g., Gender, Smoking Status, Disease Type, Treatment Type, Recovered): Missing values are replaced with the mode of the respective columns.
       
    2. **Duplicates**: Any duplicate rows in the dataset are removed to prevent bias in the model's predictions.

    3. **Categorical Encoding**: Categorical variables are encoded using **Label Encoding**, which converts each category into a numerical value.

    4. **Data Normalization**: Numerical features like Age, Lung Capacity, and Hospital Visits are normalized using **StandardScaler** to ensure all features are on a similar scale, improving model performance.

    5. **Train-Test Split**: The dataset is divided into a **training set** (80%) and a **testing set** (20%) to evaluate the model's performance on unseen data.
    """)

    st.header("Model Development Process")
    st.write("""
    The development of the model will follow these steps:

    1. **Data Loading**: The dataset is loaded using **Pandas**.
    2. **Data Preprocessing**: This includes filling missing values, encoding categorical variables, and normalizing numerical features.
    3. **Model Design**: A neural network with **dense layers**, **batch normalization**, and **dropout** is defined using the **Keras** library.
    4. **Model Training**: The model is trained using the **train set** and evaluated on the **test set**.
    5. **Model Evaluation**: After training, we evaluate the model's **accuracy** and **loss** to assess its performance.
    """)

   
    st.header("Deployment")
    st.write("""
    After the model is trained and evaluated, it will be deployed using **Streamlit**, a Python library that allows the creation of web applications. 
    """)

def predrecover():
    st.title("Lung Disease Prediction Model")

    age = st.number_input("Enter Age", min_value=0, max_value=120, value=30)
    lung_capacity = st.number_input("Enter Lung Capacity", min_value=0.5, max_value=6.0, value=3.5)
    hospital_visits = st.number_input("Enter Number of Hospital Visits", min_value=0, max_value=100, value=3)

    gender = st.selectbox("Select Gender", ["Male", "Female"])
    smoking_status = st.selectbox("Select Smoking Status", ["Smoker", "Non-Smoker"])
    disease_type = st.selectbox("Select Disease Type", ["Asthma", "COPD", "Pneumonia"])
    treatment_type = st.selectbox("Select Treatment Type", ["Medication", "Therapy", "Surgery"])

    # Button to trigger model training
    if st.button("Start Model Training"):
        # Load the dataset
        df = pd.read_csv("lung_disease_data.csv")

        # Handle missing values
        num_cols = ["Age", "Lung Capacity", "Hospital Visits"]
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        cat_cols = ["Gender", "Smoking Status", "Disease Type", "Treatment Type", "Recovered"]
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

        # Encode categorical variables
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        # Separate features and target
        X = df.drop(columns=["Recovered"])
        y = df["Recovered"]

        # Normalize numerical features
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define model
        model = keras.Sequential([
          Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
          BatchNormalization(),
          Dropout(0.3),
          Dense(256, activation='relu'),
          BatchNormalization(),
          Dropout(0.3),
          Dense(128, activation='relu'),
          BatchNormalization(),
          Dropout(0.2),
          Dense(64, activation='relu'),
          Dropout(0.1),
          Dense(1, activation='sigmoid')
        ])

        # Compile model
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # Train the model
        with st.spinner('Training model... please wait'):
            history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2)

            # Evaluate the model
            train_loss, train_acc = model.evaluate(X_train, y_train)
            test_loss, test_acc = model.evaluate(X_test, y_test)

            # Display results in Streamlit
            st.write(f"**Training Accuracy:** {train_acc:.4f}")
            st.write(f"**Validation Accuracy:** {test_acc:.4f}")
            st.write(f"**Training Loss:** {train_loss:.4f}")
            st.write(f"**Validation Loss:** {test_loss:.4f}")

            # Plot accuracy and loss graphs
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            # Accuracy Plot
            ax[0].plot(history.history["accuracy"], label="Train Accuracy", color="blue")
            ax[0].plot(history.history["val_accuracy"], label="Validation Accuracy", color="orange")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Accuracy")
            ax[0].set_title("Model Accuracy")
            ax[0].legend()

            # Loss Plot
            ax[1].plot(history.history["loss"], label="Train Loss", color="blue")
            ax[1].plot(history.history["val_loss"], label="Validation Loss", color="orange")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Loss")
            ax[1].set_title("Model Loss")
            ax[1].legend()

            st.pyplot(fig)

            # User input data
            user_data = {
                "Age": age,
                "Lung Capacity": lung_capacity,
                "Hospital Visits": hospital_visits,
                "Gender": gender,
                "Smoking Status": smoking_status,
                "Disease Type": disease_type,
                "Treatment Type": treatment_type
            }

            # Encode user input as the model expects
            user_data_encoded = pd.DataFrame([user_data])
            for col in ["Gender", "Smoking Status", "Disease Type", "Treatment Type"]:
                le = label_encoders[col]
                # Check for unseen labels
                if not all(user_data_encoded[col].isin(le.classes_)):
                    unseen_labels = set(user_data_encoded[col]) - set(le.classes_)
                     
                    # Handle unseen labels by replacing with the first class
                    user_data_encoded[col] = le.transform(user_data_encoded[col].replace(unseen_labels, le.classes_[0]))
                else:
                    user_data_encoded[col] = le.transform(user_data_encoded[col])

            # Normalize numerical features
            user_data_encoded[num_cols] = scaler.transform(user_data_encoded[num_cols])

            # Predict the outcome (recovered or not)
            prediction = model.predict(user_data_encoded)
            if prediction[0][0] > 0.5:
                st.write("Prediction: The person is likely to recover.")
            else:
                st.write("Prediction: The person is unlikely to recover.")

def main():    
    page = st.sidebar.selectbox("Choose a page", ["Machine Model Description", "Recommend Games", "Neuron Network Description", "Lung Disease Prediction"])
    if page == "Machine Model Description":
        home()
    elif page == "Recommend Games":
        recommend()
    elif page == "Neuron Network Description":
        home2()
    elif page == "Lung Disease Prediction":
        predrecover()

if __name__ == "__main__":
    main()