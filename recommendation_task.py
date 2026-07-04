from sentence_transformers import SentenceTransformer, util
import json
import pandas as pd

# 1. Initialize the BGE-M3 model
print("Loading BAAI/bge-m3 for semantic classification, Please Wait...")
model = SentenceTransformer("BAAI/bge-m3")

restaurants_data = json.load(open("restaurants.json", "r+"))
dishes_data = json.load(open("dishes.json", "r+"))
users_and_preferences = json.load(open("users_and_preferences.json", "r+"))

dishes_names = list(dishes_data.keys())
dishes_descriptions = list(dishes_data.values())

# Pre-compute the embeddings for the dishes descriptions
print("Embedding dishes...")
dishes_embeddings = model.encode(dishes_descriptions, convert_to_tensor=True)


restaurant_names = list(restaurants_data.keys())
restaurant_descriptions = list(restaurants_data.values())

# Pre-compute the embeddings for the restaurants descriptions
print("Embedding restaurants...")
restaurants_embeddings = model.encode(restaurant_descriptions, convert_to_tensor=True)


def convert_to_ranked_csv(data_list, recommendation_class: str):
    # 1. Load into a DataFrame with specific column names
    df = pd.DataFrame(data_list, columns=[recommendation_class, 'Score'])
    
    df = df.sort_values(by='Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    df = df[['Rank', recommendation_class, 'Score']]
    ranked_csv = df.to_csv(index=False)
    return ranked_csv


def recommend_restaurants(preferences, restaurants):
    df = pd.json_normalize(preferences)
    preferences = df.to_csv(index=False)

    # Embed the incoming text
    preference_embedding = model.encode(preferences, convert_to_tensor=True)

    # Calculate cosine similarity between the preferences and restaurants data.
    scores = util.cos_sim(preference_embedding, restaurants_embeddings)[0]

    threshold = .5
    recommendations = [
        (restaurant_names[i], scores[i].item()) 
        for i in range(len(restaurant_names))
        if scores[i] > threshold # recommend if semantic similarity is greater than 50%
    ]

    # Ranking recommendations by semantic relationship (high to low).
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations




def recommend_dishes(preferences, dishes):
    df = pd.json_normalize(preferences)
    preferences = df.to_csv(index=False)

    # Embed the user's preferences.
    preference_embedding = model.encode(preferences, convert_to_tensor=True)

    # Calculate cosine similarity between the preferences and dishes data.
    scores = util.cos_sim(preference_embedding, dishes_embeddings)[0]

    threshold = .5
    recommendations = [
        (dishes_names[i], scores[i].item()) 
        for i in range(len(dishes_names))
        if scores[i] > threshold # Only recommend if semantic similarity is greater than 50%
    ]

    # Ranking recommendations by semantic relationship (high to low).
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations



# Generating 3 sample test-cases:
for user in users_and_preferences:
    print("Fetching recommendations for", user["Name"], "...")
    restaurant_recommendations = recommend_restaurants(
        user["Preferences"],
        restaurants_data
        )

    print("RESTAURANT RECOMMENDATIONS")
    print(convert_to_ranked_csv(restaurant_recommendations, "Restaurants"))
    dish_recommendations = recommend_dishes(
        user["Preferences"],
        dishes_data
    )

    print("DISH RECOMMENDATIONS")
    print(convert_to_ranked_csv(dish_recommendations, "Dishes"))