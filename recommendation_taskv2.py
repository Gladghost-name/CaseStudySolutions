import torch
from sentence_transformers import SentenceTransformer, util
import json
import pandas as pd

# 1. Initialize the BGE-M3 model
print("Loading BAAI/bge-m3...")
model = SentenceTransformer("BAAI/bge-m3")

restaurants_data = json.load(open("recommendation_project/mixed.json", "r+"))

nigerian_restaurants = list(restaurants_data.keys())
contextualized_labels = list(restaurants_data.values())
# 4. Pre-compute the embeddings for your label descriptions
# You only need to do this once when your application starts.
print("Embedding contextual labels...")
label_embeddings = model.encode(contextualized_labels, convert_to_tensor=True)


users_and_preferences = [
    {
        "Name": "Tunde O.",
        "Preferences": {
            "Primary Cuisine Type": "Traditional Nigerian (Contemporary twist)",
            "Secondary Cuisine Type": "Pan-Asian",
            "Price Range": "₦12,000 – ₦25,000",
            "Past Orders": ["Smokey Party Jollof Rice with Grilled Peppered Turkey and Fried Plantain", 
                            "Seafood Okra Soup with Oatmeal Swallow", 
                            "Spicy Singapore Noodles with Shredded Beef", 
                            "Gourmet Asun"],
            "Dietary Restrictions": ["No Pork", "High Protein / Moderate Carb"],
            "Spice Tolerance": "Very High",
            "Ordering Channels": ["Chowdeck", "Bolt Food", "WhatsApp"],
            "Peak Ordering Times": ["Friday", "Sunday"],
            "Key Decision Drivers": ["Portion size to price ratio", "delivery speed"]
        }
    }
]

df = pd.json_normalize(users_and_preferences[0])
preferences = df.to_csv(index=False)

# # 6. Embed the incoming text
preference_embedding = model.encode(preferences, convert_to_tensor=True)

# 7. Calculate cosine similarity between the ticket and your contextual labels
# This returns a tensor of scores mapping to each label
scores = util.cos_sim(preference_embedding, label_embeddings)[0]

# 8. Sort and display the results
# Combine the bare names with their calculated scores
threshold = .5
recommendations = [
    (nigerian_restaurants[i], scores[i].item()) 
    for i in range(len(nigerian_restaurants))
    if scores[i] > threshold # if the model's confidence is above 50%
]

# Sort by highest score first
recommendations.sort(key=lambda x: x[1], reverse=True)

print("Classification Results (Ranked by Match Confidence):")
print("-" * 50)

print(recommendations)
for label, score in recommendations:
    if score > threshold:
        print(f"Label: {label.ljust(15)} | Confidence: {score:.4f}")

