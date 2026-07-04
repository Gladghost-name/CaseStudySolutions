# pip install huggingface_hub numpy
from huggingface_hub import InferenceClient
import numpy as np
import json

# 1. Initialize client with your free Hugging Face API key
client = InferenceClient(token="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")


restaurants_data = json.load(open("recommendation_project/mixed.json", "r+"))

nigerian_restaurants = list(restaurants_data.keys())
contextualized_labels = list(restaurants_data.values())

# 2. Define data
tunde_query = """
Primary Cuisine Type,"Upscale Continental, Premium Steaks, & Global Contemporary"
Secondary Cuisine Type,"Luxury Fine-Dining Pan-Asian (Sushi, Sashimi, Curries)"
Price Range,"High-End (₦35,000 – ₦70,000+ per meal / roughly $22–$45+ USD)"
Past Orders,* Wood-Fired Ribeye Steak with Truffle Fries* Premium Salmon Sashimi & Dragon Sushi Rolls* Slow-cooked Seafood Risotto* Artisanal Botanical Cocktails
Dietary Restrictions,"* Low Carb / Keto-leaning (Prioritizes high-quality fats and clean proteins)* Avoids heavy, oily local street food processing"
Spice Tolerance,"Low-to-Medium (Prefers complex herb crusts, garlic butter, and mild umami reductions over raw scotch bonnet heat)"
Ordering Channels,"Dine-in preferred for the ambiance, but uses Chowdeck Premium or direct dispatch for boardroom/office delivery."
Peak Ordering Times,* Wednesday/Thursday (Corporate client dinners & mid-week networking)* Saturday Night (Intimate social dates)
Key Decision Drivers,"Atmosphere & Presentation (Needs sleek aesthetics, flawless presentation, a curated wine list, and premium ingredients like dry-aged beef)." 
"""


# restaurant_labels = [
#     "Ofada Boy: Traditional street food heritage, native rice in leaves, spicy goat meat asun, atarodo pepper.",
#     "Blackbell: Fast-casual multi-cuisine restaurant, cheap everyday staples, mass-market standard spice."
# ]

# 3. Call the free serverless endpoint to get embeddings
print("Fetching embeddings from Hugging Face...")
ticket_embedding = client.feature_extraction(tunde_query, model="BAAI/bge-m3")
label_embeddings = [client.feature_extraction(lbl, model="BAAI/bge-m3") for lbl in contextualized_labels]

# 4. Perform the Cosine Similarity locally using basic numpy math
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("\nServerless Classification Results:")
print("-" * 40)
for i, lbl_vec in enumerate(label_embeddings):
    score = cosine_similarity(ticket_embedding, lbl_vec)
    print(f"Label: {nigerian_restaurants[i][:15]}... | Score: {score:.4f}")