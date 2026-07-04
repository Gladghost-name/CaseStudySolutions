from gliclass import GLiClassModel, ZeroShotClassificationWithChunkingPipeline
from transformers import AutoTokenizer

model = GLiClassModel.from_pretrained("knowledgator/gliclass-large-v3.0")
tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-large-v3.0")
gliclass_pipeline = ZeroShotClassificationWithChunkingPipeline(model, tokenizer, classification_type='multi-label', device='cpu')

text = """
Attribute,User Data
User ID / Name,"Tunde O. (Lekki, Lagos)"
Primary Cuisine Type,Traditional Nigerian (Contemporary twist) & Afro-Continental fusion
Secondary Cuisine Type,Pan-Asian (Strong preference for Nigerian-Chinese / Spicy Wok dishes)
Price Range,"Mid-to-High (₦12,000 – ₦25,000 per meal / roughly $8–$16 USD)"
Past Orders,*   Smokey Party Jollof Rice with Grilled Peppered Turkey and Fried Plantain (Dodo)*   Seafood Okra Soup with Oatmeal Swallow*   Spicy Singapore Noodles with Shredded Beef (Nigerian-Chinese style)*   Gourmet Asun (Spicy peppered goat meat) platter
Dietary Restrictions,*   No Pork (Religious/personal preference)*   High Protein / Moderate Carb (Leans towards fitness-conscious traditional eating)
Spice Tolerance,Very High (Standard Nigerian palette; expects heavy use of scotch bonnet/atarodo)
Ordering Channels,"Chowdeck (Primary), Bolt Food, and direct WhatsApp ordering for local kitchen caterers"
Peak Ordering Times,*   Fridays (Dinner / Late-night wind-down)*   Sundays (Family-style lunch)
Key Decision Drivers,"Portion size to price ratio (Value for money is critical, even at premium tiers), and delivery speed (expects food hot upon arrival)."
"""

nigerian_restaurants = [
    "NOK by Alara",
    "Yellow Chilli",
    "Terra Kulture Food Lounge",
    "Ile Eros",
    "Ofada Boy",
    "Z Kitchen",
    "SLoW Lagos",
    "Shiro Lagos",
    "Danfo Bistro",
    "Glover Court Suya",
    "Blackbell",
    "Craft Gourmet by Lou Baker",
    "The House Lagos",
    "RSVP Lagos",
    "Ocean Basket VI",
    "Izanagi",
    "La Veranda",
    "Cactus Restaurant",
    "Rora Lagos",
    "The Ona"
]


results = gliclass_pipeline(text, 
        nigerian_restaurants, 
        .2, 
        prompt="Recommend restaurants based on user's preference.")[0]
print(results)