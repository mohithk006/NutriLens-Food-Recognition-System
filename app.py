import os
import time
from flask import Flask, render_template, request, jsonify, session
from ultralytics import YOLO
import google.generativeai as genai

# Initialize Flask app
app = Flask(__name__, static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "nutrilens_secret_key_2025")
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours

# Initialize models
model = YOLO("yolov8n_custom.pt")

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ------------------------- Helper Functions -------------------------

def init_cart():
    """Initialize session-based shopping cart"""
    if 'cart' not in session:
        session['cart'] = []
    session.permanent = True

def handle_api_error(func):
    """Decorator for consistent API error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return jsonify({
                "error": f"Failed to process request: {str(e)}",
                "success": False
            }), 500
    wrapper.__name__ = func.__name__
    return wrapper

def detect_objects(image_path, confidence_threshold=0.3):
    """Detect objects in image using YOLO model"""
    results = model(image_path)
    detected_objects = []

    for result in results:
        names = result.names
        for box in result.boxes:
            class_id = int(box.cls.item())
            conf = float(box.conf.item())
            if conf >= confidence_threshold:
                class_name = names[class_id]
                detected_objects.append(class_name)

    return list(set(detected_objects))

def get_nutrition_info(food_item, portion_size):
    """Get comprehensive nutrition information for food item"""
    # Convert portion size to actual weight ranges
    portion_mapping = {
        "small serving (50 - 150 g)": "small serving (50-150g)",
        "medium serving (150 - 300 g)": "medium serving (150-300g)", 
        "large serving (300 - 500 g)": "large serving (300-500g)"
    }
    
    actual_portion = portion_mapping.get(portion_size, portion_size)
    
    prompt = (
        f"You are a nutritional scientist. For the given food item '{food_item}' with portion size '{actual_portion}', provide ONLY the following structured factual output in simple language. Avoid any extra explanations or formatting like quotes or chatbot language. Explain everything keeping in mind the response should be for indian-origin people. Calculate nutritional values specifically for a {actual_portion} portion. Follow this exact format for EACH food item:\n"
        f"\n"
        f"Food Item: {food_item}\n"
        f"Portion size: {actual_portion.capitalize()}\n"
        f"\n"
        f"Calories: <a number>\n"
        f"Protein: <a number>g\n"
        f"Carbohydrates: <a number>g\n"
        f"Sugars: <a number>g\n"
        f"Fat: <a number>g\n"
        f"Fiber: <a number>g\n"
        f"Calcium: <a number>mg\n"
        f"Iron: <a number>mg\n"
        f"Potassium: <a number>mg\n"
        f"Sodium: <a number>mg\n"
        f"Vitamin B12: <a number>mg\n"
        f"Vitamin A: <a number>IU\n"
        f"Magnesium: <a number>mg\n"
        f"Zinc: <a number>mg\n"
        f"\n"
        f"Nutritional benefits:\n"
        f"1. <benefit one>\n"
        f"2. <benefit two>\n"
        f"\n"
        f"Disadvantages:\n"
        f"1. <drawback one>\n"
        f"2. <drawback two>\n"
        f"\n"
        f"Better alternatives:\n"
        f"1. <alternative one>\n"
        f"2. <alternative two>\n"
        f"\n"
        f"Verdict: <overall recommendation in 1 line>\n"
        f"\n"
        f"Trivia: <one factual interesting trivia sentence about this food to let nutrition to be fun to the user>\n"
        f"\n"
        f"If the item is invalid or not a food, reply with: '{food_item} is not a recognizable food item.'"
    )
    response = gemini_model.generate_content(prompt)
    return response.text.strip() if response else "No information available."

def get_health_warnings(nutrition_data, food_item):
    """Generate health warnings based on nutritional values - only for unhealthy items"""
    prompt = (
        f"You are a health expert and nutritionist. Based on the following nutritional information for '{food_item}', "
        f"scientifically analyze if this food item has any concerning nutritional values that could be harmful to health. "
        f"IMPORTANT: Only provide warnings if the food item is genuinely unhealthy or has concerning nutritional levels. "
        f"For healthy items like fruits, vegetables, nuts, or balanced meals, respond with 'No major health concerns - this is a healthy food choice.' "
        f"Be scientific and specific in your assessment. Focus on values that might be risky for Indian population considering typical dietary patterns.\n\n"
        f"Nutritional Data:\n{nutrition_data}\n\n"
        f"Criteria for warnings:\n"
        f"Reason being whatever you find it cautious scientifically for the human in the long run looking at the given Nutritional Data\n"
        f"IMPORTANT: Provide ONLY 1-2 most critical warnings maximum. Keep each warning to one short line.\n"
        f"If warnings are needed, format as:\n"
        f"⚠️ (this is just an example) Elevated [XXX] - consume in moderation\n"
        f"Important:Respond: don't be pessissimistic but if the food item sounds healthy or if the food item is healthy for the general population as a whole respond - 'No major health concerns - this is a healthy food choice.'( Or dont respond, no response at all!)\n"
        f"Do not provide long explanations or multiple paragraphs. Maximum 2 small warning lines only."
    )
    response = gemini_model.generate_content(prompt)
    return response.text.strip() if response else ""

def get_comprehensive_diet_plan(user_data):
    """Enhanced diet planning function with cleaner formatting"""
    
    height_m = float(user_data.get('height', 170)) / 100
    weight_kg = float(user_data.get('weight', 70))
    bmi = weight_kg / (height_m ** 2)
    
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal weight"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"

    prompt = (
        f"You are an expert dietitian and nutritionist specializing in Indian dietary patterns. Create a comprehensive, personalized diet plan based on the following detailed profile:\n\n"
        f"Profile: {user_data.get('age')}Y {user_data.get('gender')} | BMI: {bmi:.1f} | {user_data.get('diet_type')} | {user_data.get('existing_conditions')} | {user_data.get('activity_level')} | {user_data.get('allergies')} | {user_data.get('fitness_goals')}\n\n"
        f"Generate EXACTLY in the below format (with no fluff and no extra information from the start):\n\n"
        f"🎯 PERSONALIZED DIET PLAN FOR {user_data.get('goal').upper()}\n"
        f"Profile: {user_data.get('age')}Y {user_data.get('gender')} | BMI: {bmi:.1f} | {user_data.get('diet_type')} Diet | {user_data.get('existing_conditions')}\n\n"
        f"📊 DAILY NUTRITIONAL TARGET:\n\n"
        f"* **Basal Metabolic Rate (BMR):** [calculate using formula give only the computed value]\n" 
        f"* **Total Daily Energy Expenditure (TDEE):** [BMR x activity factor] [give only the computed value]\n"
        f"* **Target Calories:** [specific number] calories\n"
        f"* **Protein:** [amount]g | **Carbs:** [amount]g | **Fat:** [amount]g\n\n"
        f"🍽️ MEAL PLAN ({user_data.get('diet_period').upper()}):\n\n"
        f"EARLY MORNING (6:00 AM):\n"
        f"* **Option 1:** [specific items with quantities and calories]\n"
        f"* **Option 2:** [alternative with quantities and calories]\n\n"
        f"BREAKFAST (8:00 AM):\n"
        f"* **Option 1:** [specific items with quantities and calories]\n"
        f"* **Option 2:** [alternative with quantities and calories]\n\n"
        f"MID-MORNING SNACK (10:30 AM):\n"
        f"* **Option 1:** [specific items with quantities and calories]\n"
        f"* **Option 2:** [alternative with quantities and calories]\n\n"
        f"LUNCH (1:00 PM):\n"
        f"* **Option 1:** [specific items with quantities and calories]\n"
        f"* **Option 2:** [alternative with quantities and calories]\n\n"
        f"EVENING SNACK (4:00 PM):\n"
        f"* **Option 1:** [specific items with quantities and calories]\n"
        f"* **Option 2:** [alternative with quantities and calories]\n\n"
        f"DINNER (7:30 PM):\n"
        f"* **Option 1:** [specific items with quantities and calories]\n"
        f"* **Option 2:** [alternative with quantities and calories]\n\n"
        f"🏥 THERAPEUTIC FOODS FOR {user_data.get('existing_conditions').upper()} [if existing_conditions is 'none' do not generate and move to next]:\n"
        f"* **[Food 1]:** [benefit explanation in simple one line sentence]\n"
        f"* **[Food 2]:** [benefit explanation in simple one line sentence]\n"
        f"* **[Food 3]:** [benefit explanation in simple one line sentence]\n\n"
        f"💧 HYDRATION & SUPPLEMENTS:\n"
        f"* **Water intake:** [amount] liters/day\n"
        f"* **Recommended supplements:** [list if needed]\n\n"
        f"⚠️ FOODS TO AVOID:\n"
        f"* **[Category 1]:** [specific foods to avoid]\n"
        f"* **[Category 2]:** [specific foods to avoid]\n\n"
        f"📝 ADDITIONAL TIPS:\n"
        f"* **[Tip 1]:** [practical advice in simple one line sentence]\n"
        f"* **[Tip 2]:** [practical advice in simple one line sentence]\n\n"
        f"Note: Adjust portions based on progress. Consult healthcare provider for medical conditions."
    )
    
    response = gemini_model.generate_content(prompt)
    return response.text.strip() if response else "Unable to generate diet plan. Please try again."

def get_recipe_info(food_name, portion_size="2 persons (150 - 300g)"):
    """Enhanced recipe function - short, sweet, and Indian-focused"""
    prompt = (
        f"You are an expert Indian chef. Create a simple, easy-to-follow recipe for '{food_name}' for {portion_size}. "
        f"Keep it short, sweet, and to the point. Focus on Indian cooking methods and ingredients easily available in India. "
        f"No fancy words or complicated techniques. Write for Indian home cooks.\n\n"
        f"Format your response EXACTLY like this:\n\n"
        f"Nutrition: Calories: [number] | Protein: [number]g | Carbs: [number]g | Fat: [number]g\n\n"
        f"Verdict: [One simple line about health benefits or concerns for Indians]\n\n"
        f"Ingredients:\n"
        f"- [item 1 with quantity in Indian measurements like cups, tablespoons]\n"
        f"- [item 2 with quantity]\n"
        f"- [continue for all ingredients]\n\n"
        f"Steps:\n"
        f"1. [First step]\n"
        f"2. [Second step]\n"
        f"3. [Continue with all steps - maximum 6-8 steps]\n\n"
        f"Keep the entire response under 200 words. Use simple language that any Indian home cook can understand."
    )
    
    response = gemini_model.generate_content(prompt)
    return response.text.strip() if response else "Recipe not available."

def get_alternative_recipe(original_food, portion_size="2 persons (150 - 300g)"):
    """Generate alternative healthier recipe for the given food"""
    prompt = (
        f"You are an expert Indian chef and nutritionist. The user has requested an alternative to '{original_food}'. "
        f"Suggest a healthier or different variation of this dish for {portion_size}. "
        f"Make it more nutritious while keeping it tasty and easy to make at home in India.\n\n"
        f"If the original food is already healthy, suggest a different cooking method or flavor variation. "
        f"If it's unhealthy, suggest a healthier version with better ingredients.\n\n"
        f"Format your response EXACTLY like this:\n\n"
        f"Alternative: [Name of the healthier/different version]\n\n"
        f"Nutrition: Calories: [number] | Protein: [number]g | Carbs: [number]g | Fat: [number]g\n\n"
        f"Verdict: [One line explaining why this alternative is better]\n\n"
        f"Ingredients:\n"
        f"- [item 1 with quantity in Indian measurements]\n"
        f"- [item 2 with quantity]\n"
        f"- [continue for all ingredients]\n\n"
        f"Steps:\n"
        f"1. [First step]\n"
        f"2. [Second step]\n"
        f"3. [Continue with all steps - maximum 6-8 steps]\n\n"
        f"Keep the entire response under 250 words. Focus on Indian ingredients and cooking methods."
    )
    
    response = gemini_model.generate_content(prompt)
    return response.text.strip() if response else "Alternative recipe not available."

def analyze_ingredients(ingredients_list, commodity_type):
    """Analyze ingredients and provide positives, negatives, verdict, and better alternatives"""
    prompt = (
        f"You are a food safety expert and certified nutritionist specializing in food ingredient analysis. "
        f"Analyze the following ingredients list for a {commodity_type} product and provide a detailed assessment.\n\n"
        f"Ingredients: {ingredients_list}\n"
        f"Product Type: {commodity_type}\n\n"
        f"Provide your analysis in EXACTLY this format:\n\n"
        f"POSITIVES:[Judge these ingredients for general audience as a whole]\n"
        f"• [Positive ingredient 1]: [Short explanation of why this ingredient is healthy/beneficial]\n"
        f"• [Positive ingredient 2]: [Short explanation of why this ingredient is healthy/beneficial]\n"
        f"• [Continue for all positive ingredients]\n\n"
        f"NEGATIVES:[Judge these ingredients for general audience as a whole]\n"
        f"• [Negative ingredient 1]: [Short explanation of why this ingredient is harmful/concerning]\n"
        f"• [Negative ingredient 2]: [Short explanation of why this ingredient is harmful/concerning]\n"
        f"• [Continue for all negative ingredients]\n\n"
        f"VERDICT:\n"
        f"Health Rating: [Must be avoided / Poor / Bad / Moderately Healthy / Healthy / Extremely Healthy]\n"
        f"Reason: [short, precise explanation of why this product received this rating and what consumers should know]\n\n"
        f"Ingredients to Avoid: [List the most concerning ingredients that should be avoided in this type of product]\n\n"
        f"BETTER ALTERNATIVE INGREDIENTS:\n"
        f"• [Alternative ingredient 1]: [Why this is a better choice]\n"
        f"• [Alternative ingredient 2]: [Why this is a better choice]\n"
        f"• [Continue with 3-4 better alternatives that would make this product healthier]\n\n"
        f"Keep explanations concise (1-2 lines maximum per ingredient). Focus on health impact for Indian consumers."
    )
    
    response = gemini_model.generate_content(prompt)
    return response.text.strip() if response else "Analysis not available."

def render_page_or_fallback(template_name, fallback_message):
    """Helper to render template or show fallback message"""
    try:
        return render_template(template_name)
    except:
        return f"<h1>{fallback_message}</h1>"

# ------------------------- Routes -------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/nutritional.html")
def nutritional():
    return render_page_or_fallback("nutritional.html", "Nutritional Content Page Coming Soon")

@app.route("/mealplan.html")
def mealplan():
    return render_page_or_fallback("mealplan.html", "Meal Plan Page Coming Soon")

@app.route("/recipe.html")
def recipe():
    # Try to serve from paste-2.txt first, then template
    try:
        with open("paste-2.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return render_page_or_fallback("recipe.html", "Recipe Page Coming Soon")

@app.route("/cart.html")
def cart():
    return render_page_or_fallback("cart.html", "Shopping Cart Page Coming Soon")

@app.route("/ingredients.html")
def ingredients():
    return render_page_or_fallback("ingredients.html", "Ingredients Page Coming Soon")

@app.route("/comingsoon.html")
def coming_soon():
    return "<h1>This feature is Coming Soon...</h1>"

# ------------------------- API Endpoints -------------------------

@app.route("/upload", methods=["POST"])
@handle_api_error
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Get portion size from form data
    portion_size = request.form.get("portion_size", "medium serving (150 - 300 g)")

    timestamp = str(int(time.time()))
    filename = f"uploaded_{timestamp}.jpg"
    img_path = os.path.join("static", filename)
    file.save(img_path)

    detected_foods = detect_objects(img_path)
    nutrition_data = []
    
    for food in detected_foods:
        nutrition_info = get_nutrition_info(food, portion_size)
        health_warnings = get_health_warnings(nutrition_info, food)
        nutrition_data.append({
            "food": food,
            "details": nutrition_info,
            "health_warnings": health_warnings
        })

    return jsonify({
        "detected": detected_foods,
        "nutrition": nutrition_data,
        "success": True
    })

@app.route("/get_nutrition", methods=["POST"])
@handle_api_error
def get_nutrition():
    data = request.get_json()
    food_item = data.get("food_name")
    portion_size = data.get("portion_size")

    if not food_item or not portion_size:
        return jsonify({"error": "Food name and portion size are required"}), 400

    nutrition_info = get_nutrition_info(food_item, portion_size)
    health_warnings = get_health_warnings(nutrition_info, food_item)
    
    return jsonify({
        "nutrition": [{
            "food": food_item,
            "details": nutrition_info,
            "health_warnings": health_warnings
        }],
        "success": True
    })

@app.route("/get_diet_plan", methods=["POST"])
@handle_api_error
def get_diet_plan():
    data = request.get_json()
    
    required_fields = ['age', 'gender', 'height', 'weight']
    missing_fields = [field for field in required_fields if not data.get(field)]
    
    if missing_fields:
        return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400
    
    # Prepare user data with defaults for optional fields
    user_data = {
        'age': data.get('age'),
        'gender': data.get('gender'),
        'height': data.get('height'),
        'weight': data.get('weight'),
        'activity_level': data.get('activity_level', 'Moderately active'),
        'goal': data.get('goal', 'Maintain health'),
        'diet_type': data.get('diet_type', 'Vegetarian'),
        'diet_period': data.get('diet_period', '1 week'),
        'existing_conditions': data.get('existing_conditions', 'none'),
        'allergies': data.get('allergies', 'None'),
        'fitness_goals': data.get('fitness_goals', 'Improved energy')
    }
    
    diet_plan = get_comprehensive_diet_plan(user_data)
    return jsonify({"diet_plan": diet_plan, "success": True})

@app.route("/get_recipe", methods=["POST"])
@handle_api_error
def get_recipe():
    data = request.get_json()
    food_name = data.get("food_name")
    portion_size = data.get("portion_size", "2 persons (150 - 300g)")

    if not food_name:
        return jsonify({"error": "Please enter a food name"}), 400

    recipe_text = get_recipe_info(food_name, portion_size)
    return jsonify({
        "food": food_name,
        "recipe": recipe_text,
        "portion_size": portion_size,
        "success": True
    })

@app.route("/suggest_alternative", methods=["POST"])
@handle_api_error
def suggest_alternative():
    """NEW ENDPOINT: Suggest alternative recipe"""
    data = request.get_json()
    original_food = data.get("original_food")
    portion_size = data.get("portion_size", "2 persons (150 - 300g)")

    if not original_food:
        return jsonify({"error": "Original food name is required"}), 400

    alternative_recipe = get_alternative_recipe(original_food, portion_size)
    
    # Extract alternative name from the response
    alternative_name = "Alternative Recipe"
    if alternative_recipe.startswith("Alternative:"):
        lines = alternative_recipe.split('\n')
        if lines and lines[0].startswith("Alternative:"):
            alternative_name = lines[0].replace("Alternative:", "").strip()
    
    return jsonify({
        "success": True,
        "alternative_recipe": alternative_recipe,
        "alternative_name": alternative_name,
        "original_food": original_food,
        "portion_size": portion_size
    })

@app.route("/analyze_ingredients", methods=["POST"])
@handle_api_error
def analyze_ingredients_endpoint():
    data = request.get_json()
    ingredients_list = data.get("ingredients")
    commodity_type = data.get("commodity_type")

    if not ingredients_list or not commodity_type:
        return jsonify({"error": "Both ingredients list and commodity type are required"}), 400

    analysis_result = analyze_ingredients(ingredients_list, commodity_type)
    return jsonify({
        "success": True,
        "analysis": analysis_result,
        "ingredients": ingredients_list,
        "commodity_type": commodity_type
    })

# ------------------------- Cart Endpoints -------------------------

@app.route("/add_to_cart", methods=["POST"])
@handle_api_error
def add_to_cart():
    init_cart()
    
    data = request.get_json()
    food_name = data.get("food_name")
    portion_size = data.get("portion_size")

    if not food_name:
        return jsonify({"error": "Food name is required"}), 400

    # Get recipe to extract ingredients
    recipe_text = get_recipe_info(food_name, portion_size)
    
    # Parse ingredients from recipe - PRESERVE FULL TEXT INCLUDING QUANTITIES
    ingredients = []
    lines = recipe_text.split('\n')
    in_ingredients_section = False
    
    for line in lines:
        line = line.strip()
        if line.lower().startswith('ingredients:'):
            in_ingredients_section = True
            continue
        elif line.lower().startswith('steps:'):
            in_ingredients_section = False
            break
        elif in_ingredients_section and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
            # ONLY remove the bullet point symbol, keep everything else including quantities
            ingredient = line[1:].strip() if line.startswith('-') else line[1:].strip() if line.startswith('•') else line[1:].strip() if line.startswith('*') else line
            if ingredient and len(ingredient) > 1:
                ingredients.append(ingredient)
    
    # Add ingredients to cart with full details preserved
    added_count = 0
    for ingredient in ingredients:
        # Simple duplicate check based on first word only
        first_word = ingredient.split()[0].lower() if ingredient.split() else ""
        existing_item = next((item for item in session['cart'] 
                            if item['name'].split()[0].lower() == first_word), None)
        
        if not existing_item:
            session['cart'].append({
                "name": ingredient,  # Keep full ingredient text with quantities
                "quantity": "As per recipe",
                "source": f"Recipe: {food_name}",
                "timestamp": time.time()
            })
            added_count += 1
    
    session.modified = True
    
    return jsonify({
        "success": True,
        "message": f"Added {added_count} ingredients from {food_name} to cart",
        "ingredients_count": len(ingredients)
    })

@app.route("/get_cart", methods=["GET"])
def get_cart():
    init_cart()
    return jsonify({
        "success": True,
        "cart": session.get('cart', []),
        "total_items": len(session.get('cart', []))
    })

@app.route("/add_item_to_cart", methods=["POST"])
@handle_api_error
def add_item_to_cart():
    init_cart()
    
    data = request.get_json()
    item_name = data.get("item_name", "").strip()
    item_quantity = data.get("item_quantity", "").strip()

    if not item_name:
        return jsonify({"error": "Item name is required"}), 400

    # Check if item already exists
    existing_item = next((item for item in session['cart'] if item['name'].lower() == item_name.lower()), None)
    if existing_item:
        existing_item['quantity'] = item_quantity
        existing_item['timestamp'] = time.time()
    else:
        session['cart'].append({
            "name": item_name,
            "quantity": item_quantity,
            "source": "Manual Entry",
            "timestamp": time.time()
        })
    
    session.modified = True
    
    return jsonify({
        "success": True,
        "message": f"{item_name} added to cart"
    })

@app.route("/remove_item_from_cart", methods=["POST"])
@handle_api_error
def remove_item_from_cart():
    init_cart()
    
    data = request.get_json()
    item_index = data.get("item_index")

    if item_index is None or item_index < 0:
        return jsonify({"error": "Valid item index is required"}), 400

    if item_index >= len(session['cart']):
        return jsonify({"error": "Item not found"}), 404
    
    removed_item = session['cart'].pop(item_index)
    session.modified = True
    
    return jsonify({
        "success": True,
        "message": f"{removed_item['name']} removed from cart"
    })

@app.route("/update_item_quantity", methods=["POST"])
@handle_api_error
def update_item_quantity():
    init_cart()
    
    data = request.get_json()
    item_index = data.get("item_index")
    new_quantity = data.get("quantity")

    if item_index is None or item_index < 0:
        return jsonify({"error": "Valid item index is required"}), 400
    
    if new_quantity is None or str(new_quantity).strip() == "":
        return jsonify({"error": "Quantity is required"}), 400

    if item_index >= len(session['cart']):
        return jsonify({"error": "Item not found"}), 404
    
    session['cart'][item_index]['quantity'] = str(new_quantity).strip()
    session.modified = True
    
    return jsonify({
        "success": True,
        "message": "Quantity updated"
    })

@app.route("/clear_cart", methods=["POST"])
@handle_api_error
def clear_cart():
    init_cart()
    session['cart'] = []
    session.modified = True
    return jsonify({
        "success": True,
        "message": "Cart cleared successfully"
    })

# ------------------------- App Runner -------------------------

# if __name__ == "__main__":
#     app.run()


