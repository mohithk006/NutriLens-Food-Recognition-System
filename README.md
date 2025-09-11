# NutriLens: Your AI-Powered Food Intelligence Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.x-black.svg)](https://flask.palletsprojects.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blueviolet.svg)](https://ultralytics.com/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini_API-red.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Detect. Analyze. Eat Smart. NutriLens is an end-to-end web application that leverages the power of Artificial Intelligence to provide instantaneous food recognition, detailed nutritional analysis, and personalized dietary guidance. Simply upload a picture of your meal, and let NutriLens do the rest!



---

## 📋 Table of Contents

- [Introduction](#-introduction)
- [✨ Key Features](#-key-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Setup](#installation--setup)
- [📖 User Guide](#-user-guide)
- [📈 Project Analysis](#-project-analysis)
  - [Pros](#pros)
  - [Cons & Future Improvements](#cons--future-improvements)
- [🤝 Contributors](#-contributors)
- [📜 License](#-license)

---

## Introduction

In a world with countless dietary choices, making informed decisions about nutrition can be overwhelming. NutriLens was built to solve this problem by acting as a personal AI nutritionist in your pocket. It combines state-of-the-art object detection with advanced generative AI to deliver a seamless and intuitive user experience.

This project is not just a concept; it's a fully functional, end-to-end deployed application that showcases a deep understanding of frontend development, backend architecture, API integration, and machine learning model implementation. The core of the application is a **custom-trained YOLOv8 model** designed to recognize a wide variety of food items, with a special focus on Indian cuisine.

---

## Key Features

NutriLens is packed with features designed to empower users on their health journey:

-   **📸 AI-Powered Food Recognition**: Upload an image of your meal, and our custom-trained YOLOv8 model instantly identifies the food items with high accuracy. You can also get nutritional data by manually typing the food name.
-   **🔬 Detailed Nutritional Analysis**: Get a comprehensive breakdown of your meal, including calories, macronutrients (protein, carbs, fat), vitamins, and minerals. The analysis also includes health benefits, potential disadvantages, and fun trivia about the food.
-   **🥗 Personalized Diet Planning**: Generate a complete diet plan customized to your age, gender, height, weight, activity level, dietary preferences, and fitness goals. The plan includes daily nutritional targets, meal options, and hydration tips.
-   **🧪 Ingredient Safety Analysis**: Paste the ingredients list from any packaged food item to receive an AI-driven analysis of its contents. The report highlights positive and negative ingredients and provides a final health verdict with suggestions for healthier alternatives.
-   **🍲 Recipe Generation & Alternatives**: Discover new recipes by simply entering the name of a dish. For any given recipe, you can also request a healthier, alternative version with just one click.
-   **🛒 Automated Shopping Cart**: Automatically add all ingredients from a generated recipe to a dynamic shopping list. You can also manually add, edit, and remove items, and copy the entire list to your clipboard.

---

## Tech Stack

This project integrates a variety of modern technologies to deliver a robust and feature-rich experience.

| Category          | Technology / Library                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------ |
| **Backend** | `Python`, `Flask`                                                                                      |
| **AI / ML** | `Google Gemini API` (for generative content), `YOLOv8 (Ultralytics)`, `PyTorch`, `OpenCV` |
| **Frontend** | `HTML5`, `CSS3`, `Vanilla JavaScript`                                                                  |
| **Data/Session** | `Flask Session` (for Shopping Cart)                                                                    |
| **Deployment** | `Gunicorn` (as specified in `requirements.txt`)                                                 |

---

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

-   Python 3.10 or higher
-   `pip` (Python package installer)
-   Git version control

### Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/NutriLens.git](https://github.com/your-username/NutriLens.git)
    cd NutriLens
    ```

2.  **Create a Virtual Environment**
    It's highly recommended to create a virtual environment to manage project dependencies.
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    All the required Python packages are listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    This application requires API keys to function. Create a file named `.env` in the root directory of the project.
    Open the `.env` file and add the following lines, replacing `your_api_key_here` with your actual keys:

    ```env
    GEMINI_API_KEY="your_google_gemini_api_key_here"
    FLASK_SECRET_KEY="any_strong_random_secret_key_here"
    ```
    > **Important**: The `GEMINI_API_KEY` is essential for all generative AI features. The `FLASK_SECRET_KEY` is used to secure the user session for the shopping cart.

5.  **Run the Flask Application**
    Once the dependencies are installed and environment variables are set, you can start the Flask server.
    ```bash
    flask run
    ```
    The application will be running at `http://127.0.0.1:5000`. Open this URL in your web browser.

---

## User Guide

Navigating NutriLens is simple and intuitive.

1.  **Homepage**: The homepage provides an overview of all the features. You can navigate to any feature from the "Explore" section.
2.  **Nutritional Content**:
    -   Click on "Nutritional Content".
    -   Either upload an image of your food or type the name of a food item into the text box.
    -   Select a portion size and click "Get Info" to see a detailed nutritional report.
3.  **Meal Plan**:
    -   Navigate to the "Meal Plan" page.
    -   Fill out the form with your personal details, lifestyle choices, and health goals.
    -   Click "Generate Personalized Diet Plan" to receive a comprehensive plan tailored to you.
4.  **Ingredients Analysis**:
    -   Go to the "Ingredients" page.
    -   Copy and paste the list of ingredients from a food package into the text area.
    -   Select the commodity type (e.g., Snacks, Beverage) and click "Analyze Ingredients" for a safety and health report.
5.  **Shopping Cart**:
    -   When viewing a recipe, you can automatically add its ingredients to your cart.
    -   On the "Shopping Cart" page, you can manually add items, edit quantities, and remove items.
    -   Use the "Copy Items" button to copy your shopping list for easy use.

---

## Project Analysis

### Pros

-   **Innovative AI Integration**: Effectively combines computer vision (YOLOv8) with a powerful LLM (Gemini) to provide a unique, multi-faceted user experience.
-   **Custom-Trained Model**: The use of a custom-trained YOLOv8 model for food recognition, especially with a focus on Indian cuisine, makes the application highly relevant and accurate for its target audience.
-   **Comprehensive Feature Set**: The application goes beyond simple recognition, offering a suite of genuinely useful tools like diet planning, recipe generation, and ingredient analysis.
-   **End-to-End Functionality**: This is a complete, deployable web application with a functional frontend, a robust backend, and seamless API integration, demonstrating a full-stack development skillset.

### Cons & Future Improvements

-   **No User Accounts**: The shopping cart currently relies on Flask's session management. Implementing user authentication and a database (like PostgreSQL or MongoDB) would allow users to save their diet plans, favorite recipes, and track their nutritional history over time.
-   **API Dependency & Cost**: The application's core functionality relies heavily on the Google Gemini API, which may incur costs at scale and is dependent on external service availability.
-   **Static Frontend**: The frontend is built with vanilla JavaScript. Migrating to a modern framework like React or Vue.js could create a more dynamic, responsive, and stateful user interface.
-   **Batch Image Processing**: Currently, images are processed one by one. Future versions could allow users to upload multiple images at once for batch analysis.

---

## Contributors

This project was developed as part of our College academic curriculum by a dedicated team of students. It represents our collective effort to apply our software development and machine learning skills to build a real-world, impactful application.

---

## License

This project is licensed under the MIT License.
