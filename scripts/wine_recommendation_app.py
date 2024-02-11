import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os

# Function to load data from GitHub URL
def load_data():
    # GitHub URL of the dataset
    github_csv_url = "https://raw.githubusercontent.com/shr53/WineRecommendation/main/data/preprocessed_wine_dataset.csv"
    return pd.read_csv(github_csv_url)

# Load preprocessed data
wine_recommend_df = load_data()

# Function to get image URL based on environment
def get_image_url(image_filename):
    # Define relative path to the images directory from the script's location
    IMAGES_DIR = '../images/'
    if 'streamlit' in os.getcwd():  # Running on Streamlit Cloud
        # GitHub URL of the images
        return f'https://raw.githubusercontent.com/shr53/WineRecommendation/main/images/{image_filename}'
    else:  # Running locally
        # Local path to images directory
        script_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        images_dir = os.path.join(project_root, 'images')
        return os.path.join(images_dir, image_filename)

# Update wine images dictionary with image URLs
wine_images = {
    'Red wine': get_image_url('7850104_1.png'),
    'White wine': get_image_url('7850101_1.png'),
    'Other': get_image_url('other.png'),
    'Sparkling wine/Champagne': get_image_url('Sparkling.png'),
    # Add more mappings as needed
}

# Update sentiment images dictionary with image URLs
sentiment_images = {
    'Positive': get_image_url('positive_sentiment.png'),
    'Negative': get_image_url('negative_sentiment.png'),
}

# Default image if the wine type is not in the dictionary
default_image = 'default_wine.png'

def recommend_wines(brand, wine_type, min_rating):
    # Filter wines based on user preferences
    filtered_wines = wine_recommend_df[
        (wine_recommend_df['brand'] == brand) &
        (wine_recommend_df['wine_type'] == wine_type) &
        (wine_recommend_df['average_rating'] >= min_rating)
    ]

    # Extract only the name part without the quantity information
    filtered_wines['name'] = filtered_wines['name'].str.split('-').str[0].str.strip()
    
    # Calculate the total number of reviews for the filtered wines
    review_counts = filtered_wines.groupby(['brand', 'name'])['reviews.rating'].count().reset_index()
    review_counts.rename(columns={'reviews.rating': 'total_reviews'}, inplace=True)

    # Merge the review counts back into the filtered wines DataFrame
    recommended_wines = pd.merge(filtered_wines, review_counts, on=['brand', 'name'], how='left')

    # Fill NaN values in 'total_reviews' column with 0
    recommended_wines['total_reviews'].fillna(0, inplace=True)

    # Drop duplicates to ensure unique wines
    recommended_wines = recommended_wines.drop_duplicates(subset=['brand', 'name'])

    # Sort wines by average rating and total reviews as a measure of popularity
    recommended_wines['popularity_score'] = recommended_wines['average_rating'] * recommended_wines['total_reviews']
    recommended_wines = recommended_wines.sort_values(by='popularity_score', ascending=False)

    # Select the top 5 unique recommendations
    top_recommendations = recommended_wines.head(5)
    
    return top_recommendations

# Streamlit app main section
def main():
    # Set page title and icon
    st.set_page_config(page_title="PourPerfection", page_icon="🍷", layout="wide")

    # Add background gradient color
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(120deg, #8B0000, #800020);
            background-size: cover;
            color: #ffffff; /* Text color: white */
        }
        .card {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        .star-rating {
            color: #FFD700;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🍷 PourPerfection: Expert Wine Recommendations")
    st.write("Welcome to PourPerfection, your destination for discovering the most popular wines based on brand, user ratings, and type. Select your preferred brand, wine type, and minimum rating to receive expert recommendations tailored to your taste.")

    # Define sidebar options
    st.sidebar.title("🍾 User Wine Preferences")
    # Get unique brands
    unique_brands = ['Select Brand'] + list(wine_recommend_df['brand'].unique())
    # Dropdown with default placeholder
    brand = st.sidebar.selectbox('🏷️ Select Brand', unique_brands)

    # Ensure that the default placeholder isn't selectable
    if brand == 'Select Brand':
        brand = None
    
    # Get unique wine types
    unique_wine_types = ['Select Wine Type'] + list(wine_recommend_df['wine_type'].unique())
    # Dropdown with default placeholder
    wine_type = st.sidebar.selectbox('🍷 Select Wine Type', unique_wine_types)

    # Ensure that the default placeholder isn't selectable
    if wine_type == 'Select Wine Type':
        wine_type = None
    
    min_rating = st.sidebar.slider('⭐ Minimum Rating', min_value=1, max_value=5, value=1)
    
    if brand is None or wine_type is None or min_rating == 1:
        st.markdown(
            "<h3 style='text-align: center;'>Please select your preferences to start your search in the right panel.</h3>",
            unsafe_allow_html=True
        )
    else:
        recommendations = recommend_wines(brand, wine_type, min_rating)
        display_recommendations(recommendations)

# Function to display a bar chart of top recommended wines
def display_recommendations(recommendations):
    if not recommendations.empty:
        st.markdown("<h2 style='text-align: center;'>🌟 Top Recommendations</h2>", unsafe_allow_html=True)
        cols = st.columns(len(recommendations))

        for col, (_, row) in zip(cols, recommendations.iterrows()):
            with col:
                # Display the wine image using Streamlit's image function, centered
                image_path = wine_images.get(row['wine_type'], default_image)
                col.markdown(
                    f"<div style='text-align: center; max-width: 200px; margin-left: auto; margin-right: auto;'>"
                    f"<img src='data:image/png;base64,{get_image_as_base64(image_path)}' style='max-width: 100%; height: auto;'>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # Use markdown to center the text below the image
                col.markdown(f"<div style='text-align: center;'>"
                             f"<h3>{row['name']}</h3>"
                             f"Brand: {row['brand']}<br>"
                             f"Wine Type: {row['wine_type']}<br>"
                             f"{row['average_rating']} ★ ({row['total_reviews']} reviews)</div>",
                             unsafe_allow_html=True)

                # Determine the sentiment and fetch the correct image path
                sentiment = "Positive" if row['sentiment_score'] > 0 else "Negative"
                sentiment_image = sentiment_images[sentiment]

                # Display the sentiment image centered using HTML
                col.markdown(
                    f"<div style='text-align: center;'>"
                    f"<img src='data:image/png;base64,{get_image_as_base64(sentiment_image)}' style='width: 30px; margin: 0 auto;'>"
                    f"<p>{sentiment}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )     
    else:
        # Get the directory path of the current script
        script_dir = os.path.dirname(__file__)

        # Navigate up directories until we reach the project root
        project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

        # Define the relative path to the images directory
        images_dir = os.path.join(project_root, 'images')

        # Define the relative path to the no-result image
        no_result_image_path = os.path.join(images_dir, 'no-result.png')

        # Read the image file and encode it as base64
        with open(no_result_image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')

        # Display the base64 encoded image
        st.markdown(
            f"<div style='text-align: center;'>"
            f"<img src='data:image/png;base64,{image_base64}' width='200' alt='No recommendations found'>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Hide the "Top Recommendations" title
        st.markdown("<p style='text-align: center;'>We couldn't find any recommendations based on your preferences.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Try adjusting your filters or check back later!</p>", unsafe_allow_html=True)

# Function to create visualizations based on review ratings and brand
def visualize_ratings_by_brand(data):
    # Bar plot showing average review ratings for each brand
    st.subheader("Average Review Ratings by Brand")
    avg_ratings = data.groupby('brand')['reviews.rating'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_ratings.values, y=avg_ratings.index, palette="viridis")
    plt.xlabel('Average Rating')
    plt.ylabel('Brand')
    plt.title('Average Review Ratings by Brand')
    st.pyplot()

    # Box plot showing distribution of review ratings for each brand
    st.subheader("Distribution of Review Ratings by Brand")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='reviews.rating', y='brand', data=data, palette="muted")
    plt.xlabel('Review Rating')
    plt.ylabel('Brand')
    plt.title('Distribution of Review Ratings by Brand')
    st.pyplot()

# Function to convert image to base64 for HTML rendering
def get_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
if __name__ == '__main__':
    main()
