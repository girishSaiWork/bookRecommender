"""
Semantic Book Recommender System
This module implements a book recommendation system using semantic search and emotional analysis.
"""

# Standard library imports
import os
from typing import List, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import PGVector

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Loading environment variables...")

# Database configuration
CONNECTION_STRING = os.getenv('DATABASE_URL')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

# File paths
data_dir = os.path.join(os.getcwd(), "data")
filepath = os.path.join(data_dir, "books_with_emotions.csv")
cover_image_path = os.path.join(data_dir, "cover-not-found.jpg")

def load_book_data() -> pd.DataFrame:
    """
    Load and preprocess the books dataset.
    
    Returns:
        pd.DataFrame: Processed books dataframe
    """
    logger.info(f"Reading data from {filepath}")
    try:
        books = pd.read_csv(filepath)
        books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
        books["large_thumbnail"] = np.where(
            books["large_thumbnail"].isna(),
            cover_image_path,
            books["large_thumbnail"],
        )
        logger.info(f"Successfully loaded {len(books)} books")
        return books
    except FileNotFoundError:
        logger.error(f"Error: File not found at {filepath}")
        raise

def initialize_vector_store() -> PGVector:
    """
    Initialize the PostgreSQL vector store with embeddings.
    
    Returns:
        PGVector: Initialized vector store
    """
    logger.info("Initializing vector store...")
    return PGVector(
        connection_string=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(model="mxbai-embed-large"),
        use_jsonb=True,
    )

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    """
    Retrieve book recommendations based on semantic search and filters.
    
    Args:
        query (str): User's search query
        category (str, optional): Book category filter
        tone (str, optional): Emotional tone filter
        initial_top_k (int): Initial number of recommendations
        final_top_k (int): Final number of recommendations
    
    Returns:
        pd.DataFrame: Filtered and sorted book recommendations
    """
    logger.info(f"Searching for: '{query}' with category={category}, tone={tone}")
    
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [rec.page_content.strip('"').split()[0].replace('"', '').replace("'", '') for rec in recs]
    book_recs = books[books["isbn10"].isin(books_list)].head(initial_top_k)

    # Apply category filter
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # Apply emotional tone sorting
    tone_mapping = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness"
    }
    
    if tone in tone_mapping:
        book_recs.sort_values(by=tone_mapping[tone], ascending=False, inplace=True)

    logger.info(f"Found {len(book_recs)} recommendations")
    return book_recs

def format_authors(authors: str) -> str:
    """
    Format the authors string for display.
    
    Args:
        authors (str): Semicolon-separated author names
    
    Returns:
        str: Formatted author string
    """
    authors_split = authors.split(";")
    if len(authors_split) == 2:
        return f"{authors_split[0]} and {authors_split[1]}"
    elif len(authors_split) > 2:
        return f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
    return authors

def recommend_books(
        query: str,
        category: str,
        tone: str
) -> List[Tuple[str, str]]:
    """
    Main recommendation function for the Gradio interface.
    
    Args:
        query (str): User's search query
        category (str): Book category
        tone (str): Emotional tone
    
    Returns:
        List[Tuple[str, str]]: List of (image_url, caption) pairs
    """
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = " ".join(row["description"].split()[:30]) + "..."
        authors_str = format_authors(row["authors"])
        caption = f"{row['title']} by {authors_str}: {description}"
        results.append((row["large_thumbnail"], caption))
    
    return results

# Load initial data
logger.info("Initializing application...")
books = load_book_data()
db_books = initialize_vector_store()

# Prepare UI options
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Define Gradio interface
with gr.Blocks(theme=gr.themes.Ocean()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness"
        )
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select a category:",
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select an emotional tone:",
            value="All"
        )
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(
        label="Recommended books",
        columns=8,
        rows=2
    )

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    dashboard.launch()