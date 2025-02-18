# Semantic Book Recommender System

A sophisticated book recommendation system that uses semantic search, emotion analysis, and natural language processing to provide personalized book recommendations.

## 🎯 Features

- **Semantic Search**: Find books using natural language queries
- **Category Filtering**: Filter books by fiction/non-fiction categories
- **Emotional Tone Analysis**: Sort books by emotional tones (Happy, Surprising, Angry, Suspenseful, Sad)
- **Interactive Web Interface**: User-friendly interface built with Gradio
- **Vector Database Integration**: Efficient similarity search using PostgreSQL vector store

## 🛠️ Technologies Used

### Core Technologies
- Python 3.11+
- PostgreSQL (for vector storage)
- LangChain
- Ollama (for local embeddings)

### Key Libraries
- **Data Processing**
  - Pandas
  - NumPy
  - KaggleHub (for dataset access)

- **Machine Learning & NLP**
  - LangChain Community
  - LangChain Ollama
  - Transformers

- **Web Interface**
  - Gradio
  - Gradio Themes

- **Development Tools**
  - Python-dotenv
  - Jupyter Notebook
  - IPython Widgets

## 📁 Project Structure

```
src/
├── jobs/
│ ├── main.py # Main application code
│ └── research/ # Research notebooks
│ ├── data_cleaning_and_exploration.ipynb
│ ├── vector-search.ipynb
│ ├── text-classification.ipynb
│ └── sentiment-analysis.ipynb
```

## 🚀 Getting Started

### Prerequisites
1. Python 3.11 or higher
2. PostgreSQL database
3. Ollama installed locally

### Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd semantic-book-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with:
```
DATABASE_URL=your_postgresql_connection_string
COLLECTION_NAME=your_collection_name
```

4. Download the dataset from Kaggle:
- [7K Books Dataset by Dylan Castillo](https://kaggle.com/datasets/dylanjcas...)

### Running the Application

1. Start the web interface:
```bash
python src/jobs/main.py
```

2. Open your browser and navigate to the local URL provided by Gradio

## 📚 Components

1. **Data Cleaning & Exploration**
   - Initial data processing
   - Dataset analysis and cleaning
   - Feature engineering

2. **Vector Search Implementation**
   - Text embedding generation
   - Vector database setup
   - Similarity search implementation

3. **Text Classification**
   - Fiction/Non-fiction classification
   - Zero-shot classification using LLMs
   - Category system implementation

4. **Sentiment Analysis**
   - Emotion extraction from text
   - Tone analysis implementation
   - Emotional sorting system

5. **Web Application**
   - Interactive user interface
   - Real-time recommendations
   - Filter and sort capabilities

## 🔗 Useful Resources

- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Gradio Documentation](https://gradio.app/docs/)
- [Vector Search Concepts](https://weaviate.io/developers/weaviate)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For any questions or feedback, please reach out to [Your Contact Information]

---

**Note**: This project was developed as part of a tutorial series on building semantic search applications with modern NLP techniques by FreecodeCamp.
