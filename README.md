## Email Spam Detection

A simple API to classify emails as spam or ham using Logistic Regression model

---

## Tech Stack
- FastAPI
- Pydantic
- Scikit-Learn (Logistic Regression, TF-IDF Vectorizer, Train-Test Split)
- Joblib
## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

This project uses **[uv](https://github.com/astral-sh/uv)** as the primary package and project manager. Ensure you have Python installed.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abdelouahab-aourar/EmailSpamDetection
   cd EmailSpamDetection
2. **Install uv run:**
   ```bash
   pip install uv
3. **Install all dependencies defined in the pyproject.toml file:**
   ```bash
   uv sync
### Running the API
To start the development server:
1. **Navigate to the API directory**
   ```bash
   cd api
2. **Run the Uvicorn server Replace PORT_NUM with your desired port (e.g., 8000):**
   ```bash
   uv run uvicorn app:app --port PORT_NUM --reload
   ```
   ---
### API Documentation
Once the server is running, you can access the interactive built-in API documentation at:

- Swagger UI: http://127.0.0.1:PORT_NUM/docs
- ReDoc: http://127.0.0.1:PORT_NUM/redoc