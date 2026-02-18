🚀 Running the Streamlit Financial Analysis App Locally
Prerequisites
Ensure you have the following installed on your system:

Python 3.8+
pip (Python package manager)
Virtual environment (optional but recommended)
1️⃣ Clone the Repository

git clone https://github.com/your-username/your-repo.git  
cd your-repo  

2️⃣ Create a Virtual Environment (Optional but Recommended)

python -m venv venv  
source venv/bin/activate  # On macOS/Linux  
venv\Scripts\activate  # On Windows  

3️⃣ Install Dependencies

pip install -r requirements.txt  

4️⃣ Set Up Environment Variables
Create a .env file in the root directory and add your API keys (Google Gemini API, etc.):

GEMINI_API_KEY=your_api_key  
Or set them manually:

export GEMINI_API_KEY=your_api_key  # macOS/Linux  
set GEMINI_API_KEY=your_api_key  # Windows  

5️⃣ Run the Streamlit App

streamlit run app.py  

This will open the app in your default browser.

