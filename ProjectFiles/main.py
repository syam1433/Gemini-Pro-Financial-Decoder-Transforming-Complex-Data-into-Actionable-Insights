import streamlit as st
from streamlit_lottie import st_lottie
import requests
import google.generativeai as genai
import json
import os
import pandas as pd
import plotly.express as px
import camelot
import pdfplumber
from PyPDF2 import PdfReader
from tabula import read_pdf
import base64

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# def set_background(image_path):
#     """Sets a background image for the Streamlit app."""
#     with open(image_path, "rb") as f:
#         encoded_string = base64.b64encode(f.read()).decode()

#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background: url("data:image/png;base64,{encoded_string}") no-repeat center center fixed;
#             background-size: cover;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
def set_background_video(video_path):
    """Sets a background video for the Streamlit app excluding the sidebar."""
    with open(video_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    
    video_tag = f"""
    <style>
    body {{
        margin: 0;
        overflow: hidden;
    }}
    .stApp {{
        position: relative;
        z-index: 1;
    }}
    .main {{
        position: relative;
        z-index: 1;
    }}
    .video-bg {{
        position: fixed;
        top: 0;
        left: 0;
        width: calc(100% - 300px); /* Adjust for sidebar width */
        height: 100%;
        object-fit: cover;
        z-index: -1;
    }}
    </style>
    <video class="video-bg" autoplay loop muted playsinline>
        <source src="data:video/mp4;base64,{encoded_string}" type="video/mp4">
    </video>
    """
    
    st.markdown(video_tag, unsafe_allow_html=True)


# # Set the background image (Replace 'background.png' with the actual image file path)
# set_background("display.jpg")

def go_to_questions():
    st.session_state["current_page"] = "📊 Questions"
    st.experimental_rerun()

def extract_tables_from_pdf(pdf_path):
    """Extract tables from a PDF with minimal data loss."""
    extracted_tables = []

    # 1️⃣ Try extracting tables using Camelot (works best for structured PDFs with borders)
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")  # lattice mode for bordered tables
        if tables.n > 0:
            extracted_tables = [table.df.to_dict(orient="records") for table in tables]
    except Exception as e:
        st.warning(f"Camelot extraction failed: {e}")

    # 2️⃣ If Camelot fails, try pdfplumber (works for text-based tables)
    if not extracted_tables:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])  # First row as column names
                        extracted_tables.append(df.to_dict(orient="records"))
        except Exception as e:
            st.warning(f"pdfplumber extraction failed: {e}")

    # 3️⃣ If pdfplumber fails, use Tabula (last fallback)
    if not extracted_tables:
        try:
            df_list = read_pdf(pdf_path, pages="all", multiple_tables=True)
            if df_list:
                extracted_tables = [df.dropna(how="all").to_dict(orient="records") for df in df_list]
        except Exception as e:
            st.error(f"Tabula extraction failed: {e}")

    return extracted_tables if extracted_tables else None   

def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()


def extract_text_and_tables_from_pdf(pdf_path):
    """Extracts text and tables from a PDF while minimizing data loss."""
    extracted_data = {"text": "", "tables": []}

    # Extract text using pdfplumber (more reliable than PyPDF2)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            extracted_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        extracted_data["text"] = extracted_text.strip()
    except Exception as e:
        st.error(f"Error extracting text: {e}")

    # Try extracting tables using Camelot (works for PDFs with structured lines)
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")  # or "lattice" for bordered tables
        if tables.n > 0:
            extracted_data["tables"] = [table.df.to_dict(orient="records") for table in tables]
    except Exception as e:
        st.warning(f"Camelot extraction failed: {e}")

    # If Camelot fails, use Tabula
    if not extracted_data["tables"]:
        try:
            df_list = read_pdf(pdf_path, pages="all", multiple_tables=True)
            if df_list:
                extracted_data["tables"] = [df.dropna(how="all").to_dict(orient="records") for df in df_list]
        except Exception as e:
            st.error(f"Tabula extraction failed: {e}")

    return extracted_data


# Initialize session state
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "🏠 Home"

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📂 Analysis", "📊 Questions","🔍📊 Compare","📈 Dynamic Dashboard"])

# Update session state based on selection
st.session_state["current_page"] = page

# Page Routing
if st.session_state["current_page"] == "🏠 Home":
    # set_background_video("video.mp4")
    st.title("Welcome to the Financial Decoder App!")
    st.write("Your end-to-end solution for analyzing financial data and interacting with a financial chatbot.")
    st.write("Select a page from the sidebar to get started.")


elif st.session_state["current_page"] == "📂 Analysis":
# Load Lottie animation
    loading_animation = load_lottieurl("https://lottie.host/28c73c9d-92b0-4979-ac1e-8ec806c80096/TrSxsWSMLI.json")

    def analyze_financial_data(json_data):
        """Summarize financial performance using Gemini AI."""
        model = genai.GenerativeModel("gemini-pro")
        
        prompt = f"""
        Analyze the following financial data and provide a structured summary for the Indian market. 

        Key aspects to cover:
        - Categorized review with pros and cons (5 points each) in a technical manner
        - Drawbacks to be addressed precisely in bullet points
        - Areas of improvement based on the financial data
        - Overall short summarization and conclusion
        - All the above data should be in the Indian context
        -If any of the data misses try to infer from the relevant financial principles
        
        Provide the response in strict JSON format with the following structure:
        {{
            "Pros": ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"],
            "Cons": ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"],
            "Drawbacks": ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"],
            "Areas of Improvement": ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"],
            "Summary": "Short summarization and conclusion."
        }}
        
        Financial Data:
        {json.dumps(json_data, indent=2)}
        """
        
        placeholder = st.empty()
        with placeholder.container():
            st_lottie(loading_animation, speed=1, width=200, height=200, key="loading1")
            st.write("Analyzing financial data... Please wait.")
        
        response = model.generate_content(prompt)
        placeholder.empty()
        
        if not response or not response.text:
            st.error("No response received from AI.")
            return None
        
        try:
            response_text = response.text.strip()
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            response_json = response_text[json_start:json_end]
            analysis_json = json.loads(response_json)
            return analysis_json
        except json.JSONDecodeError:
            st.error("Failed to parse AI response. Showing raw output:")
            st.text(response.text)
            return None

    # Streamlit UI
    st.title("Financial Data Analysis using Gemini AI")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        temp_pdf_path = "temp_uploaded.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write("Extracting tables from PDF...")
        json_data = extract_tables_from_pdf(temp_pdf_path)
        
        if json_data:
            st.success("Tables extracted successfully!")
            
            st.subheader("AI Analysis")
            analysis = analyze_financial_data(json_data)
            
            if analysis:
                # Display pros and cons in columns without markdown borders
                st.subheader("Pros and Cons")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("### Pros")
                    for point in analysis.get("Pros", []):
                        st.write(f"- {point}")
                    
                with col2:
                    st.write("### Cons")
                    for point in analysis.get("Cons", []):
                        st.write(f"- {point}")

                # Add space between Pros/Cons and Drawbacks/Areas of Improvement
                st.write("")  # This adds some space (a blank line)

                st.write("")

        # Center the button using columns
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("Ask Questions"):
                        go_to_questions()

                # Display drawbacks and areas of improvement in columns
                st.subheader("Drawbacks & Areas of Improvement")
                col3, col4 = st.columns(2)
                with col3:
                    st.write("### Drawbacks")
                    for point in analysis.get("Drawbacks", []):
                        st.write(f"- {point}")

                with col4:
                    st.write("### Areas of Improvement")
                    for point in analysis.get("Areas of Improvement", []):
                        st.write(f"- {point}")

                # Add space between Drawbacks/Areas of Improvement and the summary
                st.write("")  # Adds another blank line for spacing
                
                # Display summarization
                st.subheader("Summary and Conclusion")
                st.write(analysis.get("Summary", "No summary available."))


            else:
                st.error("Error processing AI analysis.")
        else:
            st.error("No tables found in the PDF.")
        
        os.remove(temp_pdf_path)

elif st.session_state["current_page"] == "📊 Questions":
    st.title("Interactive Financial Chatbot")
    st.sidebar.subheader("Upload Financial Document")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"], key="question_file")
    
    if uploaded_file:
        temp_pdf_path = "temp_question.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        extracted_data = extract_text_and_tables_from_pdf(temp_pdf_path)
        os.remove(temp_pdf_path)

        if extracted_data:
            st.session_state["financial_data"] = extracted_data
            st.sidebar.success("File uploaded successfully! You can now chat.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("Ask about the uploaded data:")
    loading_animation = load_lottieurl("https://lottie.host/49b9c55b-71e1-4ba2-907f-83a13c78988f/kg2zTwg79c.json")

    if user_input:
        model = genai.GenerativeModel("gemini-pro")
        chat_context = "\n".join([f"User: {msg['user']}\nAI: {msg['ai']}" for msg in st.session_state["chat_history"]])
        
        prompt = f"""
                    You are a highly skilled financial analyst specializing in the **Indian market**, including trends in banking, stock markets, taxation, corporate finance, and economic policies. 
                    Your role is to provide **precise, data-backed, and insightful** responses to user queries.

                   ### **Instructions for AI:**
                    - Act as a **Personal Financial Assistant**, strictly analyzing the provided data.
                    - Use **only extracted financial tables and text**—no assumptions, only factual insights.
                    - If data is **incomplete**, infer only from relevant financial principles.
                    - Ensure responses are **structured, concise, and relevant** to:
                    - **Indian stock market (NSE, BSE)**
                    - **RBI policies, taxation, GST**
                    - **Economic indicators (GDP, inflation, interest rates)**
                    - **Corporate finance under Indian regulations**
                    - **No extra commentary or assumptions**—stick to **data-backed analysis** only.
                    -**If any of the data misses try to infer from the relevant financial principles

                    ### **Extracted Financial Text:**
                    {extracted_data.get("text", "No extracted text available.")}

                    ### **Extracted Financial Tables:**
                    {json.dumps(extracted_data.get("tables", {}), indent=2)}

                    ### **Previous Conversation Context:**
                    {chat_context}

                    ### **User Query:**
                    {user_input}

                    ### **AI Response:**
                    """
        placeholder = st.empty()
        with placeholder.container():
            st_lottie(loading_animation, speed=1, width=200, height=200, key="loading1")
            st.write("Analyzing financial data... Please wait.")
        response = model.generate_content(prompt)
        placeholder.empty()
        ai_response = response.text if response else "Error generating response."

        st.session_state["chat_history"].append({"user": user_input, "ai": ai_response})
        
        st.write("### AI Response:")
        st.write(ai_response)

    if st.session_state["chat_history"]:
        st.subheader("Chat History")
        for msg in st.session_state["chat_history"][-5:]:  # Show last 5 interactions
            st.write(f"**User:** {msg['user']}")
            st.write(f"**AI:** {msg['ai']}")

elif st.session_state["current_page"] == "🔍📊 Compare":
    st.title("Compare Financial Documents")
    st.write("Upload two financial documents to compare their key insights.")
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file1 = st.file_uploader("Upload First PDF", type=["pdf"], key="file1")
    with col2:
        uploaded_file2 = st.file_uploader("Upload Second PDF", type=["pdf"], key="file2")
    
    if uploaded_file1 and uploaded_file2:
        temp_pdf_path1 = "temp_file1.pdf"
        temp_pdf_path2 = "temp_file2.pdf"
        
        with open(temp_pdf_path1, "wb") as f1, open(temp_pdf_path2, "wb") as f2:
            f1.write(uploaded_file1.getbuffer())
            f2.write(uploaded_file2.getbuffer())
        
        st.write("Extracting data from both PDFs...")
        data1 = extract_text_and_tables_from_pdf(temp_pdf_path1)
        data2 = extract_text_and_tables_from_pdf(temp_pdf_path2)
        
        os.remove(temp_pdf_path1)
        os.remove(temp_pdf_path2)
        
        if data1 and data2:
            st.success("Data extracted successfully!")
            
            col3, col4 = st.columns(2)
            # with col3:
            #     st.subheader("Document 1 Summary")
            #     st.write(data1.get("text", "No text available"))
            # with col4:
            #     st.subheader("Document 2 Summary")
            #     st.write(data2.get("text", "No text available"))
                
            st.subheader("Comparison Summary")
            model = genai.GenerativeModel("gemini-pro")
            prompt = f"""
            Compare the following two financial reports and provide insights:
            
            **Document 1:**
            {data1.get("text", "No text available")}
            
            **Document 2:**
            {data2.get("text", "No text available")}
            
            **Key Aspects to Compare:**
            - Present key differences and similarities in a structured table format with four columns as follows Aspects|Document 1 (Infosys)|Document 2 (Reliance Industries) .
            - Clearly highlight areas where one document outperforms the other.
            - Provide an overall insight and conclusion below the table.
            -Provide th suggestions for investment based on the comparison.
            Ensure the response maintains a professional, structured, and financial analysis-oriented approach.
            """

            response = model.generate_content(prompt)
            if response and response.text:
                st.write(response.text)
            else:
                st.error("Error generating comparison insights.")

elif st.session_state["current_page"] == "📈 Dynamic Dashboard":
    st.title("📈 Interactive Financial Dashboard")
    st.sidebar.subheader("Upload Financial Document")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"], key="dashboard_file")
    
    if uploaded_file:
        temp_pdf_path = "temp_dashboard.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write("Extracting financial data...")
        extracted_data = extract_tables_from_pdf(temp_pdf_path)
        os.remove(temp_pdf_path)
        
        if extracted_data:
            st.success("Financial data extracted successfully!")
            df_list = []
            
            for table in extracted_data:
                df = pd.DataFrame(table).dropna(how='all').drop_duplicates()
                
                # Assign the first row as column headers if missing
                if df.iloc[0].notna().all():
                    df.columns = df.iloc[0]
                    df = df[1:].reset_index(drop=True)
                
                # Clean numeric columns
                df = df.applymap(lambda x: str(x).replace(',', '').replace('(', '-').replace(')', '') if isinstance(x, str) else x)
                df = df.apply(pd.to_numeric, errors='coerce')
                
                # Drop columns that are still non-numeric
                df = df.dropna(axis=1, how='all')
                df_list.append(df)
                
            for idx, df in enumerate(df_list):
                st.subheader(f"Table {idx+1}")
                st.write(df)
                
                # Filtering Options
                with st.expander("🔍 Filter Data"):
                    for col in df.columns:
                        unique_values = df[col].dropna().unique()
                        if len(unique_values) < 20:
                            selected_values = st.multiselect(f"Filter {col}", unique_values, default=unique_values, key=f"filter_{idx}_{col}")
                            df = df[df[col].isin(selected_values)]
                
                # Visualization Options
                st.subheader("📊 Interactive Charts")
                numeric_columns = df.select_dtypes(include=['number']).columns
                if len(numeric_columns) >= 2:
                    x_axis = st.selectbox("Select X-axis", numeric_columns, key=f"x_{idx}")
                    y_axis = st.selectbox("Select Y-axis", numeric_columns, key=f"y_{idx}")
                    
                    if x_axis and y_axis:
                        fig1 = px.line(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
                        st.plotly_chart(fig1)
                        
                        fig2 = px.bar(df, x=x_axis, y=y_axis, title=f"Bar Chart: {x_axis} vs {y_axis}")
                        st.plotly_chart(fig2)
                        
                        fig3 = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot: {x_axis} vs {y_axis}")
                        st.plotly_chart(fig3)
                        
                        fig4 = px.pie(df, names=x_axis, values=y_axis, title=f"Pie Chart: {x_axis} Distribution")
                        st.plotly_chart(fig4)
                        
                        fig5 = px.histogram(df, x=x_axis, title=f"Histogram of {x_axis}")
                        st.plotly_chart(fig5)
                        
                        fig6 = px.imshow(df.corr(), text_auto=True, title="Correlation Matrix")
                        st.plotly_chart(fig6)
                else:
                    st.warning("Not enough numeric data for visualization.")
        else:
            st.error("No financial data found in the PDF.")


# elif st.session_state["current_page"] == "📈 Dynamic Dashboard":
#     st.title("📊 Interactive Financial Dashboard")
# def extract_tables_from_pdf(pdf_path):
#     """Extracts tables from a PDF with minimal data loss."""
#     extracted_tables = []

#     # Try extracting tables using Camelot (structured PDFs with borders)
#     try:
#         tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
#         if tables.n > 0:
#             extracted_tables = [table.df for table in tables]
#     except Exception:
#         pass

#     # If Camelot fails, use pdfplumber (text-based tables)
#     if not extracted_tables:
#         try:
#             with pdfplumber.open(pdf_path) as pdf:
#                 for page in pdf.pages:
#                     table = page.extract_table()
#                     if table:
#                         df = pd.DataFrame(table[1:], columns=table[0])
#                         extracted_tables.append(df)
#         except Exception:
#             pass

#     # If both fail, use Tabula (last fallback)
#     if not extracted_tables:
#         try:
#             df_list = read_pdf(pdf_path, pages="all", multiple_tables=True)
#             if df_list:
#                 extracted_tables = [df.dropna(how="all") for df in df_list]
#         except Exception:
#             pass

#     return extracted_tables if extracted_tables else None

# def clean_dataframe(df):
#     """Cleans extracted dataframe by handling duplicate columns, empty headers, and formatting issues."""
#     # Rename empty columns
#     df.columns = [f"Column_{i}" if not col.strip() else col for i, col in enumerate(df.columns)]
    
#     # Remove duplicate column names
#     df = df.loc[:, ~df.columns.duplicated()]
    
#     # Drop fully empty rows and columns
#     df = df.dropna(how="all").dropna(axis=1, how="all")
    
#     # Convert numeric columns
#     for col in df.columns:
#         df[col] = df[col].astype(str).str.replace(',', '').str.strip()
#         df[col] = pd.to_numeric(df[col], errors='ignore')
    
#     return df

# st.title("Interactive Financial Dashboard")

# uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# if uploaded_file:
#     temp_pdf_path = "temp_uploaded.pdf"
#     with open(temp_pdf_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     st.write("Extracting tables from PDF...")
#     tables = extract_tables_from_pdf(temp_pdf_path)

#     if tables:
#         st.success("Tables extracted successfully!")
        
#         # Display all extracted tables
#         for i, table in enumerate(tables):
#             st.write(f"### Raw Extracted Table {i+1}")
#             st.write(table)
        
#         df = clean_dataframe(tables[0])  # Use the first table by default
#         st.write("### Cleaned Data")
#         st.write(df)
        
#         # Interactive Chart
#         numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
#         if numeric_columns:
#             selected_column = st.selectbox("Select a column to visualize", numeric_columns)
#             fig = px.line(df, y=selected_column, title=f"Trend of {selected_column}")
#             st.plotly_chart(fig)
#         else:
#             st.warning("No numeric data available for visualization.")
#     else:
#         st.error("No tables found in the PDF.")
