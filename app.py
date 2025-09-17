import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import pdfplumber
import openai
from io import BytesIO

# ➤ Use your OpenAI API key directly here
openai.api_key = "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"

# Helper function to read PDF pages as text
def read_pdf(file, pages=None):
    text = ""
    with pdfplumber.open(file) as pdf:
        selected_pages = pages if pages else range(len(pdf.pages))
        for i in selected_pages:
            page = pdf.pages[i]
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Function to send text to GPT model and extract fields
def extract_fields(text):
    prompt = f"""
You are an expert document parser. Extract all possible field names and their corresponding values from the following document text.
Provide the result as a JSON object where keys are the field names and values are their respective values.

Document Text:
{text}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1500
        )
        result = response['choices'][0]['message']['content']
        return result
    except Exception as e:
        st.error(f"Error extracting fields: {e}")
        return "{}"

# Convert JSON string to DataFrame
def json_to_dataframe(json_str):
    try:
        df = pd.read_json(json_str)
        return df
    except Exception as e:
        st.error(f"Error converting JSON: {e}")
        return pd.DataFrame()

# Convert DataFrame to downloadable file
def convert_df(df, file_type="csv"):
    if file_type == "csv":
        return df.to_csv(index=False).encode('utf-8')
    else:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        return output.getvalue()

# Streamlit UI
st.title("📄 GPT-Powered Data Extraction from PDF")

# Step 1: Upload files
uploaded_files = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=['pdf'])

if uploaded_files:
    cumulative_data = []
    
    for uploaded_file in uploaded_files:
        st.subheader(f"File: {uploaded_file.name}")

        # Select file type
        file_type = st.radio("Select file type", ["PDF", "OCR"], key=uploaded_file.name)
        
        # Preview PDF pages
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            num_pages = doc.page_count
            st.write(f"Total pages: {num_pages}")
            preview_page = st.number_input(f"Select page to preview (1-{num_pages})", min_value=1, max_value=num_pages, value=1, key=uploaded_file.name + "_preview")
            page = doc.load_page(preview_page - 1)
            text_preview = page.get_text()
            st.text_area("Page Content Preview", text_preview, height=300)
        
        # Select pages to extract
        page_list_input = st.text_input("Enter pages to extract (comma-separated, leave blank for all)", key=uploaded_file.name + "_pages")
        if page_list_input.strip():
            pages = [int(p.strip()) - 1 for p in page_list_input.split(",") if p.strip().isdigit()]
        else:
            pages = None
        
        # Extract fields
        text = read_pdf(uploaded_file, pages)
        fields_json = extract_fields(text)
        df = json_to_dataframe(fields_json)

        if not df.empty:
            st.write("Extracted Fields:")
            selected_columns = st.multiselect("Select fields to keep", df.columns.tolist(), default=df.columns.tolist(), key=uploaded_file.name + "_selectfields")
            df_selected = df[selected_columns]
            
            # Preview extracted data
            st.dataframe(df_selected)
            
            # Download option
            download_format = st.radio("Choose download format", ["csv", "excel"], key=uploaded_file.name + "_format")
            file_ext = "csv" if download_format == "csv" else "xlsx"
            data_file = convert_df(df_selected, download_format)
            
            st.download_button(
                label="Download File",
                data=data_file,
                file_name=f"{uploaded_file.name.split('.')[0]}_extracted.{file_ext}",
                mime="text/csv" if file_ext=="csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            cumulative_data.append(df_selected)

    # Combined download if multiple files
    if len(cumulative_data) > 1:
        final_df = pd.concat(cumulative_data, ignore_index=True)
        st.subheader("Combined Data from All Files")
        st.dataframe(final_df)
        combined_data = convert_df(final_df, "csv")
        st.download_button(
            label="Download Combined CSV",
            data=combined_data,
            file_name="combined_extracted_data.csv",
            mime="text/csv"
        )
