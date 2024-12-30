import streamlit as st
import pandas as pd
from utils.call_aws_api import call_api
from utils.load_amazon_reviews import load_data
from docx import Document
from nltk.tokenize import sent_tokenize, word_tokenize 
import nltk

# Download required NLTK data
nltk.download('punkt_tab')

def split_into_paragraphs(text, sentence_min_len=8, sentence_max_len=200):
    # Simple sentence splitting on common punctuation
    sentences = sent_tokenize(text)
    
    filtered_sentences = []
    skipped_short_sentences = []
    skipped_long_sentences = []
    
    for sentence in sentences:
        wc = len(word_tokenize(sentence))
        if sentence_min_len <= wc <= sentence_max_len:
            filtered_sentences.append(sentence)
        else:
            if wc < sentence_min_len:
                skipped_short_sentences.append(sentence)
            else:
                skipped_long_sentences.append(sentence)
    
    return filtered_sentences
# Constants
MAX_REVIEWS = 1000
MAX_CHARS_PER_REVIEW = 2000

def validate_review_length(review: str) -> str:
    """Validate and truncate review if it exceeds maximum length."""
    if len(review) > MAX_CHARS_PER_REVIEW:
        return review[:MAX_CHARS_PER_REVIEW]
    return review

def validate_reviews_count(reviews: list) -> list:
    """Validate and limit the number of reviews."""
    if len(reviews) > MAX_REVIEWS:
        return reviews[:MAX_REVIEWS]
    return reviews

def process_file_upload(upload_file) -> tuple[pd.DataFrame, str]:
    """Process uploaded file and return dataframe and review column name."""
    review_column = 'Review'
    file_type = upload_file.name.lower().split('.')[-1]
    if file_type == "csv" or file_type == "xlsx":
        if file_type == "csv":
            df = pd.read_csv(upload_file)
        else:
            df = pd.read_excel(upload_file)
        if len(df.columns) > 1:
            review_column = st.selectbox("Select Review Column:", df.columns.tolist())
        else:
            review_column = df.columns[0]
        return df, review_column
    elif file_type == "txt":
        content = upload_file.read().decode()
        reviews = split_into_paragraphs(content)
        return pd.DataFrame({review_column: reviews}), review_column
    elif file_type == "docx":
        doc = Document(upload_file)
        content = "\n".join([para.text for para in doc.paragraphs])
        reviews = split_into_paragraphs(content)
        return pd.DataFrame({review_column: reviews}), review_column
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def process_text_input(review_input: str) -> tuple[pd.DataFrame, str]:
    """Process text input and return dataframe and review column name."""
    review_column = 'Review'
    reviews = [r.strip() for r in review_input.split('\n') if r.strip()]
    return pd.DataFrame({review_column: reviews}), review_column

def process_reviews(input_df: pd.DataFrame, review_column: str, category_name: str, display_container: st.container) -> pd.DataFrame:
    """Process reviews through the LLM and return results."""
    reviews = input_df[review_column].tolist()
    reviews = validate_reviews_count(reviews)
    
    # Validate review lengths
    valid_reviews = [validate_review_length(review) for review in reviews]
    input_df = input_df.copy()  # Create a copy to avoid modifying the original
    input_df[review_column] = valid_reviews
    
    # Initialize output DataFrame with all columns
    output_df = input_df.copy()
    output_df['LLM Response'] = ''  # Add the new column with empty strings
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    status_container = st.empty()
    
    total = len(valid_reviews)
    for i, row in input_df.iterrows():
        progress_text.text(f"Processing review {i+1}/{total} (This may take a few minutes)")
        progress_bar.progress((i+1)/total)
        review = row[review_column]
        response = call_api(category_name, review, status_container=status_container)
        output_df.at[i, 'LLM Response'] = response
        with display_container:
            st.dataframe(output_df, use_container_width=True)
    
    progress_text.empty()
    status_container.empty()
    progress_bar.empty()
    return output_df

def handle_download_reviews():
    """Handle the download reviews tab functionality."""
    st.header("Download Sample Amazon Reviews from Hugging Face")
    AMAZON_CATEGORIES = [
        "All_Beauty", "Toys_and_Games", "Cell_Phones_and_Accessories", 
        "Industrial_and_Scientific", "Gift_Cards", "Musical_Instruments",
        "Electronics", "Handmade_Products", "Arts_Crafts_and_Sewing",
        "Baby_Products", "Health_and_Household", "Office_Products",
        "Digital_Music", "Grocery_and_Gourmet_Food", "Sports_and_Outdoors", 
        "Home_and_Kitchen", "Subscription_Boxes", "Tools_and_Home_Improvement",
        "Pet_Supplies", "Video_Games", "Kindle_Store",
        "Clothing_Shoes_and_Jewelry", "Patio_Lawn_and_Garden", "Unknown",
        "Books", "Automotive", "CDs_and_Vinyl", "Beauty_and_Personal_Care",
        "Amazon_Fashion", "Magazine_Subscriptions", "Software",
        "Health_and_Personal_Care", "Appliances", "Movies_and_TV"
    ]

    category = st.selectbox(
        "Amazon Review Category *",
        options=AMAZON_CATEGORIES,
        index=AMAZON_CATEGORIES.index("Digital_Music"),
        help="Required. Select the category from the Hugging Face dataset: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023#grouped-by-category"
    )
    num_examples = st.number_input(
        "Number of Examples *", 
        min_value=1, 
        max_value=MAX_REVIEWS, 
        value=10,
        help=f"Required. Choose between 1-{MAX_REVIEWS} reviews to download."
    )
    
    if st.button("Load Data"):
        progress_text = st.empty()
        progress_bar = st.progress(0)
        data_container = st.empty()
        
        data = load_data(category, num_examples, progress_text, progress_bar, data_container)
        
        progress_text.empty()
        progress_bar.empty()
        
        if data is not None and not data.empty:
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{category}-first_{num_examples}_reviews.csv",
                mime="text/csv"
            )

def handle_llm_submission():
    """Handle the LLM submission tab functionality."""
    st.header("Submit Reviews to LLM")
    
    category_name = st.text_input(
        "Review Category *",
        help="Required. Enter the category that best describes these reviews (e.g. Electronics, Books, etc.)"
    )
    
    upload_file = st.file_uploader(
        "Upload File (Optional)", 
        type=["csv", "txt", "docx", "xlsx"],
        help="Optional. Upload a .csv file or an Excel file with headers, a .txt file, or a .docx file."
    )
    
    if upload_file is not None:
        input_df, review_column = process_file_upload(upload_file)
        display_container = st.container()
        with display_container:
            st.subheader("Data & Results")
            st.dataframe(input_df, use_container_width=True)
    else:
        review_input = st.text_area(
            "Enter Reviews; separate multiple reviews by new lines (Optional, not needed if file uploaded)",
            help="Enter one or more reviews, each on a new line. Not needed if you uploaded a file above."
        )
        input_df, review_column = process_text_input(review_input)
        display_container = st.container()
        with display_container:
            st.subheader("Data & Results") 
            st.dataframe(input_df, use_container_width=True)
    
    if st.button("Submit Reviews to LLM", disabled=not category_name):
        output_df = process_reviews(input_df, review_column, category_name, display_container)
        csv_response = output_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Response CSV", csv_response, f"{category_name}-response.csv", "text/csv")

def main():
    """Main function to run the Streamlit app."""
    st.title("Voice of Customer Analysis with Supervised-Finetuned LLM")
    
    tab1, tab2 = st.tabs(["Download Reviews (Optional)", "Submit Reviews to LLM"])
    
    with tab1:
        handle_download_reviews()
    
    with tab2:
        handle_llm_submission()

if __name__ == "__main__":
    main()