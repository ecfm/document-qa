from datasets import load_dataset
import re
import pandas as pd

def load_data(category, size, progress_text=None, progress_bar=None, data_container=None):
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{category}", trust_remote_code=True, streaming=True)
    reviews = []
    for i, review in enumerate(dataset['full'], 1):
        # check if review['title'] ends with punctuations
        if re.search(r'[^\w\s]', review['title']):
            title = review['title'] + " ."
        else:
            title = review['title']
        reviews.append(title + " " + review['text'])
        
        if progress_text:
            progress_text.text(f"Loading review {i}/{size}")
        if progress_bar:
            progress_bar.progress(i/size)
        if data_container:  # Update display every 5 reviews
            out_df = pd.DataFrame(reviews, columns=['review'])
            data_container.dataframe(out_df)
            
        if len(reviews) == size:
            break
            
    out_df = pd.DataFrame(reviews, columns=['review'])
    if data_container:
        data_container.dataframe(out_df)
    return out_df

if __name__ == "__main__":
    out_df = load_data("Digital_Music", 10)
    print(out_df)