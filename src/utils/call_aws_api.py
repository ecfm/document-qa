import requests
import time
import streamlit as st

# Remove hardcoded values and use st.secrets
url = st.secrets["AWS_API_URL"]
headers = {'x-api-key': st.secrets["AWS_API_KEY"]}

MAX_RETRY = 10

def call_api(category, review, retry=MAX_RETRY, status_container=None): 
    request_body = {
        "category": category,
        "review": review
    }   
    response = requests.post(url, json=request_body, headers=headers)
    if 'generation' in response.json():
        return response.json()['generation']
    elif 'timed out' in response.json()['errorMessage'] and retry > 0:
        retry_status = status_container.empty() if status_container else None
        for i in range(30, 0, -1):
            if retry_status:
                retry_status.text(f"Initial run could trigger a timeout error. Retrying in {i} seconds. Retries left: {retry}")
            time.sleep(1)
        if retry_status:
            retry_status.empty()
        return call_api(category, review, retry-1, status_container)  
    else:
        raise Exception(response.json())

if __name__ == "__main__":
    call_api("sleep aids", "I utilized breathing techniques with it like inhaling, hold, exhaling, hold and this spray paired with that is great! This has a frosty cool lavender scent not to strong very light but you can catch the smell when its on something or blankets.")
