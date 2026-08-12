import os
import streamlit as st
from AutomatedLeads import ColdEmailEngine

st.set_page_config(page_title="Cold Email Generator", layout="centered")
st.title("Cold Email Generator")

# Initialize persistent session state
if "engine" not in st.session_state:
    st.session_state.engine = ColdEmailEngine()

if "generated_email" not in st.session_state:
    st.session_state.generated_email = ""

# File upload section for VectorDB context
uploaded_file = st.file_uploader("Upload Product / Portfolio Context (PDF, CSV, Excel)", type=["pdf", "csv", "xlsx", "xls"])
if uploaded_file:
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Processing file into VectorDB..."):
        try:
            st.session_state.engine.create_vector_store(temp_path)
            st.success("Context file loaded into VectorDB!")
        except Exception as e:
            st.error(f"Error loading file: {e}")

st.divider()

# Input Mode: Radio toggle simplifies interface by removing redundant text boxes
input_mode = st.radio("Select Input Source:", ["Web Link", "Direct Text"], horizontal=True)
lead_data = ""

if input_mode == "Web Link":
    job_url = st.text_input("Enter Job URL:")
    if job_url:
        with st.spinner("Scraping webpage..."):
            try:
                lead_data = st.session_state.engine.scrape_url(job_url)
                st.success("Webpage content scraped!")
            except Exception as e:
                st.error(f"Failed to scrape URL: {e}")
else:
    lead_data = st.text_area("Paste Job Description:", height=150)

# Single Action button to trigger workflow
if st.button("Generate Email", type="primary"):
    if not lead_data:
        st.warning("Please provide a URL or paste text first.")
    else:
        with st.spinner("Generating personalized email..."):
            try:
                st.session_state.generated_email = st.session_state.engine.generate_email(lead_data)
            except Exception as e:
                st.error(f"Generation error: {e}")

# Email Display and Dispatch
if st.session_state.generated_email:
    st.divider()
    edited_email = st.text_area("Generated Cold Email (Editable):", value=st.session_state.generated_email, height=220)
    
    recipient = st.text_input("Recipient Email:")
    if st.button("Send Email"):
        if not recipient:
            st.warning("Please enter a recipient email.")
        else:
            with st.spinner("Sending email..."):
                try:
                    st.session_state.engine.send_email(
                        recipient_email=recipient,
                        subject="Quick Question",
                        body=edited_email
                    )
                    st.success("Email sent successfully!")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")