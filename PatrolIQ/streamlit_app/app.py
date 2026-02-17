
# streamlit_app/app.py

import streamlit as st

st.set_page_config(
    page_title="PatrolIQ - Smart Safety Analytics",
    layout="wide"
)

st.title("🚓 PatrolIQ - Smart Safety Analytics Platform")
st.markdown("""
### Urban Crime Intelligence System

This platform analyzes 500,000 crime records from Chicago to:

- Identify geographic crime hotspots
- Analyze temporal crime patterns
- Perform clustering and dimensionality reduction
- Support data-driven policing strategies
""")

st.info("Use the sidebar to navigate between analysis modules.")
