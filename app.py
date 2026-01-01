import streamlit as st
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import pandas as pd

# --- PAGE CONFIG ---
# Sets the browser tab title and expands the layout to use the whole screen
st.set_page_config(page_title="AI News Sentiment Tracker", layout="wide")

# --- API INITIALIZATION ---
# Using the key you provided directly
API_KEY = '79da3bff5f4543ed978677d09538e660'
newsapi = NewsApiClient(api_key=API_KEY)
analyzer = SentimentIntensityAnalyzer()

# --- UI HEADER ---
st.title("📊 Industry News Sentiment Dashboard")
st.markdown("""
    This industry-level tool fetches real-time global news and performs **Sentiment Analysis** using the VADER (Valence Aware Dictionary and sEntiment Reasoner) NLP model.
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Configuration")
query = st.sidebar.text_input("Search Keyword", "Global Economy")
num_articles = st.sidebar.slider("Articles to Analyze", 5, 50, 15)
sort_order = st.sidebar.selectbox("Sort by", ["relevancy", "publishedAt", "popularity"])

if st.sidebar.button("Run Analysis"):
    with st.spinner('Accessing NewsAPI and running NLP models...'):
        try:
            # Fetch News Data
            data = newsapi.get_everything(q=query, 
                                          language='en', 
                                          sort_by=sort_order, 
                                          page_size=num_articles)
            articles = data['articles']

            if not articles:
                st.warning("No news found for that topic. Try a broader keyword.")
            else:
                # Data Processing logic
                results = []
                sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}

                for art in articles:
                    title = art['title']
                    # Perform Sentiment Analysis
                    score = analyzer.polarity_scores(title)['compound']
                    
                    if score >= 0.05:
                        label = "Positive"
                    elif score <= -0.05:
                        label = "Negative"
                    else:
                        label = "Neutral"
                    
                    sentiment_counts[label] += 1
                    results.append({
                        "Source": art['source']['name'],
                        "Title": title,
                        "Sentiment": label,
                        "Score": score,
                        "URL": art['url']
                    })

                # Convert to DataFrame for easier handling
                df = pd.DataFrame(results)

                # --- TOP ROW: VISUALS ---
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader(f"Sentiment Analysis for '{query}'")
                    fig = px.pie(
                        names=list(sentiment_counts.keys()), 
                        values=list(sentiment_counts.values()),
                        color=list(sentiment_counts.keys()),
                        color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#f1c40f', 'Negative':'#e74c3c'},
                        hole=0.5
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Data Summary")
                    st.metric("Total Articles", len(df))
                    st.metric("Dominant Sentiment", max(sentiment_counts, key=sentiment_counts.get))
                    
                    # CSV Export for Industry Standard reporting
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Data as CSV",
                        data=csv,
                        file_name=f"{query}_sentiment_report.csv",
                        mime="text/csv",
                    )

                # --- BOTTOM ROW: NEWS FEED ---
                st.divider()
                st.subheader("Detailed Sentiment Breakdown")
                
                # Using a table for a clean industry look
                st.dataframe(df[['Source', 'Sentiment', 'Score', 'Title']], use_container_width=True)

                st.markdown("### Article Links")
                for index, row in df.iterrows():
                    st.markdown(f"- **[{row['Title']}]({row['URL']})** ({row['Source']})")

        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.info("Note: Free API keys sometimes have limits. Check your NewsAPI.org dashboard if this persists.")

else:
    st.info("Enter a keyword in the sidebar and click 'Run Analysis' to see industry insights.")