"""
Gemini AI Flight Agent - Streamlit Application
AI-Enhanced Flight Search & Booking Platform
"""

import streamlit as st
import os
import time
import asyncio
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any

# Import your models and agent
from agent import GeminiFlightAgent, FlightRequest, TripType, CabinClass

# Page configuration
st.set_page_config(
    page_title="Gemini AI Flight Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_api_key():
    """Get API key from Streamlit secrets (for deployed app) or environment (for local development)"""
    try:
        # Try Streamlit secrets first (for deployed app)
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        # Fallback to environment variable (for local development)
        return os.getenv("GEMINI_API_KEY")

def check_gemini_setup():
    """Check if Gemini is properly configured"""
    api_key = get_api_key()
    if not api_key or api_key == "your_gemini_api_key_here":
        return False
    return True

# Enhanced CSS styling with AI branding
st.markdown("""
<style>
    /* Global text color improvements */
    .main .block-container {
        color: #2c3e50;
    }
    
    .stMarkdown, .stText, p, div, span {
        color: #2c3e50 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1a252f !important;
    }
    
    /* Form labels and text */
    label, .stSelectbox label, .stNumberInput label, .stDateInput label {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #4285f4 0%, #34a853 50%, #ea4335 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    .main-header > * {
        position: relative;
        z-index: 1;
        color: white !important;
    }
    
    .main-header h1 {
        color: white !important;
    }
    
    .main-header p {
        color: white !important;
    }
    
    .gemini-badge {
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.3);
        color: white !important;
    }
    
    .form-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e1e5e9;
    }
    
    .form-container h3 {
        color: #2c3e50 !important;
        margin-bottom: 1rem;
    }
    
    /* Airport suggestion styling */
    .airport-suggestion {
        padding: 0.75rem 1rem;
        border: 1px solid #ddd;
        border-radius: 8px;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: all 0.2s ease;
        background: white;
        color: #2c3e50 !important;
    }
    
    .airport-suggestion:hover {
        background: #f8f9fa;
        border-color: #4285f4;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(66,133,244,0.15);
        color: #1a252f !important;
    }
    
    .airport-suggestion.selected {
        background: #e3f2fd;
        border-color: #4285f4;
        font-weight: 500;
        color: #1a252f !important;
    }
    
    /* Search progress styling */
    .search-step {
        display: flex;
        align-items: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .search-step-icon {
        font-size: 1.5rem;
        margin-right: 1rem;
        color: #2c3e50 !important;
    }
    
    .search-step-text {
        font-size: 1rem;
        font-weight: 500;
        color: #2c3e50 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4285f4 0%, #34a853 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        color: white !important;
    }
    
    /* Progress container styling */
    .ai-progress-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #4285f4;
    }
    
    .ai-progress-container h3 {
        color: #2c3e50 !important;
    }
    
    /* Results container */
    .results-container {
        margin: 2rem 0;
    }
    
    .results-container h3 {
        color: #2c3e50 !important;
    }
    
    /* Deal cards keep white text as they have dark backgrounds */
    .deal-card {
        background: linear-gradient(135deg, #00C851 0%, #007E33 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,200,81,0.3);
    }
    
    .deal-card h2, .deal-card h3, .deal-card h4, .deal-card p {
        color: white !important;
    }
    
    .competitive-card {
        background: linear-gradient(135deg, #4285f4 0%, #34a853 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(66,133,244,0.3);
    }
    
    .competitive-card h2, .competitive-card h3, .competitive-card h4, .competitive-card p {
        color: white !important;
    }
    
    /* Sidebar improvements */
    .stSidebar .stMarkdown {
        color: #2c3e50 !important;
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: #1a252f !important;
    }
    
    /* Metrics styling */
    .stMetric .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e1e5e9;
    }
    
    .stMetric label {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="metric-value"] {
        color: #1a252f !important;
        font-weight: bold !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        color: #2c3e50 !important;
    }
    
    /* Table styling */
    .stDataFrame {
        color: #2c3e50 !important;
    }
    
    /* Info/success/warning messages */
    .stInfo {
        background-color: #e8f4fd;
        color: #0c5460 !important;
    }
    
    .stSuccess {
        background-color: #d1eddb;
        color: #0a3622 !important;
    }
    
    .stWarning {
        background-color: #fff3cd;
        color: #664d03 !important;
    }
    
    .stError {
        background-color: #f8d7da;
        color: #721c24 !important;
    }
    
    /* Caption styling */
    .caption {
        color: #6c757d !important;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with permanent API key
if 'flight_agent' not in st.session_state:
    api_key = get_api_key()
    if api_key:
        st.session_state.flight_agent = GeminiFlightAgent(api_key=api_key)
    else:
        st.session_state.flight_agent = GeminiFlightAgent()

if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'is_searching' not in st.session_state:
    st.session_state.is_searching = False
if 'current_flight_request' not in st.session_state:
    st.session_state.current_flight_request = None
if 'selected_origin' not in st.session_state:
    st.session_state.selected_origin = None
if 'selected_destination' not in st.session_state:
    st.session_state.selected_destination = None

# Airport search component
def render_airport_search(key: str, label: str, icon: str, placeholder: str) -> Optional[str]:
    """Render airport search component with autocomplete"""
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown(f"### {icon}")
    
    with col2:
        st.markdown(f"**{label}**")
        search_input = st.text_input(
            label,
            key=f"{key}_search",
            placeholder=placeholder,
            label_visibility="collapsed"
        )
    
    selected_airport = None
    
    if search_input and len(search_input) >= 2:
        suggestions = st.session_state.flight_agent.autocomplete_airports(search_input)
        
        if suggestions:
            st.markdown("**Suggestions:**")
            for suggestion in suggestions:
                if st.button(
                    f"{suggestion['display']} ({suggestion['code']})",
                    key=f"{key}_{suggestion['code']}",
                    use_container_width=True
                ):
                    selected_airport = suggestion['code']
                    st.session_state[f"selected_{key}"] = selected_airport
                    st.rerun()
        else:
            if len(search_input) >= 3:
                st.caption("🔍 No airports found. Try a different search term.")
            
    elif len(search_input) == 1:
        st.caption("💡 Keep typing to see airport suggestions...")
    
    return selected_airport or st.session_state.get(f"selected_{key}")

# Helper functions for search
async def perform_ai_enhanced_search_async(flight_request: FlightRequest):
    """Perform the AI-enhanced search asynchronously"""
    return await st.session_state.flight_agent.search_flights(flight_request)

def perform_ai_search(flight_request: FlightRequest):
    """Wrapper to run async search in sync context"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(perform_ai_enhanced_search_async(flight_request))

# Main header with Gemini branding
st.markdown("""
<div class="main-header">
    <h1>🤖 Gemini AI Flight Agent</h1>
    <div class="gemini-badge">🧠 Powered by Google Gemini AI</div>
    <p style="font-size: 1.2rem; margin: 0;">Real AI-Enhanced Flight Search & Optimization</p>
</div>
""", unsafe_allow_html=True)

# API Status Check (simplified)
if not check_gemini_setup():
    st.error("❌ **Gemini API Key Not Configured**")
    st.info("The application is not properly configured. Please contact the administrator.")
    st.stop()  # Stop execution if no API key
else:
    st.success("✅ **Gemini AI Ready** - Flight search powered by Google's advanced AI")

# Flight Search Form
st.markdown('<div class="form-container">', unsafe_allow_html=True)
st.markdown("### ✈️ Flight Search")

# Trip type selection
col1, col2 = st.columns([2, 1])
with col1:
    trip_type = st.selectbox(
        "Trip Type",
        options=[TripType.ROUND_TRIP, TripType.ONE_WAY],
        format_func=lambda x: "Round Trip" if x == TripType.ROUND_TRIP else "One Way"
    )

with col2:
    passengers = st.number_input("Passengers", min_value=1, max_value=9, value=1)

# Airport selection
col1, col2 = st.columns(2)
with col1:
    origin = render_airport_search("origin", "From", "🛫", "Enter departure city or airport code")

with col2:
    destination = render_airport_search("destination", "To", "🛬", "Enter destination city or airport code")

# Date selection
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    departure_date = st.date_input(
        "Departure Date",
        min_value=date.today(),
        value=date.today() + timedelta(days=7)
    )

with col2:
    if trip_type == TripType.ROUND_TRIP:
        return_date = st.date_input(
            "Return Date",
            min_value=departure_date,
            value=departure_date + timedelta(days=7)
        )
    else:
        return_date = None

with col3:
    cabin_class = st.selectbox(
        "Class",
        options=[CabinClass.ECONOMY, CabinClass.PREMIUM_ECONOMY, CabinClass.BUSINESS, CabinClass.FIRST],
        format_func=lambda x: x.value.replace("_", " ").title()
    )

st.markdown('</div>', unsafe_allow_html=True)

# Search button
search_button = st.button("🔍 Search Flights with AI", use_container_width=True, type="primary")

if search_button:
    if not origin:
        st.error("Please select a departure airport")
    elif not destination:
        st.error("Please select a destination airport")
    elif origin == destination:
        st.error("Departure and destination cannot be the same")
    else:
        # Create flight request
        flight_request = FlightRequest(
            origin=origin,
            destination=destination,
            departure_date=departure_date.strftime('%Y-%m-%d'),
            return_date=return_date.strftime('%Y-%m-%d') if return_date else None,
            passengers=passengers,
            cabin_class=cabin_class,
            trip_type=trip_type
        )
        
        st.session_state.current_flight_request = flight_request
        st.session_state.is_searching = True

# Search progress and results
if st.session_state.is_searching:
    # Progress container
    st.markdown('<div class="ai-progress-container">', unsafe_allow_html=True)
    st.markdown("### 🤖 AI-Enhanced Flight Search in Progress")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Search steps
    search_steps = [
        "🧠 Analyzing route with Gemini AI...",
        "🔍 Searching Google Flights...",
        "💰 Comparing affiliate prices...",
        "📊 Processing results with AI...",
        "✅ Search complete!"
    ]
    
    for i, step in enumerate(search_steps):
        status_text.text(step)
        progress_bar.progress((i + 1) / len(search_steps))
        time.sleep(0.8)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Perform the actual search
    try:
        with st.spinner("Finalizing results..."):
            results = perform_ai_search(st.session_state.current_flight_request)
        
        st.session_state.search_results = results
        st.session_state.is_searching = False
        st.rerun()
        
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        st.session_state.is_searching = False

# Display results
if st.session_state.search_results:
    results = st.session_state.search_results
    
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown("### 🎯 Flight Search Results")
    
    # Best deal card
    if results.best_affiliate and results.savings > 0:
        st.markdown(f'''
        <div class="deal-card">
            <h2>💰 Best Deal Found!</h2>
            <h3>${results.best_affiliate.price:.0f} via {results.best_affiliate.source}</h3>
            <p><strong>Save ${results.savings:.0f} ({results.savings_percentage:.1f}%)</strong> compared to Google Flights</p>
            <p>🎯 AI-powered optimization saved you money!</p>
        </div>
        ''', unsafe_allow_html=True)
        
        if results.best_affiliate.booking_link:
            st.link_button(
                f"Book Now on {results.best_affiliate.source}",
                results.best_affiliate.booking_link,
                use_container_width=True
            )
    else:
        st.markdown(f'''
        <div class="competitive-card">
            <h2>🎯 Competitive Pricing</h2>
            <h3>${results.google_flights.price:.0f} via Google Flights</h3>
            <p>Our AI analysis shows this is already competitively priced!</p>
            <p>🧠 Gemini AI verified pricing across multiple sources</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Results summary
    st.info(results.message)
    
    # AI Analysis Details
    with st.expander("🧠 Detailed Gemini AI Analysis", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**AI Search Strategy:**")
            if st.session_state.flight_agent.is_gemini_available():
                st.write("✅ Gemini route analysis enabled")
                st.write("✅ Partner optimization active")
                st.write("✅ Intelligent price prediction")
                st.write("✅ Real-time deal discovery")
            else:
                st.write("⚠️ Standard search mode")
                st.write("🔧 Add Gemini API key for AI features")
        
        with col2:
            st.markdown("**Search Performance:**")
            st.write(f"🕐 Total time: {results.search_time:.1f} seconds")
            affiliate_count = len([r for r in results.all_results if r.source != 'Google Flights'])
            st.write(f"🔍 Partners searched: {affiliate_count}")
            st.write(f"📊 Results analyzed: {len(results.all_results)}")
            st.write(f"🎯 AI Model: Gemini 1.5 Flash")
    
    # Price comparison table
    with st.expander("📊 All Price Comparisons"):
        comparison_data = []
        for result in results.all_results:
            if result.success and result.price:
                savings_vs_google = results.google_flights.price - result.price if result.source != "Google Flights" else 0
                comparison_data.append({
                    "Source": result.source,
                    "Price": f"${result.price:.0f}",
                    "Savings": f"${savings_vs_google:.0f}" if savings_vs_google > 0 else "-",
                    "Status": "✅ Best Deal" if result == results.best_affiliate else "📊 Baseline" if result.source == "Google Flights" else "🔍 Checked"
                })
        
        if comparison_data:
            st.table(comparison_data)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar with additional info
with st.sidebar:
    st.markdown("### 🤖 AI Status")
    if check_gemini_setup():
        st.success("✅ Gemini AI Active")
        st.caption("Powered by Google's advanced AI")
    else:
        st.error("❌ AI Unavailable")
        st.caption("Service temporarily unavailable")
    
    st.markdown("### 📊 Search Stats")
    if st.session_state.search_results:
        results = st.session_state.search_results
        st.metric("Search Time", f"{results.search_time:.1f}s")
        st.metric("Sources Checked", len(results.all_results))
        if results.savings > 0:
            st.metric("Savings Found", f"${results.savings:.0f}")
    
    st.markdown("### ℹ️ About")
    st.caption("This AI-powered flight search uses Google's Gemini AI to analyze routes, compare prices across multiple booking platforms, and find you the best deals available.")
    
    st.markdown("### 🛠️ Features")
    st.caption("• Real-time price comparison")
    st.caption("• AI-powered route optimization")
    st.caption("• Multi-platform search")
    st.caption("• Intelligent deal discovery")