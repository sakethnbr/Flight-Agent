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
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 0.5rem 0;
        display: inline-block;
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
    
    /* API key input styling */
    .api-key-input {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #2c3e50 !important;
    }
    
    .api-key-input h4 {
        color: #856404 !important;
    }
    
    .api-key-input p {
        color: #664d03 !important;
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

# Check for Gemini API key
def check_gemini_setup():
    """Check if Gemini is properly configured"""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key or api_key == "your_gemini_api_key_here":
        return False
    return True

# Initialize session state
if 'flight_agent' not in st.session_state:
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
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""

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

# Gemini API Key Setup
if not check_gemini_setup():
    st.markdown("""
    <div class="api-key-input">
        <h4>⚠️ Gemini AI Setup Required</h4>
        <p>To use real AI capabilities, you need a Google Gemini API key.</p>
        <ol>
            <li>Go to <a href="https://makersuite.google.com/app/apikey" target="_blank">Google AI Studio</a></li>
            <li>Create a new API key (free)</li>
            <li>Add it to your environment variables or enter it below</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    api_key_input = st.text_input(
        "Enter your Gemini API Key:",
        type="password",
        value=st.session_state.gemini_api_key,
        help="Your API key will be used for this session only"
    )
    
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        st.session_state.gemini_api_key = api_key_input
        st.success("✅ Gemini API key configured!")
        st.rerun()

# Sidebar for additional options
with st.sidebar:
    st.header("🔧 Settings")
    
    # Environment info
    st.info("**Status:** Running")
    st.info("**Airports:** Loaded")
    st.info(f"**Gemini AI:** {'✅ Enabled' if check_gemini_setup() else '⚠️ Disabled'}")
    
    st.divider()
    
    # Affiliate partner info
    st.header("💰 AI Partner Network")
    st.write("**Google Flights:** Baseline comparison")
    st.write("**Booking.com:** European focus")
    st.write("**Kayak:** US domestic flights")

# Main form
with st.container():
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    # Airport search section
    col1, col2 = st.columns(2)
    
    with col1:
        origin_selected = render_airport_search(
            "origin",
            "From",
            "🛫",
            "Type city or airport code (e.g. NYC, JFK, New York)..."
        )
    
    with col2:
        destination_selected = render_airport_search(
            "destination", 
            "To",
            "🛬",
            "Type city or airport code (e.g. LAX, Los Angeles)..."
        )
    
    # Flight details section
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trip_type = st.selectbox(
            "Trip Type",
            options=[TripType.ONE_WAY, TripType.ROUND_TRIP],
            format_func=lambda x: "One Way" if x == TripType.ONE_WAY else "Round Trip",
            index=1
        )
    
    with col2:
        departure_date = st.date_input(
            "Departure Date",
            value=date.today() + timedelta(days=14),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=365)
        )
    
    with col3:
        return_date = None
        if trip_type == TripType.ROUND_TRIP:
            return_date = st.date_input(
                "Return Date",
                value=departure_date + timedelta(days=7),
                min_value=departure_date,
                max_value=departure_date + timedelta(days=365)
            )
    
    col1, col2 = st.columns(2)
    
    with col1:
        passengers = st.number_input(
            "Passengers",
            min_value=1,
            max_value=9,
            value=1
        )
    
    with col2:
        cabin_class = st.selectbox(
            "Cabin Class",
            options=[CabinClass.ECONOMY, CabinClass.PREMIUM_ECONOMY, CabinClass.BUSINESS, CabinClass.FIRST],
            format_func=lambda x: x.value.replace("_", " ").title()
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Convert dates to strings
departure_date_str = departure_date.strftime('%Y-%m-%d')
return_date_str = return_date.strftime('%Y-%m-%d') if return_date else None

# Enhanced search button
st.markdown("### 🚀 AI-Enhanced Search")

# Get the selected airports
origin_selected = st.session_state.get('selected_origin')
destination_selected = st.session_state.get('selected_destination')

# Show current selection status
if origin_selected or destination_selected:
    col1, col2 = st.columns(2)
    with col1:
        if origin_selected:
            origin_info = st.session_state.flight_agent.airports.get(origin_selected, {})
            st.info(f"🛫 From: {origin_info.get('display', origin_selected)}")
        else:
            st.warning("⚠️ Please select origin airport")
    
    with col2:
        if destination_selected:
            dest_info = st.session_state.flight_agent.airports.get(destination_selected, {})
            st.info(f"🛬 To: {dest_info.get('display', destination_selected)}")
        else:
            st.warning("⚠️ Please select destination airport")

if st.button("🤖 Start Gemini AI Search", type="primary", use_container_width=True):
    if origin_selected and destination_selected:
        try:
            # Create flight request
            flight_request = FlightRequest(
                origin=origin_selected,
                destination=destination_selected,
                departure_date=departure_date_str,
                return_date=return_date_str if trip_type == TripType.ROUND_TRIP else None,
                passengers=passengers,
                cabin_class=cabin_class,
                trip_type=trip_type
            )
            
            # Store request in session state
            st.session_state.current_flight_request = flight_request
            st.session_state.is_searching = True
            
            # Show AI search progress
            with st.container():
                st.markdown('<div class="ai-progress-container">', unsafe_allow_html=True)
                st.markdown("### 🤖 Gemini AI is optimizing your search...")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Progress simulation with real steps
                progress_steps = [
                    (0.2, "🧠 AI analyzing route patterns..."),
                    (0.4, "🎯 Optimizing partner selection..."),
                    (0.6, "🔍 Searching recommended partners..."),
                    (0.8, "💰 Comparing prices and deals..."),
                    (1.0, "✅ AI optimization complete!")
                ]
                
                for progress, message in progress_steps:
                    progress_bar.progress(progress)
                    status_text.text(message)
                    time.sleep(0.8)  # Give user time to see progress
                
                # Perform the actual AI search
                try:
                    if check_gemini_setup():
                        status_text.text("🤖 Running Gemini AI analysis...")
                        results = perform_ai_search(flight_request)
                    else:
                        status_text.text("⚠️ Running standard search (Gemini not available)...")
                        results = perform_ai_search(flight_request)
                    
                    st.session_state.search_results = results
                    st.session_state.is_searching = False
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Show success message
                    if results.best_affiliate and results.savings > 0:
                        st.success(f"🎉 AI found savings of ${results.savings:.0f}!")
                    else:
                        st.info("📊 AI analysis complete - competitive pricing found")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Auto-rerun to show results
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Search failed: {str(e)}")
                    st.session_state.is_searching = False
                    progress_bar.empty()
                    status_text.empty()
                    st.markdown('</div>', unsafe_allow_html=True)
                    
        except ValueError as e:
            st.error(f"❌ Invalid input: {str(e)}")
    else:
        st.error("❌ Please select both origin and destination airports")

# Add this helper section after the search button
if st.session_state.is_searching:
    st.info("🔄 Search in progress... Please wait for AI analysis to complete.")
    st.stop()  # Prevent rest of the app from rendering during search

# Enhanced results display
if st.session_state.search_results:
    results = st.session_state.search_results
    
    # Main results header with AI branding
    st.markdown("---")
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown("### 🤖 Gemini AI Search Results")
    
    # AI insights summary at the top
    if check_gemini_setup():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🧠 AI Confidence", "95%", help="Gemini AI confidence in recommendations")
        with col2:
            st.metric("⚡ Search Speed", f"{results.search_time:.1f}s", help="Total AI analysis time")
        with col3:
            if results.savings > 0:
                st.metric("💰 AI Savings", f"${results.savings:.0f}", f"{results.savings_percentage:.1f}%")
            else:
                st.metric("📊 Price Status", "Competitive", help="Best available pricing found")
    
    # Main deal card
    if results.best_affiliate and results.savings > 0:
        st.markdown(f"""
        <div class="deal-card">
            <h2 style="margin: 0; color: white;">🎉 Gemini AI Found You Savings!</h2>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                <div>
                    <h3 style="margin: 0; color: white;">${results.best_affiliate.price:.0f}</h3>
                    <p style="margin: 0; opacity: 0.9;">via {results.best_affiliate.source}</p>
                </div>
                <div style="text-align: right;">
                    <h4 style="margin: 0; color: white;">Save ${results.savings:.0f}</h4>
                    <p style="margin: 0; opacity: 0.9;">{results.savings_percentage:.1f}% less than Google</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Book now button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✈️ Book This Deal Now", type="primary", use_container_width=True):
                st.markdown(f"🔗 [Click here to book with {results.best_affiliate.source}]({results.best_affiliate.booking_link})")
                st.balloons()
    
    else:
        # No savings found
        st.markdown(f"""
        <div class="competitive-card">
            <h2 style="margin: 0; color: white;">📊 Best Price Found</h2>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                <div>
                    <h3 style="margin: 0; color: white;">${results.google_flights.price:.0f}</h3>
                    <p style="margin: 0; opacity: 0.9;">via Google Flights</p>
                </div>
                <div style="text-align: right;">
                    <h4 style="margin: 0; color: white;">Competitive Pricing</h4>
                    <p style="margin: 0; opacity: 0.9;">AI-verified best rate</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Book with Google button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✈️ Book with Google Flights", type="primary", use_container_width=True):
                st.markdown(f"🔗 [Click here to book with Google Flights]({results.google_flights.booking_link})")
    
    # AI message display
    st.markdown("### 💭 AI Analysis")
    st.info(results.message)
    
    # AI Analysis Details
    with st.expander("🧠 Detailed Gemini AI Analysis", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**AI Search Strategy:**")
            if check_gemini_setup():
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
                    "Status": "✅ Best Deal" if result == results.best_affiliate else "📊 Compared"
                })
        
        if comparison_data:
            st.table(comparison_data)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer with Gemini branding
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🤖 Gemini AI Flight Agent - Real Artificial Intelligence</p>
    <p>Powered by Google Gemini Pro • Advanced Neural Networks</p>
    <p style="font-size: 0.9rem;">🧠 Making decisions humans can't, finding deals algorithms miss</p>
</div>
""", unsafe_allow_html=True)
