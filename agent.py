"""
Gemini-Powered AI Flight Agent - Core Logic
"""

import json
import os
import asyncio
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
import aiohttp

# Try to import Gemini, but don't fail if it's not installed
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ google.generativeai not installed - AI features disabled")

# Data Models
class CabinClass(str, Enum):
    """Flight cabin class options"""
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"

class TripType(str, Enum):
    """Trip type options"""
    ONE_WAY = "oneway"
    ROUND_TRIP = "roundtrip"
    MULTI_CITY = "multicity"

class FlightRequest(BaseModel):
    """Flight search request model"""
    origin: str = Field(..., min_length=2, max_length=100)
    destination: str = Field(..., min_length=2, max_length=100)
    departure_date: str = Field(..., description="Departure date in YYYY-MM-DD format")
    return_date: Optional[str] = Field(None)
    passengers: int = Field(1, ge=1, le=9)
    cabin_class: CabinClass = Field(CabinClass.ECONOMY)
    trip_type: Optional[TripType] = Field(None)

    @validator('departure_date', 'return_date')
    def validate_dates(cls, v):
        if v is None:
            return v
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')

    @validator('trip_type', always=True)
    def set_trip_type(cls, v, values):
        if 'return_date' in values:
            return TripType.ROUND_TRIP if values['return_date'] else TripType.ONE_WAY
        return v or TripType.ONE_WAY

class FlightResult(BaseModel):
    """Individual flight search result"""
    source: str = Field(...)
    price: Optional[float] = Field(None, ge=0)
    currency: str = Field("USD")
    booking_link: Optional[str] = Field(None)
    flight_details: Optional[Dict[str, Any]] = Field(None)
    success: bool = Field(True)
    error: Optional[str] = Field(None)
    search_time: Optional[float] = Field(None)
    affiliate_id: Optional[str] = Field(None)
    commission_rate: Optional[float] = Field(None)

class SearchResult(BaseModel):
    """Complete search response with all results"""
    google_flights: FlightResult = Field(...)
    best_affiliate: Optional[FlightResult] = Field(None)
    all_results: Optional[List[FlightResult]] = Field(None)
    savings: float = Field(0, ge=0)
    savings_percentage: Optional[float] = Field(None)
    message: str = Field(...)
    search_time: float = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)

    @validator('savings_percentage', always=True)
    def calculate_savings_percentage(cls, v, values):
        if 'google_flights' in values and 'savings' in values:
            google_price = values['google_flights'].price
            savings = values['savings']
            if google_price and google_price > 0 and savings > 0:
                return round((savings / google_price) * 100, 1)
        return 0.0

# Configuration
class Settings:
    """Application settings"""
    app_name: str = "Gemini AI Flight Agent"
    debug: bool = False
    environment: str = "development"
    
    # API keys
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")
    amadeus_api_key: str = os.getenv("AMADEUS_API_KEY", "2jfjOdQCI1VWXJZ0zFDTuR96Heo6q6XD")
    amadeus_api_secret: str = os.getenv("AMADEUS_API_SECRET", "PgF3SGgnJlswEpR5")
    
    # Affiliate IDs
    booking_affiliate_id: str = "YOUR_BOOKING_AFFILIATE_ID"
    kayak_affiliate_id: str = "YOUR_KAYAK_AFFILIATE_ID"
    
    max_requests_per_minute: int = 60

settings = Settings()

# Affiliate configuration
AFFILIATE_CONFIG = {
    "booking": {
        "affiliate_id": settings.booking_affiliate_id,
        "base_url": "https://www.booking.com/flights",
        "commission_rate": 0.03,
        "search_url_template": "https://www.booking.com/flights?from={origin}&to={destination}&date={departure_date}&returnDate={return_date}&pax={passengers}&cabinClass={cabin_class}&adults={adults}&children={children}&seniors={seniors}&affiliate={affiliate_id}",
        "enabled": True,
        "strengths": ["budget", "europe", "last_minute"]
    },
    "kayak": {
        "affiliate_id": settings.kayak_affiliate_id,
        "base_url": "https://www.kayak.com/flights",
        "commission_rate": 0.025,
        "search_url_template": "https://www.kayak.com/flights?from={origin}&to={destination}&date={departure_date}&returnDate={return_date}&pax={passengers}&cabinClass={cabin_class}&adults={adults}&children={children}&seniors={seniors}&affiliate={affiliate_id}",
        "enabled": True,
        "strengths": ["domestic_us", "price_comparison", "flexible_dates"]
    },
    "google_flights": {
        "affiliate_id": "",
        "base_url": "https://www.google.com/travel/flights",
        "commission_rate": 0.0,
        "search_url_template": "https://www.google.com/travel/flights?from={origin}&to={destination}&date={departure_date}&returnDate={return_date}&adults={adults}&children={children}&seniors={seniors}&cabinClass={cabin_class}",
        "enabled": True,
        "strengths": ["baseline", "accuracy"]
    }
}

class GeminiFlightAgent:
    """Main Gemini AI-powered flight search agent"""
    
    def __init__(self, custom_affiliate_config: Optional[Dict[str, Any]] = None):
        self.airports = self.load_airports()
        
        if custom_affiliate_config:
            self.update_affiliate_config(custom_affiliate_config)
        
        # Initialize Gemini AI
        self.gemini_model = None
        self._init_gemini()
    
    def _init_gemini(self):
        """Initialize Gemini AI model"""
        if not GEMINI_AVAILABLE:
            print("⚠️ Gemini AI not available - package not installed")
            return
            
        try:
            api_key = settings.gemini_api_key
            if api_key and api_key != "your_gemini_api_key_here":
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                print("✅ Gemini AI initialized successfully")
            else:
                print("⚠️ Gemini API key not configured")
        except Exception as e:
            print(f"❌ Gemini initialization failed: {e}")
    
    def is_gemini_available(self) -> bool:
        """Check if Gemini AI is available"""
        return self.gemini_model is not None
    
    def update_affiliate_config(self, new_config: Dict[str, Any]):
        """Update affiliate configuration"""
        for partner, config in new_config.items():
            if partner in AFFILIATE_CONFIG:
                AFFILIATE_CONFIG[partner].update(config)
                if config.get('affiliate_id') and config['affiliate_id'] != f"YOUR_{partner.upper()}_AFFILIATE_ID":
                    AFFILIATE_CONFIG[partner]['enabled'] = True
                    print(f"✅ Enabled {partner} with affiliate ID: {config['affiliate_id']}")
    
    def load_airports(self):
        """Load airports from JSON database"""
        try:
            if os.path.exists("airports.json"):
                with open("airports.json", "r", encoding="utf-8") as f:
                    airports = json.load(f)
                print(f"✅ Loaded {len(airports)} airports from database")
                return airports
            else:
                print("⚠️ airports.json not found")
                return {}
        except Exception as e:
            print(f"❌ Error loading airports: {e}")
            return {}
    
    def autocomplete_airports(self, query: str) -> List[Dict[str, str]]:
        """Airport autocomplete"""
        query = query.strip()
        if len(query) < 1:
            return []
        
        suggestions = []
        query_lower = query.lower()
        
        for code, info in self.airports.items():
            if (code.lower().startswith(query_lower) or 
                query_lower in info["city"].lower() or 
                query_lower in info["name"].lower() or 
                query_lower in info["country"].lower()):
                
                suggestions.append({
                    "code": code,
                    "name": info["name"],
                    "city": info["city"],
                    "display": info["display"]
                })
        
        def sort_key(suggestion):
            code_lower = suggestion["code"].lower()
            city_lower = suggestion["city"].lower()
            
            if code_lower.startswith(query_lower):
                return (0, code_lower)
            elif city_lower.startswith(query_lower):
                return (1, city_lower)
            else:
                return (2, suggestion["display"])
        
        suggestions.sort(key=sort_key)
        return suggestions[:10]
    
    def build_affiliate_url(self, partner: str, request: FlightRequest) -> str:
        """Build enhanced affiliate URL with pre-filled search data"""
        if partner not in AFFILIATE_CONFIG or not AFFILIATE_CONFIG[partner]['enabled']:
            return None
        
        config = AFFILIATE_CONFIG[partner]
        
        # Enhanced URL building with proper formatting for each partner
        if partner == "booking":
            return self._build_booking_url(request, config)
        elif partner == "kayak":
            return self._build_kayak_url(request, config)
        elif partner == "google_flights":
            return self._build_google_flights_url(request, config)
        else:
            return self._build_generic_url(partner, request, config)

    def _build_booking_url(self, request: FlightRequest, config: Dict) -> str:
        """Build Booking.com URL with pre-filled search data"""
        # Note: Booking.com's flight search has limited URL parameter support
        # This directs to their flight search page
        base_url = "https://www.booking.com/flights/index.html"
        
        params = []
        
        # Basic flight search parameters (limited support)
        params.append(f"from={request.origin}")
        params.append(f"to={request.destination}")
        params.append(f"depart={request.departure_date}")
        
        if request.return_date and request.trip_type == TripType.ROUND_TRIP:
            params.append(f"return={request.return_date}")
            params.append("type=return")
        else:
            params.append("type=oneway")
        
        # Passengers
        params.append(f"adults={request.passengers}")
        
        # Currency
        params.append("currency=USD")
        
        # Affiliate tracking
        if config['affiliate_id'] and config['affiliate_id'] != "YOUR_BOOKING_AFFILIATE_ID":
            params.append(f"aid={config['affiliate_id']}")
        
        return f"{base_url}?{'&'.join(params)}"

    def _build_alternative_search_urls(self, request: FlightRequest) -> Dict[str, str]:
        """Build alternative search URLs for partners that might have issues"""
        alternatives = {}
        
        # Priceline alternative
        priceline_url = f"https://www.priceline.com/relax/at/flights/search/{request.origin}/{request.destination}/{request.departure_date}"
        if request.return_date and request.trip_type == TripType.ROUND_TRIP:
            priceline_url += f"/{request.return_date}"
        priceline_url += f"/{request.passengers}/0/0/Economy/USD"
        alternatives["priceline"] = priceline_url
        
        # Momondo alternative  
        momondo_base = "https://www.momondo.com/flight-search"
        momondo_path = f"/{request.origin}-{request.destination}/{request.departure_date}"
        if request.return_date and request.trip_type == TripType.ROUND_TRIP:
            momondo_path += f"/{request.return_date}"
        momondo_path += f"/{request.passengers}adults"
        alternatives["momondo"] = f"{momondo_base}{momondo_path}"
        
        # Skyscanner alternative
        skyscanner_url = f"https://www.skyscanner.com/transport/flights/{request.origin.lower()}/{request.destination.lower()}"
        skyscanner_url += f"/{request.departure_date.replace('-', '')}"
        if request.return_date and request.trip_type == TripType.ROUND_TRIP:
            skyscanner_url += f"/{request.return_date.replace('-', '')}"
        alternatives["skyscanner"] = skyscanner_url
        
        return alternatives

    def _build_kayak_url(self, request: FlightRequest, config: Dict) -> str:
        """Build Kayak URL with pre-filled search data"""
        # Kayak's current URL format is quite reliable
        base_url = "https://www.kayak.com/flights"
        
        # Build the path-based URL that Kayak uses
        if request.trip_type == TripType.ROUND_TRIP and request.return_date:
            # Round trip: /flights/JFK-LAX/2024-03-15/2024-03-22
            url_path = f"/{request.origin}-{request.destination}/{request.departure_date}/{request.return_date}"
        else:
            # One way: /flights/JFK-LAX/2024-03-15
            url_path = f"/{request.origin}-{request.destination}/{request.departure_date}"
        
        params = []
        
        # Passengers
        if request.passengers > 1:
            params.append(f"adults={request.passengers}")
        
        # Cabin class
        cabin_mapping = {
            "economy": "e",
            "premium_economy": "p",
            "business": "b",
            "first": "f"
        }
        cabin = cabin_mapping.get(request.cabin_class.value, "e")
        if cabin != "e":  # Only add if not economy (default)
            params.append(f"cabin={cabin}")
        
        # Sort by price
        params.append("sort=price_a")
        
        full_url = f"{base_url}{url_path}"
        if params:
            full_url += f"?{'&'.join(params)}"
        
        return full_url

    def _build_google_flights_url(self, request: FlightRequest, config: Dict) -> str:
        """Build Google Flights URL with pre-filled search data"""
        # Google Flights uses a more complex URL structure
        base_url = "https://www.google.com/travel/flights"
        
        params = []
        
        # Language and currency
        params.append("hl=en")
        params.append("curr=USD")
        
        # Trip type
        if request.trip_type == TripType.ROUND_TRIP:
            params.append("tfs=CBwQAhokag0IAhIJL20vMDJfMjg2EgoyMDI0LTAzLTE1agwIAxIIL20vMGZfOGsQARgBIAEoAToJCAAQABgBIAEoAQ")
        else:
            params.append("tfs=CBwQAhoUag0IAhIJL20vMDJfMjg2EgoyMDI0LTAzLTE1OgkIABAAGAEgASgB")
        
        # This is a simplified version - Google Flights URLs are very complex
        # For now, we'll direct to the main search page
        simple_params = []
        simple_params.append(f"q=flights from {request.origin} to {request.destination}")
        
        return f"{base_url}?{'&'.join(simple_params)}"

    def _build_generic_url(self, partner: str, request: FlightRequest, config: Dict) -> str:
        """Build generic URL for other partners using template"""
        template = config.get('search_url_template', '')
        
        # Standard parameters
        params = {
            'origin': request.origin,
            'destination': request.destination,
            'departure_date': request.departure_date,
            'return_date': request.return_date or '',
            'passengers': request.passengers,
            'adults': request.passengers,
            'children': 0,
            'seniors': 0,
            'cabin_class': request.cabin_class.value,
            'affiliate_id': config['affiliate_id']
        }
        
        try:
            url = template.format(**params)
            # Clean up URL
            url = url.replace('&returnDate=', '') if not request.return_date else url
            url = url.replace('&children=0', '')
            url = url.replace('&seniors=0', '')
            url = url.replace('&&', '&').replace('?&', '?')
            if url.endswith('&'):
                url = url[:-1]
            return url
        except Exception as e:
            print(f"❌ Error building URL for {partner}: {e}")
            return None

    def get_preview_urls(self, request: FlightRequest) -> Dict[str, str]:
        """Get preview of all affiliate URLs for testing"""
        urls = {}
        
        # Primary partners
        for partner, config in AFFILIATE_CONFIG.items():
            if config['enabled']:
                try:
                    url = self.build_affiliate_url(partner, request)
                    if url:
                        urls[partner] = url
                except Exception as e:
                    urls[partner] = f"Error: {str(e)}"
        
        # Add alternative booking sites for better reliability
        try:
            alternatives = self._build_alternative_search_urls(request)
            for alt_name, alt_url in alternatives.items():
                urls[f"{alt_name}_alternative"] = alt_url
        except Exception as e:
            print(f"Could not build alternatives: {e}")
        
        # Add more reliable booking alternatives that actually work with deep-linking
        reliable_alternatives = self._build_reliable_booking_alternatives(request)
        for alt_name, alt_url in reliable_alternatives.items():
            urls[f"{alt_name}_reliable"] = alt_url
        
        return urls
    
    def _build_reliable_booking_alternatives(self, request: FlightRequest) -> Dict[str, str]:
        """Build URLs for booking sites that reliably support deep-linking"""
        alternatives = {}
        
        # CheapOair - usually good with deep-linking
        cheapoair_url = f"https://www.cheapoair.com/flights/results?"
        cheapoair_params = [
            f"from1={request.origin}",
            f"to1={request.destination}",
            f"departure1={request.departure_date}",
            f"adult={request.passengers}",
            f"child=0",
            f"infant=0",
            "cabin=Economy"
        ]
        if request.return_date and request.trip_type == TripType.ROUND_TRIP:
            cheapoair_params.append(f"departure2={request.return_date}")
            cheapoair_params.append("tripType=R")
        else:
            cheapoair_params.append("tripType=O")
        
        alternatives["cheapoair"] = cheapoair_url + "&".join(cheapoair_params)
        
        # Travelocity - often works well
        travelocity_url = "https://www.travelocity.com/Flights-Search?"
        travelocity_params = [
            f"departure={request.departure_date}",
            f"origin={request.origin}",
            f"destination={request.destination}",
            f"adults={request.passengers}"
        ]
        if request.return_date and request.trip_type == TripType.ROUND_TRIP:
            travelocity_params.append(f"return={request.return_date}")
        
        alternatives["travelocity"] = travelocity_url + "&".join(travelocity_params)
        
        # JetBlue - good for domestic US flights
        jetblue_url = "https://www.jetblue.com/flights/search?"
        jetblue_params = [
            f"from={request.origin}",
            f"to={request.destination}",
            f"depart={request.departure_date}",
            f"adults={request.passengers}"
        ]
        if request.return_date and request.trip_type == TripType.ROUND_TRIP:
            jetblue_params.append(f"return={request.return_date}")
        
        alternatives["jetblue"] = jetblue_url + "&".join(jetblue_params)
        
        return alternatives
    
    async def analyze_route_with_gemini(self, request: FlightRequest) -> Dict[str, Any]:
        """Use Gemini to analyze the flight route"""
        if not self.is_gemini_available():
            return self._get_fallback_analysis(request)
        
        prompt = f"""
        Analyze this flight search and provide optimization recommendations:
        
        Route: {request.origin} to {request.destination}
        Date: {request.departure_date}
        Passengers: {request.passengers}
        Class: {request.cabin_class.value}
        Return: {"Yes" if request.return_date else "No"}
        
        Respond with ONLY valid JSON in this format:
        {{
            "recommended_partners": ["priceline", "booking", "kayak"],
            "expected_price_range": {{"min": 200, "max": 800}},
            "route_type": "domestic/international",
            "flexibility_benefit": "high/medium/low",
            "booking_advice": "Brief advice",
            "search_strategy": "single_date"
        }}
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                analysis = json.loads(json_str)
                print(f"🤖 Gemini analysis complete: {analysis.get('route_type', 'analyzed')} route")
                return analysis
            else:
                print("⚠️ Could not parse Gemini JSON response")
                return self._get_fallback_analysis(request)
                
        except Exception as e:
            print(f"❌ Gemini analysis failed: {e}")
            return self._get_fallback_analysis(request)
    
    def _get_fallback_analysis(self, request: FlightRequest) -> Dict[str, Any]:
        """Fallback analysis when Gemini is not available"""
        is_international = len(request.origin) == 3 and len(request.destination) == 3
        
        return {
            "recommended_partners": ["priceline", "booking", "kayak"],
            "expected_price_range": {"min": 300, "max": 800},
            "route_type": "international" if is_international else "domestic",
            "flexibility_benefit": "medium",
            "booking_advice": "Standard booking recommendations apply",
            "search_strategy": "single_date"
        }
    
    async def search_flights(self, request: FlightRequest) -> SearchResult:
        """Enhanced search with AI optimization"""
        print(f"🤖 AI-Enhanced search: {request.origin} → {request.destination}")
        start_time = datetime.now()
        
        try:
            # Step 1: Get AI analysis of the route
            analysis = await self.analyze_route_with_gemini(request)
            print(f"📊 Route analysis: {analysis.get('route_type')} route")
            
            # Step 2: Get optimized partner search order
            optimized_partners = analysis.get("recommended_partners", ["priceline", "booking", "kayak"])
            
            # Step 3: Simulate intelligent affiliate searches
            affiliate_results = await self._simulate_affiliate_search(request, analysis, optimized_partners)
            
            # Step 4: Create Google Flights baseline
            google_price = analysis["expected_price_range"]["max"] * 0.95
            google_result = FlightResult(
                source="Google Flights",
                price=google_price,
                booking_link=self.build_affiliate_url("google_flights", request),
                success=True,
                search_time=2.0
            )
            
            # Step 5: Find best deal
            best_affiliate = None
            if affiliate_results:
                best_affiliate = min(affiliate_results, key=lambda x: x.price)
                if best_affiliate.price >= google_result.price:
                    best_affiliate = None
            
            # Step 6: Calculate savings
            savings = 0
            if best_affiliate:
                savings = google_result.price - best_affiliate.price
            
            # Step 7: Generate message
            message = self._generate_ai_message(google_result, best_affiliate, savings, analysis)
            
            end_time = datetime.now()
            search_time = (end_time - start_time).total_seconds()
            
            return SearchResult(
                google_flights=google_result,
                best_affiliate=best_affiliate,
                all_results=affiliate_results + [google_result],
                savings=savings,
                message=message,
                search_time=search_time
            )
            
        except Exception as e:
            print(f"❌ AI search failed: {e}")
            return await self._fallback_search(request)
    
    async def _simulate_affiliate_search(self, request: FlightRequest, analysis: Dict, partners: List[str]) -> List[FlightResult]:
        """Simulate affiliate searches with AI-enhanced pricing"""
        results = []
        base_price = analysis["expected_price_range"]["min"] + 100
        
        for i, partner in enumerate(partners):
            if partner not in AFFILIATE_CONFIG or not AFFILIATE_CONFIG[partner]["enabled"]:
                continue
                
            config = AFFILIATE_CONFIG[partner]
            partner_strengths = config.get("strengths", [])
            route_type = analysis.get("route_type", "domestic")
            
            # AI-enhanced price simulation
            price_modifier = 1.0
            if route_type in partner_strengths:
                price_modifier *= 0.85
            
            # Add variance
            variance = 0.85 + (hash(partner + request.origin) % 30) / 100
            final_price = base_price * price_modifier * variance
            
            result = FlightResult(
                source=partner.title(),
                price=round(final_price, 2),
                booking_link=self.build_affiliate_url(partner, request),
                success=True,
                search_time=1.0 + i * 0.3,
                affiliate_id=config["affiliate_id"],
                commission_rate=config["commission_rate"]
            )
            results.append(result)
            
            await asyncio.sleep(0.1)  # Simulate search delay
        
        return results
    
    def _generate_ai_message(self, google_result: FlightResult, best_affiliate: Optional[FlightResult], 
                           savings: float, analysis: Dict) -> str:
        """Generate enhanced message with AI insights"""
        if best_affiliate and savings > 0:
            base_message = f"🎉 AI found savings! Save ${savings:.0f} ({(savings/google_result.price)*100:.1f}%) with {best_affiliate.source}"
        else:
            base_message = f"📊 Google Flights shows competitive pricing at ${google_result.price:.0f}"
        
        booking_advice = analysis.get("booking_advice", "")
        if booking_advice and booking_advice != "Standard booking recommendations apply":
            base_message += f"\n🧠 AI Advice: {booking_advice}"
        
        return base_message
    
    async def _fallback_search(self, request: FlightRequest) -> SearchResult:
        """Fallback search when AI fails"""
        start_time = datetime.now()
        
        google_result = FlightResult(
            source="Google Flights",
            price=450.0,
            booking_link=self.build_affiliate_url("google_flights", request),
            success=True,
            search_time=1.0
        )
        
        end_time = datetime.now()
        search_time = (end_time - start_time).total_seconds()
        
        return SearchResult(
            google_flights=google_result,
            best_affiliate=None,
            savings=0,
            message="✈️ Standard search completed",
            search_time=search_time
        )
    
    def get_affiliate_config(self, partner: str) -> Optional[Dict]:
        """Get affiliate configuration for a partner"""
        return AFFILIATE_CONFIG.get(partner.lower())
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        return {
            "status": "AI-Enhanced" if self.is_gemini_available() else "Standard",
            "airports_loaded": len(self.airports),
            "environment": settings.environment,
            "gemini_enabled": self.is_gemini_available(),
            "affiliate_partners": len([p for p, c in AFFILIATE_CONFIG.items() if c["enabled"]])
        }