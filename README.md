# ✈️ Flight Agent - Streamlit Edition

A modern, beautiful Streamlit application that helps you find better flight deals than Google Flights.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## ✨ Features

- **Smart Airport Autocomplete** - Type airport codes, city names, or airport names
- **Beautiful UI** - Modern, responsive design with gradient backgrounds
- **Price Comparison** - Compare Google Flights prices with affiliate deals
- **Real-time Search** - Find the best deals instantly
- **Affiliate Integration** - Direct booking links to save money
- **Responsive Design** - Works on desktop and mobile devices

## 🏗️ Project Structure

```
Flight-Agent/
├── app.py              # Main Streamlit application
├── agent.py            # Flight search logic and models
├── requirements.txt    # Python dependencies
├── airports.json       # Airport database
└── README.md          # This file
```

## 🔧 Configuration

The app includes:
- **Affiliate Partners**: Expedia, Booking.com, Kayak
- **Commission Rates**: Configurable affiliate commissions
- **Search Settings**: Customizable search parameters
- **Environment**: Development/production settings

## 💡 How It Works

1. **User Input**: Enter origin, destination, dates, and preferences
2. **Smart Search**: The agent searches multiple sources simultaneously
3. **Price Comparison**: Compares Google Flights with affiliate deals
4. **Best Deal**: Shows the best price and savings
5. **Direct Booking**: Provides direct links to book the best deal

## 🛠️ Development

- **Frontend**: Streamlit with custom CSS
- **Backend**: Python with async support
- **Models**: Pydantic for data validation
- **Data**: JSON-based airport database

## 📱 Usage

1. **Enter Flight Details**: Fill in origin, destination, dates, passengers, and cabin class
2. **Autocomplete**: Use the smart airport suggestions for accurate input
3. **Search**: Click "Find Better Deals" to start the search
4. **Results**: View price comparison and savings
5. **Book**: Click the booking link to secure your deal

## 📄 License

This project is open source and available under the MIT License.

---

**Happy flying! ✈️** Find the best deals and save money on your next trip!
