# Car Price Prediction Test Examples

## 🚗 Test Cases for XGBoost Model

Use these examples to test your Car Price Prediction System. Each example represents different car scenarios with expected price ranges.

### Required Input Fields:
1. **Year** (1990-2024)
2. **Engine Size** (0.5-8.0 liters)
3. **Annual Tax** ($0-$1000)
4. **Mileage** (0-500,000 miles)
5. **MPG** (10-100)
6. **Transmission** (Manual=1, Automatic=0)

---

## 📊 Test Examples

### 1. **Budget Car - Older High Mileage**
- **Year**: 2010
- **Engine Size**: 1.2
- **Tax**: $120
- **Mileage**: 85,000
- **MPG**: 45.0
- **Transmission**: Manual (1)
- **Expected Range**: $8,000 - $12,000

### 2. **Mid-Range Car - Good Condition**
- **Year**: 2018
- **Engine Size**: 1.6
- **Tax**: $180
- **Mileage**: 35,000
- **MPG**: 42.0
- **Transmission**: Automatic (0)
- **Expected Range**: $15,000 - $20,000

### 3. **Nearly New Car - Low Mileage**
- **Year**: 2022
- **Engine Size**: 1.5
- **Tax**: $200
- **Mileage**: 8,000
- **MPG**: 55.0
- **Transmission**: Manual (1)
- **Expected Range**: $20,000 - $25,000

### 4. **Performance Car - Larger Engine**
- **Year**: 2020
- **Engine Size**: 2.0
- **Tax**: $250
- **Mileage**: 15,000
- **MPG**: 35.0
- **Transmission**: Automatic (0)
- **Expected Range**: $25,000 - $30,000

### 5. **Economy Car - High MPG**
- **Year**: 2019
- **Engine Size**: 1.0
- **Tax**: $140
- **Mileage**: 25,000
- **MPG**: 65.0
- **Transmission**: Manual (1)
- **Expected Range**: $16,000 - $21,000

### 6. **Luxury Car - Premium Features**
- **Year**: 2021
- **Engine Size**: 2.5
- **Tax**: $350
- **Mileage**: 12,000
- **MPG**: 30.0
- **Transmission**: Automatic (0)
- **Expected Range**: $30,000 - $40,000

### 7. **Old Reliable - Classic**
- **Year**: 2005
- **Engine Size**: 1.8
- **Tax**: $160
- **Mileage**: 120,000
- **MPG**: 32.0
- **Transmission**: Manual (1)
- **Expected Range**: $5,000 - $8,000

### 8. **Brand New - Latest Model**
- **Year**: 2024
- **Engine Size**: 1.4
- **Tax**: $220
- **Mileage**: 500
- **MPG**: 58.0
- **Transmission**: Automatic (0)
- **Expected Range**: $25,000 - $35,000

### 9. **High Mileage Commuter**
- **Year**: 2016
- **Engine Size**: 1.3
- **Tax**: $150
- **Mileage**: 95,000
- **MPG**: 48.0
- **Transmission**: Manual (1)
- **Expected Range**: $10,000 - $14,000

### 10. **Sports Car - High Performance**
- **Year**: 2020
- **Engine Size**: 3.0
- **Tax**: $400
- **Mileage**: 18,000
- **MPG**: 25.0
- **Transmission**: Manual (1)
- **Expected Range**: $35,000 - $45,000

---

## 🧪 Testing Instructions

### Step 1: Access the System
1. Go to http://127.0.0.1:5000/
2. Register/Login to access the prediction page
3. Navigate to the prediction form

### Step 2: Test Each Example
1. Enter the values from each test case
2. Click "Predict Car Price"
3. Note the predicted price
4. Compare with expected range

### Step 3: Validation Checks
- **Reasonable Results**: Prices should make sense for the car specifications
- **Consistency**: Similar cars should have similar prices
- **Trends**: Newer cars, lower mileage, better MPG should generally cost more

### Step 4: Edge Cases to Test
- **Minimum Values**: Year=1990, Engine=0.5L, Tax=$0, Mileage=0, MPG=10
- **Maximum Values**: Year=2024, Engine=8.0L, Tax=$1000, Mileage=500000, MPG=100
- **Typical Values**: Year=2018, Engine=1.6L, Tax=$180, Mileage=35000, MPG=42

---

## 📈 Expected Behavior

### Price Factors (Generally):
- **↑ Higher Prices**: Newer year, lower mileage, higher MPG, moderate engine size
- **↓ Lower Prices**: Older year, higher mileage, lower MPG, very large engines
- **Tax Impact**: Higher tax usually indicates more expensive/newer cars
- **Transmission**: May vary by market preference

### System Validation:
- ✅ All predictions should be positive numbers
- ✅ No error messages should appear
- ✅ Results should save to your dashboard
- ✅ Similar inputs should give similar outputs

---

## 🔧 Troubleshooting

If you encounter issues:
1. **Check Terminal**: Look for error messages in the Flask console
2. **Verify Inputs**: Ensure all values are within valid ranges
3. **Browser Console**: Check for JavaScript errors
4. **Database**: Confirm predictions are saving to MongoDB

Happy Testing! 🚗💰