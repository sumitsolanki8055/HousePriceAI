import numpy as np

def test_prediction(index=None):
    """
    Picks a house from the test set, runs the AI model, 
    and compares the result with the real price.
    """
    
    # 1. Pick a random house if no index is provided
    if index is None:
        index = np.random.randint(0, len(X_test))
    
    # 2. Grab the data for that single house
    # We need to reshape it to (1, -1) because the model expects a batch, even of 1
    house_features = X_test[index].reshape(1, -1)
    
    # 3. ASK THE BRAIN (Make the prediction)
    # verbose=0 keeps the logs quiet
    predicted_scaled = model.predict(house_features, verbose=0)
    
    # 4. Convert math back to money (Inverse Transform)
    # We assume y_test is also scaled. If your y_test is already dollars, skip the inverse on it.
    predicted_price = y_scaler.inverse_transform(predicted_scaled)[0][0]
    real_price = y_scaler.inverse_transform(y_test[index].reshape(-1, 1))[0][0]
    
    # 5. Calculate Logic
    diff = predicted_price - real_price
    
    # 6. The "Receipt" Printout
    print("\n" + "="*40)
    print(f" Test Index: {index}")
    # Note: We can't easily get Bed/Bath info here unless we kept a reference 
    # to the original unscaled DataFrame. We are looking at raw math inputs now.
    print("-" * 20)
    print(f" REAL Price:      ${real_price:,.0f}")
    print(f" AI Prediction:   ${predicted_price:,.0f}")
    
    # Color code the difference (if supported by your terminal/notebook)
    if diff > 0:
        print(f" Overpriced by:   +${diff:,.0f}")
    else:
        print(f" Underpriced by:  ${diff:,.0f}")
        
    print("="*40 + "\n")

# --- Run it 3 times to see different results ---
test_prediction()
test_prediction()
test_prediction()
