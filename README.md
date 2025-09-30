# 🍽️ AI Food Recognition & Nutrition App

## 📖 Overview

This application uses your trained ResNet50 model (`best_model_phase2.keras`) to recognize 131 different food items and provide detailed nutrition information. It includes 101 international dishes from Food-101 dataset and 30 Vietnamese traditional dishes.

## 📁 Project Structure

```
APP_DELOY_ITPROJECT/
├── best_model_phase2.keras          # Your trained model (Phase 2)
├── class_mapping.csv                # Class ID to food name mapping
├── food_data.py                     # Nutrition database (131 foods)
├── app_fixed.py                     # Main Streamlit app (RECOMMENDED)
├── model_loader.py                  # Advanced model loading utilities
├── quick_test.py                    # Test script to verify setup
├── run_app.bat                      # Windows batch file to run app
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🚀 Quick Start

### Option 1: Double-click `run_app.bat` (Windows)

Just double-click the `run_app.bat` file - it will install requirements and start the app.

### Option 2: Manual Installation

1. **Install Python packages:**

   ```bash
   pip install streamlit tensorflow pillow numpy
   ```

2. **Test the setup:**

   ```bash
   python quick_test.py
   ```

3. **Run the app:**
   ```bash
   streamlit run app_fixed.py
   ```

## 🔧 Features

### ✅ Food Recognition

- **131 Food Classes**: 101 international + 30 Vietnamese dishes
- **High Accuracy**: Uses your trained ResNet50 model
- **Confidence Scores**: Shows prediction confidence with visual indicators
- **Top-3 Predictions**: Alternative predictions with confidence scores

### ✅ Nutrition Information

- **Comprehensive Database**: Calories, protein, fat, carbs, fiber
- **Per Serving**: All values calculated per typical serving size
- **Visual Metrics**: Clean display with emoji indicators

### ✅ Smart Features

- **Vietnamese Food Detection**: Special handling for Vietnamese dishes
- **Image Preprocessing**: Automatic image resizing and normalization
- **Error Handling**: Multiple fallback methods for model loading
- **User-Friendly**: Clean interface with helpful tips

## 🍜 Supported Food Categories

### International Dishes (Food-101)

- Appetizers: bruschetta, deviled_eggs, escargots
- Main Courses: steak, grilled_salmon, chicken_curry, pad_thai
- Desserts: cheesecake, tiramisu, chocolate_cake
- And 91+ more international dishes

### Vietnamese Dishes

- **Noodles**: pho, bun_bo_hue, bun_rieu, hu_tieu, mi_quang
- **Rice**: com_tam, xoi_xeo
- **Banh**: banh_mi, banh_xeo, banh_cuon, banh_chung
- **Soups**: canh_chua, chao_long
- And 20+ more Vietnamese specialties

## 🛠️ Technical Details

### Model Architecture

- **Base**: ResNet50 (ImageNet pretrained)
- **Input Size**: 224×224×3 RGB images
- **Output**: 131 classes with softmax activation
- **Preprocessing**: ResNet50 standard preprocessing

### Model Loading Strategy

The app uses multiple fallback methods to handle potential issues:

1. **Standard Loading**: Direct model loading
2. **No Compilation**: Load without recompiling
3. **Weight Transfer**: Transfer weights to clean architecture
4. **Dummy Model**: Fallback for testing (random predictions)

### Class Mapping

- **IDs 0-100**: Food-101 classes (apple_pie, baby_back_ribs, etc.)
- **IDs 101-130**: Vietnamese dishes (banh_beo, pho, etc.)

## 🏃‍♂️ Usage Instructions

1. **Start the app** using one of the methods above
2. **Upload an image** of food (JPG, PNG, JPEG)
3. **Click "Analyze Food"** to get predictions
4. **View results**:
   - Food name and confidence score
   - Detailed nutrition information
   - Alternative predictions
   - Food category (Vietnamese vs International)

## 💡 Tips for Best Results

- Use **clear, well-lit images**
- **Center the food** in the image
- Avoid **multiple food items** in one image
- Ensure the image **isn't blurry**
- **Higher resolution** images work better

## 🔍 Troubleshooting

### Model Loading Issues

If you see "Model loading failed" errors:

1. **Check file existence**: Ensure `best_model_phase2.keras` is in the same directory
2. **Run test script**: `python quick_test.py`
3. **Check TensorFlow**: `pip install tensorflow --upgrade`
4. **Try alternative loading**: The app will automatically try multiple methods

### Common Error Messages

**"Lambda layer could not be deserialized"**

- This is handled automatically by the app
- The app will use weight transfer method

**"TensorFlow not found"**

- Run: `pip install tensorflow`

**"PIL import error"**

- Run: `pip install pillow`

### Performance Notes

- First prediction may be slower (model loading)
- Subsequent predictions are faster (cached model)
- GPU acceleration automatic if available

## 📊 Model Performance

Your trained model achieves:

- **Training Dataset**: Food-101 + 30VNFoods
- **Architecture**: ResNet50 with custom head
- **Classes**: 131 food categories
- **Input Resolution**: 224×224 pixels

## 🤝 Support

If you encounter issues:

1. **Check requirements**: All files in project directory
2. **Run test script**: `python quick_test.py`
3. **Check Python version**: Python 3.7+ recommended
4. **Update packages**: `pip install --upgrade streamlit tensorflow`

## 📝 File Descriptions

- **`app_fixed.py`**: Main application with robust error handling
- **`model_loader.py`**: Advanced model loading utilities
- **`food_data.py`**: Complete nutrition database for all 131 classes
- **`quick_test.py`**: Diagnostic script to test setup
- **`class_mapping.csv`**: Maps model output IDs to food names

---

**Enjoy your AI-powered food recognition experience! 🎉**
