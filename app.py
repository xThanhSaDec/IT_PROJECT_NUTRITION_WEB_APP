"""
Food Recognition App - Stable Production Version
Handles model loading issues gracefully and provides a working app
"""
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Configure page
st.set_page_config(
    page_title="🍽️ Food Recognition App", 
    page_icon="🍽️",
    layout="centered"
)

# Load food database
try:
    from food_data import NUTRITION_DATABASE, CLASS_MAPPING, get_display_name
    FOOD_DATA_LOADED = True
except ImportError:
    st.error("❌ Food database not found. Please ensure food_data.py is available.")
    FOOD_DATA_LOADED = False
    CLASS_MAPPING = {}
    NUTRITION_DATABASE = {}

# streamlit run app.py  
# uvicorn main:app --reload


#command about git on vscode
#.gitignore
#notepad .gitignore
#*.keras

#day git len
# git init
# git branch -M main
# git remote remove origin
# git remote add origin https://github.com/xThanhSaDec/IT_PROJECT_NUTRITION_WEB_APP.git
# git add .
# git commit -m "Initial commit without keras model"
# git push -u origin main
# git push origin main --force: Đè code local lên repo GitHub (xóa sạch remote cũ)

@st.cache_resource  
def load_model_stable():
    """
    Stable model loading with clear status reporting
    """
    model_path = 'best_model_phase2.keras'
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None, "file_not_found"
    
    file_size = os.path.getsize(model_path) / (1024*1024)
    st.info(f"📁 Model file found: {file_size:.1f} MB")
    
    # Try loading methods in order of preference
    loading_methods = [
        ("Standard loading", load_with_custom_objects),
        ("Without compilation", load_without_compile),
        ("Functional fallback", create_functional_fallback)
    ]
    
    for method_name, method_func in loading_methods:
        st.write(f"🔄 Trying: {method_name}...")
        try:
            model = method_func(model_path)
            if model is not None:
                st.success(f"✅ Success: {method_name}")
                return model, method_name.lower().replace(" ", "_")
        except Exception as e:
            st.warning(f"❌ {method_name} failed: {str(e)[:80]}...")
            continue
    
    st.error("❌ All model loading methods failed")
    return None, "all_failed"

def load_with_custom_objects(model_path):
    """Method 1: Load with custom objects for Lambda layer"""
    from tensorflow.keras.applications.resnet50 import preprocess_input
    custom_objects = {'preprocess_input': preprocess_input}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

def load_without_compile(model_path):
    """Method 2: Load without compilation"""
    return tf.keras.models.load_model(model_path, compile=False)

def create_functional_fallback(model_path):
    """Method 3: Create functional model that works"""
    st.info("Model has Lambda layer serialization issues - using functional fallback")
    
    # Create a working ResNet50-based model
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',  # Use pretrained weights
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    base_model.trainable = False
    
    # Add classification head
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(131, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Note about accuracy
    st.warning("⚠️ Using ImageNet + random final layer - accuracy will be limited until model is retrained")
    st.info("💡 This provides a stable foundation for your app. You can improve the model later.")
    
    return model

def preprocess_image_stable(image):
    """
    Stable image preprocessing that works with any model type
    """
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input
    image = image.resize((224, 224))
    
    # Convert to array and normalize
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0  # Normalize to [0,1]
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_food_stable(model, image_array, model_type):
    """
    Stable prediction function
    """
    try:
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        
        # Get top prediction
        predicted_class_id = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_id])
        
        # Validate prediction
        if predicted_class_id >= len(CLASS_MAPPING):
            return {"error": f"Invalid prediction ID: {predicted_class_id}"}
        
        # Get food information
        if predicted_class_id in CLASS_MAPPING:
            class_name = CLASS_MAPPING[predicted_class_id]
            display_name = get_display_name(class_name)
            nutrition = NUTRITION_DATABASE.get(class_name, {})
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = []
            for idx in top_3_indices:
                if idx in CLASS_MAPPING:
                    pred_class = CLASS_MAPPING[idx]
                    pred_confidence = float(predictions[0][idx])
                    top_3_predictions.append({
                        "name": get_display_name(pred_class),
                        "confidence": pred_confidence
                    })
            
            # Add model performance note
            performance_note = ""
            if model_type == "functional_fallback":
                performance_note = "Note: Using fallback model - predictions may not be accurate until model is retrained"
            
            return {
                "success": True,
                "food_name": display_name,
                "class_name": class_name,
                "confidence": confidence,
                "nutrition": nutrition,
                "top_3_predictions": top_3_predictions,
                "model_type": model_type,
                "performance_note": performance_note
            }
        else:
            return {"error": f"Unknown class ID: {predicted_class_id}"}
            
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

def display_results(result):
    """Display prediction results"""
    if "error" in result:
        st.error(f"❌ {result['error']}")
        return
    
    if not result.get("success"):
        st.error("❌ Analysis failed")
        return
    
    # Performance note
    if result.get("performance_note"):
        st.warning(f"⚠️ {result['performance_note']}")
    
    # Main result
    st.success(f"🍽️ **{result['food_name']}**")
    
    # Confidence
    confidence_pct = result['confidence'] * 100
    if confidence_pct >= 70:
        st.success(f"🎯 Confidence: {confidence_pct:.1f}%")
    elif confidence_pct >= 40:
        st.warning(f"⚠️ Confidence: {confidence_pct:.1f}%")
    else:
        st.error(f"❓ Low Confidence: {confidence_pct:.1f}%")
    
    # Progress bar
    st.progress(result['confidence'])
    
    # Nutrition information
    nutrition = result.get('nutrition', {})
    if nutrition:
        st.subheader("🥗 Nutrition Information")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        metrics = [
            ("🔥 Calories", nutrition.get('calories', 0), ""),
            ("🥩 Protein", nutrition.get('protein', 0), "g"),
            ("🧈 Fat", nutrition.get('fat', 0), "g"),
            ("🍞 Carbs", nutrition.get('carbs', 0), "g"),
            ("🌾 Fiber", nutrition.get('fiber', 0), "g")
        ]
        
        for col, (label, value, unit) in zip([col1, col2, col3, col4, col5], metrics):
            with col:
                st.metric(label, f"{value}{unit}")
    
    # Top predictions
    top_3 = result.get('top_3_predictions', [])
    if len(top_3) > 1:
        st.subheader("🏆 Top 3 Predictions")
        for i, pred in enumerate(top_3, 1):
            confidence_pct = pred["confidence"] * 100
            if confidence_pct > 1:
                st.write(f"{i}. **{pred['name']}** - {confidence_pct:.1f}%")
    
    # Food category
    class_name = result.get('class_name', '').lower()
    vietnamese_foods = ['banh', 'bun', 'pho', 'com', 'goi', 'hu', 'mi', 'nem', 'xoi', 'ca', 'canh', 'cao', 'chao']
    
    if any(vn_word in class_name for vn_word in vietnamese_foods):
        st.info("🇻🇳 Vietnamese Cuisine")
    else:
        st.info("🌍 International Cuisine")

def main():
    # Header
    st.title("🍽️ Food Recognition & Nutrition App")
    st.markdown("### Stable Version - Ready for Production")
    
    # Model loading section
    # with st.expander("🤖 Model Status", expanded=True):
    #     model, model_type = load_model_stable()
        
    #     if model is None:
    #         st.error("❌ Could not load any model. Please check your setup.")
    #         st.stop()
    #     else:
    #         if model_type == "functional_fallback":
    #             st.info("📝 **Development Plan:**")
    #             st.write("1. ✅ **Current**: Stable app with basic functionality")
    #             st.write("2. 🔄 **Next**: Retrain model without Lambda layer issues")
    #             st.write("3. 🎯 **Future**: Deploy improved model for better accuracy")
    
    # File upload
    st.subheader("📸 Upload Food Image")
    uploaded_file = st.file_uploader(
        "Choose a food image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of food"
    )
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("🔍 Analyze Food", type="primary", use_container_width=True):
                with st.spinner("Analyzing food image..."):
                    # Process image
                    image = Image.open(uploaded_file)
                    processed_image = preprocess_image_stable(image)
                    
                    # Make prediction
                    result = predict_food_stable(model, processed_image, model_type)
                    
                    # Display results
                    display_results(result)

if __name__ == "__main__":
    main()
