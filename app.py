import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import imghdr

# Load the pre-trained model
try:
    model = load_model('model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

def predict(image):
    try:
        img = Image.open(image).resize((50, 50))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None
# def predict(image):
#     img = Image.open(image).resize((25, 25))
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = np.transpose(img_array, (0, 3, 1, 2))  # Match the expected input shape
#     prediction = model.predict(img_array)
#     return prediction

def app():
    # Set up the sidebar
    st.set_page_config(layout="wide")
    st.sidebar.title("Navigation")


    def is_valid_image(file):
        try:
            image = Image.open(file)
            image.verify()
            return True
        except (IOError, SyntaxError) as e:
            print(f"Error: {e}")
            return False


    # Create a folder object to hold images
    folder = st.sidebar.file_uploader("Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    # Create navigation options
    nav_option = st.sidebar.radio("Select an option", ["About", "Predict"])

    col1, col2 = st.columns(2)

    # About page
    if nav_option == "About":
        with col1:
            st.title("About")
            st.write("This is a breast cancer prediction app. You can upload an image, and the app will classify the risk of cancer based on the image.")

    # Predict page
    elif nav_option == "Predict":
        with col1:
            st.title("Cancer Prediction")
            st.write("Upload an image to predict cancer risk.")

            # Initialize session state for prediction history
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []

            # Check if images are uploaded
            if folder:
                for image in folder:
                    img = Image.open(image)
                    st.image(img, caption=image.name, use_column_width=True)

                    if st.button(f"Predict for {image.name}"):
                        prediction = predict(image)
                        if prediction[0][0] > prediction[0][1]:
                            st.write(f"The image {image.name} is classified as low risk for cancer.")
                        else:
                            st.write(f"The image {image.name} is classified as high risk for cancer.")

                        # Save prediction to history
                        st.session_state.prediction_history.append((image.name, prediction[0][0] > prediction[0][1]))

                        # Refresh the app to clear the current state
                        st.experimental_rerun()
            else: ##only predict when there's an image
                st.write("Please upload some images to test the system.")

        with col2:
            # Display prediction history
            st.subheader("Prediction History")
            for name, result in st.session_state.prediction_history:
                st.write(f"{name}: {'Low Risk' if result else 'High Risk'}")

if __name__ == "__main__":
    app()