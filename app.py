import cv2
from PIL import Image
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Page config
st.set_page_config(page_title="Road Damage & Severity Predictor", layout="centered")

st.title("🚧 Smart Road Damage & Pothole Severity Predictor")
st.write("Upload a road image to detect potholes, cracks, and manholes.")


# Load trained YOLO model (looks for best.pt in the same folder)
@st.cache_resource
def load_model():
  return YOLO("best.pt")


model = load_model()

uploaded_file = st.file_uploader(
    "Choose a road image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  img_array = np.array(image)
  h, w, _ = img_array.shape

  st.image(image, caption="Uploaded Image", use_column_width=True)

  if st.button("Detect and Analyze"):
    with st.spinner("Analyzing road condition..."):
      results = model(img_array)[0]
      annotated_img = img_array.copy()

      for box in results.boxes:
        coords = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        x1, y1, x2, y2 = coords
        box_area = (x2 - x1) * (y2 - y1)
        img_area = w * h
        ratio = (box_area / img_area) * 100

        # Handle manholes differently from structural damage
        if class_name == "manhole":
          color = (255, 255, 0)  # Cyan for infrastructure/manholes
          label_text = f"{class_name} {conf:.2f}"
        else:
          # Severity calculation based on area ratio for potholes and cracks
          if ratio < 5:
            severity = "Low"
            color = (0, 255, 0)  # Green
          elif ratio < 20:
            severity = "Medium"
            color = (0, 165, 255)  # Orange
          else:
            severity = "High"
            color = (255, 0, 0)  # Red

          label_text = f"{class_name} ({severity}) {conf:.2f}"

        # Draw bounding boxes and labels
        cv2.rectangle(
            annotated_img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            3,
        )
        cv2.putText(
            annotated_img,
            label_text,
            (int(x1), max(int(y1) - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

      st.image(
          annotated_img, caption="Processed Detection Results", use_column_width=True
      )
      st.success("Analysis complete!")