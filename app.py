import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from ultralytics import YOLO

# Page configuration
st.set_page_config(
    page_title="Smart Road Damage Predictor",
    page_icon="🚧",
    layout="centered",
)

st.title("🚧 Smart Road Damage & Pothole Severity Predictor")
st.write(
    "Upload a road image to detect potholes, cracks, and manholes, and"
    " analyze their severity."
)


# Load YOLO model (Cached)
@st.cache_resource
def load_model():
  return YOLO("best.pt")


model = load_model()

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose a road image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert("RGB")
  w, h = image.size

  st.image(image, caption="Uploaded Road Image", use_column_width=True)

  if st.button("Run Damage & Severity Analysis"):
    with st.spinner("Analyzing road conditions..."):
      # Run inference
      results = model(image)[0]
      draw = ImageDraw.Draw(image)

      detection_count = 0

      for box in results.boxes:
        detection_count += 1
        coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        x1, y1, x2, y2 = coords
        box_area = (x2 - x1) * (y2 - y1)
        img_area = w * h
        ratio = (box_area / img_area) * 100

        # Assign colors and labels
        if class_name == "manhole":
          color = (0, 255, 255)  # Cyan
          label_text = f"{class_name} {conf:.2f}"
        else:
          if ratio < 5:
            severity = "Low"
            color = (0, 255, 0)  # Green
          elif ratio < 20:
            severity = "Medium"
            color = (255, 165, 0)  # Orange
          else:
            severity = "High"
            color = (255, 0, 0)  # Red

          label_text = f"{class_name} ({severity}) {conf:.2f}"

        # Draw bounding box (using Pillow instead of OpenCV)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1, max(y1 - 15, 0)), label_text, fill=color)

      # Show final annotated image
      st.image(
          image,
          caption=f"Analysis Result ({detection_count} objects detected)",
          use_column_width=True,
      )
      st.success("Analysis complete!")
        )

      st.image(
          annotated_img, caption="Processed Detection Results", use_column_width=True
      )
      st.success("Analysis complete!")
