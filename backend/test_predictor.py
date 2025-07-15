from predictor import predict_from_image

# Replace with your test image path
image_path = "test_images/chart1.jpg"


result = predict_from_image(image_path)
print("Prediction result:")
print(result)
 