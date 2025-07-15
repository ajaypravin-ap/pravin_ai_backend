from preprocess import extract_numbers_and_features

results = extract_numbers_and_features("../test_images/bht_chart.jpg")

for item in results:
    print(f"Number: {item['number']} → {item['size']}, {item['color']}")