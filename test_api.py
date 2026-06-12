import requests

# Sample real news
real_news = [
    "COVID-19 virus was first reported in Wuhan, China in December 2019.",
    "The United States elected Joe Biden as the 46th President in 2020.",
    "India launched its Mars mission Mangalyaan in 2013.",
    "The stock market crashed in March 2020 due to the COVID-19 pandemic.",
    "Scientists discovered water on Mars in 2015.",
    "The Paris Climate Agreement was signed in 2015 by 196 countries.",
    "Bitcoin reached an all-time high of over $60,000 in 2021.",
    "The WHO declared COVID-19 a pandemic on March 11, 2020.",
    "Tesla released the Model 3 electric car in 2017.",
    "The Nobel Peace Prize 2020 was awarded to the World Food Programme."
]

# Sample fake news
fake_news = [
    "Aliens landed in New York and declared war on humans.",
    "Bill Gates plans to microchip everyone through COVID vaccines.",
    "The moon landing was faked by NASA in 1969.",
    "Drinking bleach cures COVID-19 according to a secret government report.",
    "Obama was born in Kenya, not Hawaii.",
    "5G towers cause COVID-19 and mind control.",
    "The Earth is flat and NASA is hiding the truth.",
    "Vaccines contain microchips to track people.",
    "COVID-19 is a hoax created by pharmaceutical companies.",
    "Dinosaurs lived with humans 10,000 years ago."
]

def test_news(news_list, expected_label):
    url = "http://127.0.0.1:8000/predict"
    correct = 0
    total = len(news_list)
    for news in news_list:
        response = requests.post(url, json={"news": news})
        if response.status_code == 200:
            data = response.json()
            prediction = data.get("prediction")
            confidence = data.get("confidence")
            source = data.get("source")
            print(f"News: {news[:50]}...")
            print(f"Prediction: {prediction}, Confidence: {confidence}, Source: {source}")
            if prediction == expected_label:
                correct += 1
            print("---")
        else:
            print(f"Error: {response.status_code}")
    accuracy = correct / total * 100
    print(f"{expected_label} Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    return accuracy

if __name__ == "__main__":
    print("Testing Real News:")
    real_acc = test_news(real_news, "Real News")
    print("\nTesting Fake News:")
    fake_acc = test_news(fake_news, "Fake News")
    print(f"\nOverall: Real {real_acc:.1f}%, Fake {fake_acc:.1f}%")