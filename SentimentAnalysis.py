import os
from scipy.special import softmax
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

def analyze_sentiment(data):

    if not os.path.isdir("./metrics"):
        os.mkdir("./metrics")

    if not os.path.isdir(f"./metrics/sentiment"):
        os.mkdir(f"./metrics/sentiment")

    pretrained_model = "cardiffnlp/twitter-roberta-base-sentiment"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)

    output = str(model)

    model_file = open(f"./metrics/sentiment/model-info.txt", "w")

    model_file.write(output)

    model_file.close()

    differences = []

    file = open(f"./metrics/sentiment/sentiment-output.txt", "w")
    for index, row in data.iterrows():

        reviewer_score = (data['xOTIOverall'][index] / 6) * 100

        mission = data["Mission Statement"][index]
        mission = mission[:512]
        encoded_mission = tokenizer(mission, return_tensors="pt")
        output = model(**encoded_mission)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        total = (float(scores[2]) + float(scores[1])) * 100

        difference = abs(total - reviewer_score)

        differences.append(difference)
        file.write(data["CompanyName"][index])
        file.write("\n")
        file.write(f"Reviewer positivity score: {reviewer_score}")
        file.write("\n")
        file.write(f"Non-negative: {total}, Negative: {scores[0] * 100}")
        file.write("\n")
        file.write(f"Difference: {difference}")
        file.write("\n")
        file.write("\n")
        file.write("\n")

    file.write(f"Average difference: {sum(differences) / len(differences)}")

    file.close()
