import sys
import argparse
import os
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
import pandas as pd
import random

from sklearn.linear_model import LogisticRegression
from lib.score_semantic_variation import ScoreSemanticVariation
from lib.score_feature_relevance import ScoreFeatureRelevance

parser = argparse.ArgumentParser(description="pydab: Dataset Anomaly Benchmark generator")
parser.add_argument("--input", help="Input csv file. Headers should have x1,x2,x3...y where y is either 0 for anomalies and 1 for normal", required=True)
parser.add_argument("--svk", help="Semantic variation value. Integer", required=True, type=int)
parser.add_argument("--output", help="Output csv file containing the categorical measures", required=True)
parser.add_argument("--data_out", help="Output csv file for the generated data set")

args = parser.parse_args()

if __name__ == '__main__':
    input_file  = args.input
    svk         = args.svk
    output_file = args.output
    data_out    = args.data_out

    data = {
        "anomalies": []
    }

    input_data      = pd.read_csv(input_file)
    input_normal    = input_data[input_data['y'] == 1]
    input_anomalies = input_data[input_data['y'] == 0]

    x_normal    = input_normal.drop(['y'], axis=1).values
    x_anomalies = input_anomalies.drop(['y'], axis=1).values

    y_normal    = np.ones(len(input_normal))
    y_anomalies = np.zeros(len(input_anomalies))

    X = np.concatenate((x_normal, x_anomalies))
    Y = np.concatenate((y_normal, y_anomalies), axis=None)

    dimensionality = len(x_normal[0])

    # Load initial anomalies
    for i in range(len(x_anomalies)):
        data["anomalies"].append({
            "x": x_anomalies[i],
            "pd": 0.0,
            "pd_classification": "",
            "sv": 0.0,
            "sv_classification": "",
            "fr": 0.0,
            "fr_classification": ""
        })

    # Compute Point Difficulty
    print("Computing point difficulty...")

    clf = LogisticRegression(random_state=0).fit(X, Y)
    probabilities = clf.predict_proba(x_anomalies)

    for i in range(len(probabilities)):
        p = probabilities[i][1]

        data["anomalies"][i]["pd"] = p

        if p >= 0 and p < 0.16:
            data["anomalies"][i]["pd_classification"] = "EASY"
        elif p >= 0.16 and p < 0.3:
            data["anomalies"][i]["pd_classification"] = "MEDIUM"
        elif p >= 0.3 and p < 0.5:
            data["anomalies"][i]["pd_classification"] = "HARD"
        elif p >= 0.5:
            data["anomalies"][i]["pd_classification"] = "VERY_HARD"

    print("Done computing point difficulty...")

    # Compute Semantic Variation
    print("Computing semantic variation...")

    sv_scorer = ScoreSemanticVariation(svk, x_normal, x_anomalies)
    sv_scorer.run()

    for i in range(len(x_anomalies)):
        data["anomalies"][i]["sv_classification"] = sv_scorer.scores[i]["sv_classification"]
        data["anomalies"][i]["sv"] = sv_scorer.scores[i]["sv_score"]

    print("Done computing semantic variation...")

    # Compute Feature Relevance
    print("Computing feature relevance...")

    fr_scorer   = ScoreFeatureRelevance(x_normal, x_anomalies)
    fr_scorer.run()

    for i in range(len(x_anomalies)):
        data["anomalies"][i]["fr_classification"]   = fr_scorer.scores[i]["fr_classification"]
        data["anomalies"][i]["fr"]                  = fr_scorer.scores[i]["fr_score"]

    print("Done computing feature relevance...")

    print("Computing final categorical measure...")
    for i in range(len(x_anomalies)):
        temp = [data["anomalies"][i]["pd_classification"], data["anomalies"][i]["sv_classification"], data["anomalies"][i]["fr_classification"]]
        final_classification = max(set(temp), key=temp.count)

        data["anomalies"][i]["final_classification"] = final_classification

    print("Point Difficulty Statistics:")
    print("=====================================")
    print("EASY:", len([d for d in data["anomalies"] if d["pd_classification"] == "EASY"]))
    print("MEDIUM:", len([d for d in data["anomalies"] if d["pd_classification"] == "MEDIUM"]))
    print("HARD:", len([d for d in data["anomalies"] if d["pd_classification"] == "HARD"]))
    print("VERY_HARD:", len([d for d in data["anomalies"] if d["pd_classification"] == "VERY_HARD"]))

    print("Semantic Variation Statistics:")
    print("=====================================")
    print("EASY:", len([d for d in data["anomalies"] if d["sv_classification"] == "EASY"]))
    print("MEDIUM:", len([d for d in data["anomalies"] if d["sv_classification"] == "MEDIUM"]))
    print("HARD:", len([d for d in data["anomalies"] if d["sv_classification"] == "HARD"]))
    print("VERY_HARD:", len([d for d in data["anomalies"] if d["sv_classification"] == "VERY_HARD"]))

    print("Feature Relevance Statistics:")
    print("=====================================")
    print("EASY:", len([d for d in data["anomalies"] if d["fr_classification"] == "EASY"]))
    print("MEDIUM:", len([d for d in data["anomalies"] if d["fr_classification"] == "MEDIUM"]))
    print("HARD:", len([d for d in data["anomalies"] if d["fr_classification"] == "HARD"]))
    print("VERY_HARD:", len([d for d in data["anomalies"] if d["fr_classification"] == "VERY_HARD"]))

    print("Final Categorical Measures Statistics:")
    print("=====================================")
    print("EASY:", len([d for d in data["anomalies"] if d["final_classification"] == "EASY"]))
    print("MEDIUM:", len([d for d in data["anomalies"] if d["final_classification"] == "MEDIUM"]))
    print("HARD:", len([d for d in data["anomalies"] if d["final_classification"] == "HARD"]))
    print("VERY_HARD:", len([d for d in data["anomalies"] if d["final_classification"] == "VERY_HARD"]))

    # Create output_file
    print("Writing to file %s..." % (output_file))
    with open(output_file, 'w', newline='') as file:
        writer  = csv.writer(file)

        # Write the header
        row = []
        for l in range(dimensionality):
            row.append("x" + str(l))

        row.append("y")
        row.append("pd")
        row.append("sv")
        row.append("fr")
        row.append("pd_classification")
        row.append("sv_classification")
        row.append("fr_classification")
        row.append("final_classification")

        writer.writerow(row)

        for d in data["anomalies"]:
            row = []

            for f in d["x"]:
                row.append(f)

            row.append(0)
            row.append(d["pd"])
            row.append(d["sv"])
            row.append(d["fr"])
            row.append(d["pd_classification"])
            row.append(d["sv_classification"])
            row.append(d["fr_classification"])
            row.append(d["final_classification"])

            writer.writerow(row)

        for d in x_normal:
            row = []

            for f in d:
                row.append(f)

            row.append(1)
            row.append("N")
            row.append("N")
            row.append("N")
            row.append("N")
            row.append("N")
            row.append("N")
            row.append("N")

            writer.writerow(row)

    # If user put in derive, generate a dataset
    if(data_out):
        for derive in ["EASY", "MEDIUM", "HARD", "VERY_HARD"]:
            counter = 0
            output_data_file = data_out + "_" + derive + ".csv"

            print("Writing generated %s data to file %s..." % (derive, output_data_file))
            with open(output_data_file, 'w', newline='') as file:
                writer  = csv.writer(file)

                # Write the header
                row = []
                for l in range(dimensionality):
                    row.append("x" + str(l))

                row.append("y")

                writer.writerow(row)

                # Anomalies
                for d in data["anomalies"]:
                    if d["final_classification"] == derive:
                        counter += 1
                        row = []

                        for f in d["x"]:
                            row.append(f)

                        row.append(0)

                        writer.writerow(row)

                # Normal
                for d in x_normal:
                    row = []

                    for f in d:
                        row.append(f)

                    row.append(1)

                    writer.writerow(row)

            print("%d of %d anomalies generated..." % (counter, len(x_anomalies)))
            print("Ratio of anomalies in generated dataset: %d : %d (%0.4f%%)" % (counter, len(X), (counter / len(X)) * 100))
    print("Done.")
