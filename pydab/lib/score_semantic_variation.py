import numpy as np

class ScoreSemanticVariation:
    def __init__(self, K, data_normal, data_anomalies):
        self.K = K
        self.data_normal = data_normal
        self.data_anomalies = data_anomalies
        self.scores = []
    
    def run(self):
        counter = 1
        for a in self.data_anomalies:
            nn_anomalies = []

            print("Processing %d of %d...\r" % (counter, len(self.data_anomalies)))
            counter = counter + 1
            
            for dp in self.data_anomalies:
                s = np.sqrt(np.sum((a - dp) ** 2))

                if s > 0:
                    nn_anomalies.append({
                        "score": s,
                        "vector": dp
                    })
            
            nn_normal = []
            
            for dp in self.data_normal:
                s = np.sqrt(np.sum((a - dp) ** 2))
                
                if s > 0:
                    nn_normal.append({
                        "score": s,
                        "vector": dp
                    })
            
            temp_score = {
                "anomaly_vector": a,
                "nn_anomalies": sorted(nn_anomalies, key = lambda i: i['score'])[:self.K],
                "nn_normal": sorted(nn_normal, key = lambda i: i['score'])[:self.K]
            }
            
            var_normals = np.var([d['score'] for d in temp_score['nn_anomalies']])
            var_anomalies = np.var([d['score'] for d in temp_score['nn_normal']])
            
            svs = var_normals / var_anomalies
            
            temp_score["sv_score"] = svs
            
            if svs >= 0 and svs < 0.25:
                temp_score["sv_classification"] = "EASY"
            if svs >= 0.25 and svs < 0.5:
                temp_score["sv_classification"] = "MEDIUM"
            if svs >= 0.5 and svs < 1:
                temp_score["sv_classification"] = "HARD"
            elif svs >= 1:
                temp_score["sv_classification"] = "VERY_HARD"
            
            self.scores.append(temp_score)
