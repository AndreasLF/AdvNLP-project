# %%

from data.phishing import LoadPhishingDataset
from models.builers.retriever import Retriever
from data.dataloader import DataLoader
from data.phishing import PhishingDataset
from models.model_loader_helpers import createModels, loadModels
from utils.phishing_utils import getPhishingQueries
from models.DPR import DPR
from utils.metrics_uitls import timeFunction
from utils.phishing_utils import calculatePhishingAccuracy, evaluatePhishingByMajorityVote
from utils.misc import batch
import configparser
import torch
import os
import pickle
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# EXPERIMENT CONFIGURATION
config = configparser.ConfigParser()
config.read('configs/config.ini')
data_loader = DataLoader(config)

top_k = 25
test_split = 0.2
batch_size=25

model_descriptions = {
        "TF-IDF": {},
        "BM25": {},
        "DPR": {},
        "Crossencoder": {"n":2*top_k},
        "KMeans": {"k":3},
        "CURE": {"n": 25,
                "shrinkage_fraction" : 0.1,
                "threshold": 0.25,
                "initial_clusters": 50,
                "subsample_fraction": 0.5,
                "similarity_measure": "cosine"}}

load_saved_models = False

embedding_model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"#"bert-base-uncased"
embedding_index_folder_path = "indexes"
phishing_dataset_path = "data/datasets/phishing_dataset.pickle"
datasets_path = "data/datasets/Phishing_Email.csv"

# %%
# LOAD PHISHING DATASET
def preComputeEmbeddings(dataset: str, 
                         documents: list[dict], 
                         embedding_model_name: str, 
                         embedding_index_folder_path: str):
    embedder = DPR(documents, model_name=embedding_model_name)
    embedding_index_path = getPreComputedEmbeddingsPath(dataset, embedding_index_folder_path)
    embedder.SaveIndex(embedding_index_path)
    return embedding_index_path

def getPreComputedEmbeddingsPath(dataset: str, embedding_index_folder_path: str):
    return os.path.join(embedding_index_folder_path,dataset,"embedding_index.pickle")

# %%
if not os.path.exists(phishing_dataset_path):
    PhishingData = CreatePhishingDataset(datasets_path, save=True)
PhishingData = LoadPrecomputedPhishingDataset(phishing_dataset_path)


# %%
# RUN EXPERIMENT
def runPhishingExperiment(dataset: PhishingDataset, 
                  model_descriptions: dict[str, dict],
                  embedding_model_name: str,
                  embedding_index_folder_path: str,
                  top_k: int,
                  test_split: float):
    score_metrics: dict[str, dict[str, float]] = {}
    queries = getPhishingQueries(dataset)
    queries = queries[:int(len(queries)*test_split)]
    documents = dataset.GetDocumentDicts()
    documents = documents[int(len(queries)*test_split):]
    if load_saved_models:
        models = loadModels("phishing", model_descriptions)
    else:
        embedding_index_path = preComputeEmbeddings(
                            "phishing", 
                            documents,
                            embedding_model_name,
                            embedding_index_folder_path)
        models: dict[str, Retriever] = createModels(documents=documents, 
                                dataset_name="phishing", 
                                models=model_descriptions, 
                                embedding_index_path=embedding_index_path,
                                save=True)
    
    for model_name, model in models.items():
        retrieved_documents = []
        preds = []
        labels = []
        score_metrics[model_name] = {}
        total_time = 0
        print(f'Computing phishing results for {model_name}')
        iter_count = 0
        for query_batch in batch(queries, batch_size):
            time, retrieved_docs = timeFunction(model.Lookup, 
                                                **{"queries": [query.getQuery() for query in query_batch], 
                                                "k": top_k})
            retrieved_documents.extend(retrieved_docs)
            total_time += time
            iter_count += batch_size
            if iter_count % 250 == 0:
                print(f'Iter {iter_count}/{len(queries)}')
        
        retrieved_labels = [[dataset.GetLabelFromId(document.GetId()) for document in query] for query in retrieved_documents]
        preds = evaluatePhishingByMajorityVote(retrieved_labels)
        labels = [query.getLabel() for query in queries]
        
        score_metrics[model_name]["accuracy"] = calculatePhishingAccuracy(preds, labels)
        score_metrics[model_name]["time"] = total_time/len(queries)
    return score_metrics


# %%
score_metrics = runPhishingExperiment(PhishingData, 
                  model_descriptions,
                  embedding_model_name,
                  embedding_index_folder_path,
                  top_k,
                  test_split)
print(score_metrics)