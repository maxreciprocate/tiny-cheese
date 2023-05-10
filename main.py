import os
import json
import rich
import random
import threading
import numpy as np
from time import time
from datasets import load_dataset

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

lock = threading.Lock()
current_ix = 0
matches = []

@api.on_event("startup")
@api.get("/reset")
def reset():
    dataset = os.environ.get("dataset", "Dahoas/rm-static")
    split = os.environ.get("split", "train")
    num_matches = int(os.environ.get("num_matches", "10"))
    columns = os.environ.get("columns", "chosen,rejected").split(",")

    global current_ix, matches
    current_ix = 0
    matches = []
    dataset = load_dataset(dataset, split=split)

    for ix in range(num_matches):
        sample = dataset[ix]
        outputs = [sample[columns[0]], sample[columns[1]]]
        random.shuffle(outputs)
        matches.append({
            "prompt": sample["prompt"] if "prompt" in sample else "",
            "outputs": outputs,
            "selected": sample[columns[0]],
            "metadata": {"index": ix},
        })

@api.get("/get_match")
def get_match():
    global current_ix
    if current_ix >= len(matches):
        return {}

    with lock:
        match = matches[current_ix]
        current_ix += 1

    match = {k: v for k, v in match.items() if k != "selected"}
    match["metadata"]["start_time"] = time()
    return match

@api.post("/submit_result")
async def submit_result(request: Request):
    result = await request.json()
    match = matches[result["metadata"]["index"]]

    match["metadata"]["duration"] = time() - match["metadata"]["start_time"]
    match["winner"] = result["winner"]

    rich.print(match)

    with open("matches.json", "w") as f:
        json.dump(matches, f)

@api.get("/get_stats")
def get_stats():
    global matches
    correct = 0
    n_reviewed = 0
    for match in matches:
        if "winner" not in match or match["winner"] == "tie":
            continue

        correct += match["outputs"][match["winner"]] == match["selected"]
        n_reviewed += 1

    accuracy = correct / (n_reviewed + 1e-20)
    return {"accuracy": accuracy, "n_reviewed": n_reviewed}

api.mount("/", StaticFiles(directory="static", html=True), name="static")

