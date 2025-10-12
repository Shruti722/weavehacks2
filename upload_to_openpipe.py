from openpipe import OpenAI as OPClient
import os, json

client = OPClient(
    api_key=os.environ["OPENAI_API_KEY"],
    openpipe={"api_key": os.environ["OPENPIPE_API_KEY"]}
)

# Create dataset
resp = client.datasets.create(name="synergi-rlhf")
dataset_id = resp["id"]
print("Created dataset:", dataset_id)

# Upload entries (transform your JSONL traces)
def get_entries(jsonl_path):
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            x = json.loads(line)
            # transform to OpenPipe format: messages & split/train
            if "prompt" in x and "completion" in x and "output" in x and "reward" in x["output"]:
                entries.append({
                    "messages": [
                        {"role": "user", "content": x["prompt"]},
                        {"role": "assistant", "content": x["completion"]}
                    ],
                    "split": "train",
                    "metadata": {
                        "reward": x["output"]["reward"]
                    }
                })
    return entries

entries = get_entries("artifacts/weave-traces/traces.jsonl")
batch_size = 100
for i in range(0, len(entries), batch_size):
    chunk = entries[i : i + batch_size]
    resp = client.datasets.add_entries(id=dataset_id, entries=chunk)
    print("Uploaded batch", i, "->", resp.get("entries_added", 0))
