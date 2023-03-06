# https://github.com/Muennighoff/sgpt#use-sgpt-with-sentence-transformers
import torch
from transformers import AutoModel, AutoTokenizer

from common import *

tokenizer = AutoTokenizer.from_pretrained("Muennighoff/SGPT-125M-weightedmean-nli-bitfit")
model = AutoModel.from_pretrained("Muennighoff/SGPT-125M-weightedmean-nli-bitfit")

def model_embeddings(tokenizer, model, texts):
    # Deactivate Dropout (There is no dropout in the above models so it makes no difference here but other SGPT models may have dropout)
    # model.eval()
    batch_tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        # Get hidden state of shape [bs, seq_len, hid_dim]
        last_hidden_state = model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state
    weights = (
        torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float().to(last_hidden_state.device)
    )
    input_mask_expanded = (
        batch_tokens["attention_mask"]
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float()
    )
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
    sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

    embeddings = sum_embeddings / sum_mask
    return embeddings

def save_embeddings():
    corpus, files = load_corpus(minlen=MIN_LENGTH, skipwords=JAVA_KEYWORDS, tokens=False)
    fileids = load_fileids()
    n = len(files)

    fname = f"cocome-sgpt.vec"
    with open(fname, mode='w') as wrt:
        for i in range(n):
            file = files[i]
            if file not in fileids:
                print(f"file skipped: '{file}'")
                continue

            print(f"[{i:4}] processing {file}")
            embeddings = model_embeddings(tokenizer, model, [corpus[i]]).numpy()
            dvect = list(embeddings[0])

            svect = list(map(str, dvect))
            semb = ",".join(svect)

            fileid = fileids[file]
            wrt.write(f"{fileid},{semb}\n")
        pass

    pass


def main():
    save_embeddings()
# end


if __name__ == '__main__':
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
