from useb import run
from sentence_transformers import SentenceTransformer  # SentenceTransformer is an awesome library for providing SOTA sentence embedding methods. TSDAE is also integrated into it.
import torch

sbert = SentenceTransformer('bert-base-nli-mean-tokens')  # Build an SBERT model

# The only thing needed for the evaluation: a function mapping a list of sentences into a batch of vectors (torch.Tensor)
@torch.no_grad()
def semb_fn(sentences) -> torch.Tensor:
    return torch.Tensor(sbert.encode(sentences, show_progress_bar=False))

results, results_main_metric = run(
    semb_fn_askubuntu=semb_fn, 
    semb_fn_cqadupstack=semb_fn,  
    semb_fn_twitterpara=semb_fn, 
    semb_fn_scidocs=semb_fn,
    eval_type='test',
    data_eval_path='data-eval'  # This should be the path to the folder of data-eval
)

assert round(results_main_metric['avg'], 1) == 47.6