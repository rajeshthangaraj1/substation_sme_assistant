# from rouge_score import rouge_scorer
# import sacrebleu
from bert_score import score
from helper.common import load_document
import os
import logging
from config.constant import LOG_FILE
from langchain.text_splitter import NLTKTextSplitter

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
def evaluate_summary(file_path, summary):

    document_text = load_document(file_path)
    if not document_text:
        logging.error("No text extracted. Exiting...")
        exit()
    """Evaluate summary quality using ROUGE, BLEU, and BERTScore."""

    # # ROUGE Score
    # rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # rouge_scores = rouge.score(document_text, summary)
    #
    # # BLEU Score
    # bleu_score = sacrebleu.corpus_bleu([summary], [[document_text]]).score

    # BERTScore
    # _, _, bert_score = score([summary], [document_text], lang="en", rescale_with_baseline=True)

    text_splitter = NLTKTextSplitter()
    # Compute BERTScore **chunk-wise**
    document_chunks = text_splitter.split_text(document_text)
    summary_chunks = text_splitter.split_text(summary)

    all_scores = []
    for doc_chunk, sum_chunk in zip(document_chunks, summary_chunks):
        _, _, bert_score = score([sum_chunk], [doc_chunk], lang="en", rescale_with_baseline=True)
        all_scores.append(bert_score.mean().item())

    # Average BERTScore across chunks
    avg_bert_score = sum(all_scores) / len(all_scores)

    return {
        # "ROUGE-1": rouge_scores['rouge1'].fmeasure,
        # "ROUGE-2": rouge_scores['rouge2'].fmeasure,
        # "ROUGE-L": rouge_scores['rougeL'].fmeasure,
        # "BLEU": bleu_score,
        # "BERTScore": bert_score.mean().item()
        "BERTScore": avg_bert_score  # Improved evaluation
    }
