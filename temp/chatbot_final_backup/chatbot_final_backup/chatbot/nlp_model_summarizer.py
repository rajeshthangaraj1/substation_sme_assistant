from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from llm_summarizer import summarize_document
from config.constant import MODEL_PATH,NLP_CHUNK_SIZE,NLP_CHUNK_OVERLAP,NLP_MODEL_MAX_LENGTH,NLP_CHUNK_OUTOUT_MIN_LENGTH,NLP_CHUNK_OUTOUT_MAX_LENGTH
from helper.common import load_document
from logger_config import setup_logger

logger = setup_logger()

def split_text(text, chunk_size=NLP_CHUNK_SIZE, chunk_overlap=NLP_CHUNK_OVERLAP):
    """Split text into chunks using RecursiveCharacterTextSplitter."""
    logger.info("Splitting text into chunks with chunk_size=%d, chunk_overlap=%d", chunk_size, chunk_overlap)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents([text])
    logger.info("Document successfully split into %d chunks.", len(docs))
    return docs



def summarize_document_nlp(file_path,status_container):
    """Summarize a Word document using LangChain"""
    try:
        logger.info("Starting Nlp document summarization")
        document_text = load_document(file_path)
        if not document_text:
            logger.error("No text extracted. Exiting...")
            exit()

        status_container.write("🟡 **NLP Model: Splitting document into chunks...**")
        # Split document into chunks
        document_chunks = split_text(document_text)
        status_container.write(f"✅ **Document split into {len(document_chunks)} chunks.**")
        # Load model and tokenizer

        logger.info("Loading tokenizer and model from: %s", MODEL_PATH)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

        # Initialize the summarization pipeline
        summarizer = pipeline('summarization', model=model, tokenizer=tokenizer)

        # Summarize each chunk
        summaries = []
        for i, chunk in enumerate(document_chunks):
            logger.info("Processing chunk %d of %d", i + 1, len(document_chunks))
            status_container.write(f"🔹 **Processing NLP Chunk {i + 1}/{len(document_chunks)}...**")

            # Tokenize the chunk with truncation
            inputs = tokenizer(chunk.page_content, return_tensors='pt', truncation=True, max_length=NLP_MODEL_MAX_LENGTH)

            # Generate summary
            summary_ids = model.generate(inputs['input_ids'], max_length=NLP_CHUNK_OUTOUT_MAX_LENGTH, min_length=NLP_CHUNK_OUTOUT_MIN_LENGTH, do_sample=False)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Debugging: Print each summary
            # print(f"\nSummary for Chunk {i + 1}:\n", summary, "\n")

            summaries.append(summary)

        # Combine all summaries into a single summary
        final_summary_nlp = ' '.join(summaries)
        status_container.write("✅ **NLP Model Summarization Completed!**")
        logger.info("Nlp summarization is done")
        # logging.info("NLP summary.", final_summary_nlp)
        return summarize_document(final_summary_nlp,status_container)

    except Exception as e:
        logger.error("Error in summarization: %s", str(e))
        raise e

