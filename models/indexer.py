# models/indexer.py

import os
from diskcache import Cache
from byaldi import RAGMultiModalModel
from models.converters import convert_docs_to_pdfs
from logger import get_logger
import pickle


logger = get_logger(__name__)

class DiskCacheIndexer:
    def __init__(self, cache_dir='./cache'):
        self.cache = Cache(cache_dir)

    def store_image(self, key, image):
        self.cache.set(f"{key}_image", pickle.dumps(image))

    def store_embedding(self, key, embedding):
        self.cache.set(f"{key}_embedding", pickle.dumps(embedding))

    def get_image(self, key):
        return pickle.loads(self.cache.get(f"{key}_image"))

    def get_embedding(self, key):
        return pickle.loads(self.cache.get(f"{key}_embedding"))


def index_documents(folder_path, index_name='document_index', index_path=None, indexer_model='vidore/colpali'):
    """
    Indexes documents in the specified folder using Byaldi.

    Args:
        folder_path (str): The path to the folder containing documents to index.
        index_name (str): The name of the index to create or update.
        index_path (str): The path where the index should be saved.
        indexer_model (str): The name of the indexer model to use.

    Returns:
        RAGMultiModalModel: The RAG model with the indexed documents.
    """
    try:
        logger.info(f"Starting document indexing in folder: {folder_path}")
        # Convert non-PDF documents to PDFs
        convert_docs_to_pdfs(folder_path)
        logger.info("Conversion of non-PDF documents to PDFs completed.")

       
        # Initialize RAG model
        RAG = RAGMultiModalModel.from_pretrained(indexer_model)
        RAG.use_disk_storage = True
        RAG.disk_cache = disk_cache

         # Switch model to half precision to save GPU memory; not needed on all devices
        # RAG.half()
        
        if RAG is None:
            raise ValueError(f"Failed to initialize RAGMultiModalModel with model {indexer_model}")
        logger.info(f"RAG model initialized with {indexer_model}.")

        # Initialize disk cache
        disk_cache = DiskCacheIndexer(cache_dir=index_path)

        # Index the documents in the folder
        for file in os.listdir(folder_path):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(folder_path, file)
                images, embeddings = RAG.index_document(pdf_path)
                
                for i, (image, embedding) in enumerate(zip(images, embeddings)):
                    key = f"{file}_{i}"
                    disk_cache.store_image(key, image)
                    disk_cache.store_embedding(key, embedding)

        logger.info(f"Indexing completed. Index saved at '{index_path}'.")

        # Modify RAG model to use disk-based storage
        RAG.use_disk_storage = True
        RAG.disk_cache = disk_cache

        return RAG
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise


    #     # Index the documents in the folder
    #     RAG.index(
    #         input_path=folder_path,
    #         index_name=index_name,
    #         store_collection_with_index=True,
    #         overwrite=True
    #     )

    #     logger.info(f"Indexing completed. Index saved at '{index_path}'.")

    #     return RAG
    # except Exception as e:
    #     logger.error(f"Error during indexing: {str(e)}")
    #     raise
