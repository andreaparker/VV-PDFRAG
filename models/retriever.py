# models/retriever.py

import base64
import os
from diskcache import Cache
from PIL import Image
from io import BytesIO
from logger import get_logger
import time
import hashlib
import pickle
import numpy as np

logger = get_logger(__name__)

def retrieve_documents(RAG, query, session_id, k=3):
    """
    Retrieves relevant documents based on the user query using Byaldi.

    Args:
        RAG (RAGMultiModalModel): The RAG model with the indexed documents.
        query (str): The user's query.
        session_id (str): The session ID to store images in per-session folder.
        k (int): The number of documents to retrieve.

    Returns:
        list: A list of image filenames corresponding to the retrieved documents.
    """
    try:
        logger.info(f"Retrieving documents for query: {query}")
        
        if hasattr(RAG, 'use_disk_storage') and RAG.use_disk_storage:
            results = retrieve_from_disk(RAG, query, k)
        else:
            results = RAG.search(query, k=k)
        
        images = process_results(results, RAG, session_id)
        
        logger.info(f"Total {len(images)} documents retrieved. Image paths: {images}")
        return images

    except AttributeError as e:
        logger.error(f"AttributeError in retrieve_documents: {e}")
        logger.info("Falling back to default search method")
        results = RAG.search(query, k=k)
        images = process_results(results, RAG, session_id)
        return images

    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []

def process_results(results, RAG, session_id):
    """
    Process the search results and save images.

    Args:
        results (list): The search results.
        RAG (RAGMultiModalModel): The RAG model.
        session_id (str): The session ID.

    Returns:
        list: A list of image filenames.
    """
    images = []
    session_images_folder = os.path.join('static', 'images', session_id)
    os.makedirs(session_images_folder, exist_ok=True)

    for result in results:
        if hasattr(RAG, 'use_disk_storage') and RAG.use_disk_storage:
            image_data = result['image']
        elif hasattr(result, 'base64') and result.base64:
            image_data = base64.b64decode(result.base64)
        else:
            logger.warning(f"No image data for document {getattr(result, 'doc_id', 'unknown')}, page {getattr(result, 'page_num', 'unknown')}")
            continue

        image = Image.open(BytesIO(image_data))
        
        # Generate a unique filename based on the image content
        image_hash = hashlib.md5(image_data).hexdigest()
        image_filename = f"retrieved_{image_hash}.png"
        image_path = os.path.join(session_images_folder, image_filename)
        
        if not os.path.exists(image_path):
            image.save(image_path, format='PNG')
            logger.debug(f"Retrieved and saved image: {image_path}")
        else:
            logger.debug(f"Image already exists: {image_path}")
        
        # Store the relative path from the static folder
        relative_path = os.path.join('images', session_id, image_filename)
        images.append(relative_path)
        logger.info(f"Added image to list: {relative_path}")

    return images

def retrieve_from_disk(RAG, query, k):
    """
    Retrieves relevant documents from disk-based storage.

    Args:
        RAG (RAGMultiModalModel): The RAG model with disk storage information.
        query (str): The user's query.
        k (int): The number of documents to retrieve.

    Returns:
        list: A list of dictionaries containing retrieved document information.
    """
    query_embedding = RAG.encode_query(query)
    results = []

    for key in RAG.disk_cache.cache.iterkeys():
        if key.endswith('_embedding'):
            embedding = pickle.loads(RAG.disk_cache.cache.get(key))
            similarity = compute_similarity(query_embedding, embedding)
            image_key = key.replace('_embedding', '_image')
            image = pickle.loads(RAG.disk_cache.cache.get(image_key))
            results.append({
                'similarity': similarity,
                'image': image,
                'key': key
            })

    # Sort results by similarity and return top k
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:k]

def compute_similarity(query_embedding, document_embedding):
    """
    Compute the cosine similarity between query and document embeddings.

    Args:
        query_embedding (np.array): The embedding of the query.
        document_embedding (np.array): The embedding of the document.

    Returns:
        float: The cosine similarity between the embeddings.
    """
    return np.dot(query_embedding, document_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(document_embedding)
    )


# # models/retriever.py

# import base64
# import os
# from diskcache import Cache
# from PIL import Image
# from io import BytesIO
# from logger import get_logger
# import time
# import hashlib
# import pickle
# import numpy as np


# logger = get_logger(__name__)

# def retrieve_documents(RAG, query, session_id, k=3):
#     """
#     Retrieves relevant documents based on the user query using Byaldi.

#     Args:
#         RAG (RAGMultiModalModel): The RAG model with the indexed documents.
#         query (str): The user's query.
#         session_id (str): The session ID to store images in per-session folder.
#         k (int): The number of documents to retrieve.

#     Returns:
#         list: A list of image filenames corresponding to the retrieved documents.
#     """
#     try:
#         logger.info(f"Retrieving documents for query: {query}")
#         # results = RAG.search(query, k=k)
#         # images = []
#         # session_images_folder = os.path.join('static', 'images', session_id)
#         # os.makedirs(session_images_folder, exist_ok=True)
#         if hasattr(RAG, 'use_disk_storage') and RAG.use_disk_storage:
#             results = retrieve_from_disk(RAG, query, k)
#         else:
#             results = RAG.search(query, k=k)
        
#         images = []
#         session_images_folder = os.path.join('static', 'images', session_id)
#         os.makedirs(session_images_folder, exist_ok=True)
  
#         for i, result in enumerate(results):
#             if RAG.use_disk_storage:
#                 image_data = result['image']
#             elif result.base64:
#                 image_data = base64.b64decode(result.base64)
#             else:
#                 logger.warning(f"No image data for document {result.doc_id}, page {result.page_num}")
#                 continue

#             image = Image.open(BytesIO(image_data))
            
#             # Generate a unique filename based on the image content

#             image_hash = hashlib.md5(image_data).hexdigest()
#             image_filename = f"retrieved_{image_hash}.png"
#             image_path = os.path.join(session_images_folder, image_filename)
            
#             if not os.path.exists(image_path):
#                 image.save(image_path, format='PNG')
#                 logger.debug(f"Retrieved and saved image: {image_path}")
#             else:
#                 logger.debug(f"Image already exists: {image_path}")
            
#             # Store the relative path from the static folder

#             relative_path = os.path.join('images', session_id, image_filename)
#             images.append(relative_path)
#             logger.info(f"Added image to list: {relative_path}")
        
#         logger.info(f"Total {len(images)} documents retrieved. Image paths: {images}")
#         return images
#     except Exception as e:
#         logger.error(f"Error retrieving documents: {e}")
#         return []

# def retrieve_from_disk(RAG, query, k):
#     """
#     Retrieves relevant documents from disk-based storage.

#     Args:
#         RAG (RAGMultiModalModel): The RAG model with disk storage information.
#         query (str): The user's query.
#         k (int): The number of documents to retrieve.

#     Returns:
#         list: A list of dictionaries containing retrieved document information.
#     """
#     query_embedding = RAG.encode_query(query)
#     results = []

#     for key in RAG.disk_cache.cache.iterkeys():
#         if key.endswith('_embedding'):
#             embedding = pickle.loads(RAG.disk_cache.cache.get(key))
#             similarity = compute_similarity(query_embedding, embedding)  # Implement this function
#             image_key = key.replace('_embedding', '_image')
#             image = pickle.loads(RAG.disk_cache.cache.get(image_key))
#             results.append({
#                 'similarity': similarity,
#                 'image': image,
#                 'key': key
#             })

#     # Sort results by similarity and return top k
#     results.sort(key=lambda x: x['similarity'], reverse=True)
#     return results[:k]

# def compute_similarity(query_embedding, document_embedding):
#     """
#     Compute the cosine similarity between query and document embeddings.

#     Args:
#         query_embedding (np.array): The embedding of the query.
#         document_embedding (np.array): The embedding of the document.

#     Returns:
#         float: The cosine similarity between the embeddings.
#     """
#     return np.dot(query_embedding, document_embedding) / (
#         np.linalg.norm(query_embedding) * np.linalg.norm(document_embedding)
#     )

        
# ### original code below before you implemented disk caching functionality
#     #     for i, result in enumerate(results):
#     #         if result.base64:
#     #             image_data = base64.b64decode(result.base64)
#     #             image = Image.open(BytesIO(image_data))
                
#     #             # Generate a unique filename based on the image content
#     #             image_hash = hashlib.md5(image_data).hexdigest()
#     #             image_filename = f"retrieved_{image_hash}.png"
#     #             image_path = os.path.join(session_images_folder, image_filename)
                
#     #             if not os.path.exists(image_path):
#     #                 image.save(image_path, format='PNG')
#     #                 logger.debug(f"Retrieved and saved image: {image_path}")
#     #             else:
#     #                 logger.debug(f"Image already exists: {image_path}")
                
#     #             # Store the relative path from the static folder
#     #             relative_path = os.path.join('images', session_id, image_filename)
#     #             images.append(relative_path)
#     #             logger.info(f"Added image to list: {relative_path}")
#     #         else:
#     #             logger.warning(f"No base64 data for document {result.doc_id}, page {result.page_num}")
        
#     #     logger.info(f"Total {len(images)} documents retrieved. Image paths: {images}")
#     #     return images
#     # except Exception as e:
#     #     logger.error(f"Error retrieving documents: {e}")
#     #     return []