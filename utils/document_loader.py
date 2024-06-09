# document_loader.py
import os
from typing import List
from langchain_community.document_loaders import FireCrawlLoader
from document import Document

class DocumentLoader:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_docs(self, url: str) -> List[Document]:
        """
        Retrieves documents from the specified URL using the FireCrawlLoader.

        Args:
            url (str): The URL to crawl for documents.

        Returns:
            List[Document]: A list of Document objects containing the retrieved content.
        """
        loader = FireCrawlLoader(
            api_key=self.api_key, url=url, mode="crawl"
        )

        raw_docs = loader.load()
        docs = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in raw_docs]

        return docs