from document import Document
from utils.generate_chain import create_generate_chain

class GraphNodes:
    def __init__(self, llm, retriever, retrieval_grader, hallucination_grader, code_evaluator, question_rewriter):
        self.llm = llm
        self.retriever = retriever
        self.retrieval_grader = retrieval_grader
        self.hallucination_grader = hallucination_grader
        self.code_evaluator = code_evaluator
        self.question_rewriter = question_rewriter
        self.generate_chain = create_generate_chain(llm)

    def retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["input"]

        # Retrieval
        documents = self.retriever.invoke(question)
        return {"documents": documents, "input": question}

    def generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["input"]
        documents = state["documents"]

        # RAG generation
        generation = self.generate_chain.invoke({"context": documents, "input": question})
        return {"documents": documents, "input": question, "generation": generation}

    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["input"]
        documents = state["documents"]

        # score each doc
        filtered_docs = []

        for d in documents:
            score = self.retrieval_grader.invoke({"input": question, "document": d.page_content})
            grade = score["score"]
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT IR-RELEVANT---")
                continue

        return {"documents": filtered_docs, "input": question}

    def transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        print("---TRANSFORM QUERY---")
        question = state["input"]
        documents = state["documents"]

        # Re-write question
        better_question = self.question_rewriter.invoke({"input": question})
        return {"documents": documents, "input": better_question}