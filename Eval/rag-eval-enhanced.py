import os
import numpy as np
import json
import pandas as pd
import cohere
from tqdm import tqdm
from typing import List, Dict, Tuple, Any, Set
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import the hybrid search function from your existing files
# Try/except to handle cases where the imports might fail
try:
    from chatbotV2 import hybrid_search, db, reranker_model, generate_llama_answer
except ImportError:
    print("Warning: Could not import all functions from chatbotV2.py. Some functionality may be limited.")
    try:
        from chatbotV2 import db, reranker_model
        print("Successfully imported db and reranker_model")
    except ImportError:
        print("Warning: Could not import db and reranker_model. Will initialize new instances.")


class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-v3.5"):
        self.client = cohere.Client(api_key)
        self.model = model

    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Returns relevance scores from Cohere's rerank API for query-document pairs."""
        scores = []
        for query, doc in pairs:
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=[doc]
            )
            scores.append(response.results[0].relevance_score)
        return scores

class RAGEvaluator:
    """
    A class to evaluate RAG systems using retrieval and generation metrics:
    
    Retrieval Metrics:
    - Precision@k
    - Recall@k
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain (NDCG)
    
    Generation Metrics:
    - Semantic Similarity (between generated answer and reference answer)
    - Faithfulness (factual consistency with retrieved documents)
    - Answer Relevance (relevance of the generated answer to the query)
    """
    
    def __init__(
        self, 
        faiss_index_path: str = "Backend/faiss_index_luddy",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        faithfulness_model_name: str = "cross-encoder/qnli-distilroberta-base"
    ):
        # Initialize embedding model
        try:
            self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
            print(f"✅ Initialized embedding model: {embedding_model_name}")
        except:
            print(f"❌ Failed to initialize embedding model. Some functionality may be limited.")
        
        # Initialize sentence transformer for answer evaluation
        try:
            self.sentence_model = SentenceTransformer(embedding_model_name)
            print(f"✅ Initialized sentence transformer model for answer evaluation")
        except:
            print(f"❌ Failed to initialize sentence transformer. Answer evaluation will be limited.")
        
        # Load FAISS index if it exists
        try:
            self.db = db
            print(f"✅ FAISS index loaded from global variable")
        except:
            try:
                self.db = FAISS.load_local(faiss_index_path, self.embedding_model, allow_dangerous_deserialization=True)
                print(f"✅ FAISS index loaded from '{faiss_index_path}'")
            except Exception as e:
                print(f"❌ Failed to load FAISS index: {e}")
        
        # Use the global reranker model if available
        try:
            # self.reranker = reranker_model
            if reranker_model_name.startswith("Cohere"):
                self.reranker = CohereReranker(api_key=os.getenv("COHERE_API_KEY"), model="rerank-v3.5")
            else:
                self.reranker = CrossEncoder(reranker_model_name)

            print("✅ Using global reranker model")
        except:
            try:
                self.reranker = CrossEncoder(reranker_model_name)
                print(f"✅ Initialized new reranker model: {reranker_model_name}")
            except Exception as e:
                print(f"❌ Failed to initialize reranker model: {e}")
        
        # Initialize faithfulness cross-encoder model
        try:
            self.faithfulness_model = CrossEncoder(faithfulness_model_name)
            print(f"✅ Initialized faithfulness model: {faithfulness_model_name}")
        except Exception as e:
            print(f"❌ Failed to initialize faithfulness model: {e}")
            self.faithfulness_model = None
        
        # Check if we can use the generate_llama_answer function
        try:
            self.generate_answer_fn = generate_llama_answer
            print("✅ Using global generate_llama_answer function")
        except:
            print("❌ generate_llama_answer function not available. Will use hybrid search only.")
            self.generate_answer_fn = None
    
    def load_test_data(self, test_data_path: str) -> List[Dict[str, Any]]:
        """
        Loads test data from a JSON file.
        Expected format: List of dicts with 'question' and 'reference_answer' keys.
        """
        # Check if file exists
        if not os.path.exists(test_data_path):
            print(f"❌ Test data file not found: {test_data_path}")
            return []
        
        with open(test_data_path, 'r') as f:
            content = f.read()
            
            # Handle JSON list format
            if content.strip().startswith('['):
                test_data = json.loads(content)
            else:
                # Handle JSONL format (one JSON object per line)
                test_data = []
                for line in content.strip().split('\n'):
                    if line.strip():
                        try:
                            test_data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line: {line}")
                            print(f"Error details: {e}")
        
        # Validate the format of test data
        valid_data = []
        for item in test_data:
            if 'question' in item and 'reference_answer' in item:
                valid_data.append(item)
            else:
                print(f"⚠️ Skipping invalid test item: {item}")
        
        print(f"✅ Loaded {len(valid_data)} valid test questions")
        return valid_data
    
    def create_test_data_template(self, output_path: str, num_examples: int = 10):
        """
        Creates a template for test data that can be filled manually.
        """
        template = []
        for i in range(num_examples):
            template.append({
                "question": f"Example question {i+1}",
                "reference_answer": "The reference answer for this question."
            })
        
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        print(f"✅ Created test data template at {output_path}")
    
    def retrieve_docs(self, query: str, k: int = 10) -> List[Tuple[Any, float]]:
        """
        Retrieves documents for a query using hybrid search.
        Returns a list of (doc, score) pairs.
        """
        try:
            # Use hybrid search if available
            hybrid_results = hybrid_search(query, top_k=k*2)
            
            # Rerank using cross-encoder
            pairs = [(query, doc.page_content) for doc, _ in hybrid_results]
            scores = self.reranker.predict(pairs)
            reranked = sorted(zip([doc for doc, _ in hybrid_results], scores), key=lambda x: x[1], reverse=True)
            
            return reranked[:k]
        except Exception as e:
            print(f"Error in hybrid retrieval: {e}")
            # Fallback to basic similarity search
            try:
                results = self.db.similarity_search_with_score(query, k=k)
                return results
            except Exception as e2:
                print(f"Error in fallback retrieval: {e2}")
                return []
    
    def generate_answer(self, query: str, session_id: str = "eval_enhanced_8b") -> str:
        """
        Generates an answer using the chatbot's generation function.
        Falls back to simple document concatenation if not available.
        """
        if self.generate_answer_fn:
            try:
                # Use the generate_llama_answer function if available
                return self.generate_answer_fn(query, session_id)
            except Exception as e:
                print(f"Error generating answer with LLM: {e}")
        
        # Fallback: concatenate retrieved documents
        try:
            retrieved_docs = self.retrieve_docs(query, k=3)
            if not retrieved_docs:
                return "No information found."
            
            # Extract relevant snippets from top docs
            snippets = [doc.page_content[:300] + "..." for doc, _ in retrieved_docs[:3]]
            
            # Create a simple answer
            answer = "\n\n".join(snippets)
            return answer
        except Exception as e:
            print(f"Error in fallback answer generation: {e}")
            return "Unable to generate answer due to an error."
    
    def calculate_semantic_similarity(self, generated_answer: str, reference_answer: str) -> float:
        """
        Calculates semantic similarity between generated and reference answers using embeddings.
        
        Returns a score between 0 and 1, where 1 indicates perfect similarity.
        """
        # Check if either answer is empty
        if not generated_answer or not reference_answer:
            return 0.0
        
        # Calculate embedding similarity using sentence-transformers
        try:
            gen_embedding = self.sentence_model.encode([generated_answer])[0]
            ref_embedding = self.sentence_model.encode([reference_answer])[0]
            
            # Calculate cosine similarity
            similarity = cosine_similarity([gen_embedding], [ref_embedding])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating embedding similarity: {e}")
            return 0.0
    
    def calculate_faithfulness(self, generated_answer: str, retrieved_docs: List[Tuple[Any, float]]) -> float:
        """
        Calculates faithfulness of the generated answer to the retrieved documents.
        
        This evaluates whether the generated answer is supported by and consistent with 
        the information in the retrieved documents.
        
        Returns a score between 0 and 1, where 1 indicates high faithfulness.
        """
        if not generated_answer or not retrieved_docs:
            return 0.0
        
        try:
            if self.faithfulness_model:
                # Create faithfulness pairs for each retrieved document
                # Premise: document content, Hypothesis: generated answer
                faithfulness_scores = []
                for doc, _ in retrieved_docs[:3]:  # Check against top 3 docs
                    premise = doc.page_content
                    hypothesis = generated_answer
                    # Use NLI model to predict entailment
                    score = self.faithfulness_model.predict([(premise, hypothesis)])
                    faithfulness_scores.append(float(score))
                
                # Use max score as faithfulness score
                # Alternative: average of scores
                return max(faithfulness_scores) if faithfulness_scores else 0.0
            else:
                # Fallback if no faithfulness model: use semantic similarity
                doc_contents = " ".join([doc.page_content for doc, _ in retrieved_docs[:3]])
                return self.calculate_semantic_similarity(generated_answer, doc_contents)
        except Exception as e:
            print(f"Error calculating faithfulness: {e}")
            return 0.0
    
    def calculate_answer_relevance(self, query: str, generated_answer: str) -> float:
        """
        Calculates relevance of the generated answer to the query.
        
        Uses the reranker model to evaluate how relevant the answer is to the query.
        
        Returns a score between 0 and 1, where 1 indicates high relevance.
        """
        if not query or not generated_answer:
            return 0.0
        
        try:
            # Use reranker model to calculate relevance
            relevance_score = self.reranker.predict([(query, generated_answer)])
            
            # Normalize the score to [0, 1] if needed
            if relevance_score > 1.0:
                relevance_score = 1.0
            elif relevance_score < 0.0:
                relevance_score = 0.0
                
            return float(relevance_score)
        except Exception as e:
            print(f"Error calculating answer relevance: {e}")
            # Fallback: use semantic similarity
            return self.calculate_semantic_similarity(query, generated_answer)
    
    def evaluate_with_synthetic_relevance(
        self, 
        query: str, 
        reference_answer: str,
        k_values: List[int] = [1, 3, 5, 10],
        top_n_as_relevant: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluates using synthetic relevance judgments and generation metrics.
        
        The top_n_as_relevant parameter determines how many documents should be 
        considered relevant based on similarity to the reference answer.
        """
        # Step 1: Retrieve documents for the query
        max_k = max(k_values + [top_n_as_relevant])  # Ensure we retrieve enough docs
        retrieved_docs = self.retrieve_docs(query, k=max_k)
        
        if not retrieved_docs:
            print(f"No documents retrieved for query: {query}")
            return {
                "query": query,
                "semantic_similarity": 0.0,
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "retrieved_count": 0
            }
        
        # Step 2: Generate an answer
        generated_answer = self.generate_answer(query)
        
        # Step 3: Calculate generation evaluation metrics
        semantic_similarity = self.calculate_semantic_similarity(generated_answer, reference_answer)
        faithfulness = self.calculate_faithfulness(generated_answer, retrieved_docs)
        answer_relevance = self.calculate_answer_relevance(query, generated_answer)
        
        # Step 4: Create synthetic relevance judgments by comparing docs to reference answer
        doc_similarities = []
        for doc, _ in retrieved_docs:
            doc_sim = self.calculate_semantic_similarity(doc.page_content, reference_answer)
            doc_similarities.append((doc, doc_sim))
        
        # Sort by similarity to reference answer
        sorted_docs = sorted(doc_similarities, key=lambda x: x[1], reverse=True)
        
        # Top N most similar docs are considered "relevant"
        # relevant_docs = [doc for doc, _ in sorted_docs[:top_n_as_relevant]]
        relevant_docs = [doc for doc, sim in doc_similarities if sim >= 0.3]

        
        # Step 5: Calculate retrieval metrics using these synthetic judgments
        results = {
            "query": query,
            "semantic_similarity": semantic_similarity,
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "retrieved_count": len(retrieved_docs),
            "mrr": self._calculate_mrr(retrieved_docs, relevant_docs)
        }
        
        # Calculate precision, recall, and ndcg at different k values
        for k in k_values:
            if k > len(retrieved_docs):
                continue
                
            topk_docs = [doc for doc, _ in retrieved_docs[:k]]
            results[f"precision@{k}"] = self._calculate_precision(topk_docs, relevant_docs)
            results[f"recall@{k}"] = self._calculate_recall(topk_docs, relevant_docs)
            results[f"ndcg@{k}"] = self._calculate_ndcg(retrieved_docs[:k], relevant_docs)
        
        return results
    
    def _calculate_precision(self, retrieved_docs: List, relevant_docs: List) -> float:
        """Calculate precision."""
        if not retrieved_docs:
            return 0.0
            
        relevant_retrieved = sum(1 for doc in retrieved_docs if doc in relevant_docs)
        return relevant_retrieved / len(retrieved_docs)
    
    def _calculate_recall(self, retrieved_docs: List, relevant_docs: List) -> float:
        """Calculate recall."""
        if not relevant_docs:
            return 1.0  # All relevant docs retrieved (there are none)
            
        relevant_retrieved = sum(1 for doc in retrieved_docs if doc in relevant_docs)
        return relevant_retrieved / len(relevant_docs)
    
    def _calculate_mrr(self, retrieved_docs: List[Tuple[Any, float]], relevant_docs: List) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, (doc, _) in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_ndcg(self, retrieved_docs: List[Tuple[Any, float]], relevant_docs: List) -> float:
        """Calculate NDCG."""
        relevance = [1 if doc in relevant_docs else 0 for doc, _ in retrieved_docs]
        
        if sum(relevance) == 0:
            return 0.0
            
        # Calculate DCG
        dcg = 0
        for i, rel in enumerate(relevance):
            dcg += rel / np.log2(i + 2)  # i+2 because i is 0-indexed
        
        # Calculate ideal DCG (IDCG)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = 0
        for i, rel in enumerate(ideal_relevance):
            idcg += rel / np.log2(i + 2)
        
        return dcg / idcg
    
    def evaluate_test_set(
        self, 
        test_data: List[Dict[str, Any]], 
        k_values: List[int] = [1, 3, 5, 10],
        top_n_as_relevant: int = 3,
        output_path: str = "rag-eval-enhanced-llama-8b/rag_evaluation_results.csv",
        detailed_output_path: str = "rag-eval-enhanced-llama-8b/rag_evaluation_detailed.json"
    ) -> pd.DataFrame:
        """
        Evaluates the entire test set and returns results as a DataFrame.
        """
        all_results = []
        
        for test_item in tqdm(test_data, desc="Evaluating questions"):
            query = test_item["question"]
            reference_answer = test_item["reference_answer"]
            
            try:
                result = self.evaluate_with_synthetic_relevance(
                    query, reference_answer, k_values, top_n_as_relevant
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error evaluating query '{query}': {e}")
                all_results.append({
                    "query": query,
                    "error": str(e)
                })
        
        # Save detailed results
        if detailed_output_path:
            with open(detailed_output_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"✅ Saved detailed evaluation results to {detailed_output_path}")
        
        # Convert to DataFrame for summary metrics
        results_df = pd.DataFrame(all_results)
        
        # Calculate averages
        metric_cols = [col for col in results_df.columns if col != "query" and col != "error"]
        avg_results = {}
        for col in metric_cols:
            avg_results[col] = results_df[col].mean()
        
        # Add average row
        avg_df = pd.DataFrame([avg_results], index=["AVERAGE"])
        results_df = pd.concat([results_df, avg_df])
        
        # Save summary results
        if output_path:
            results_df.to_csv(output_path)
            print(f"✅ Saved evaluation summary to {output_path}")
        
        return results_df
    
    def plot_results(self, results_df: pd.DataFrame, output_path: str = "rag-eval-enhanced-llama-8b/evaluation_plots.png"):
        """
        Plots evaluation results.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Extract average results
            avg_results = results_df.loc["AVERAGE"].to_dict()
            
            # Group metrics by type
            precision_metrics = {k: v for k, v in avg_results.items() if k.startswith("precision")}
            recall_metrics = {k: v for k, v in avg_results.items() if k.startswith("recall")}
            ndcg_metrics = {k: v for k, v in avg_results.items() if k.startswith("ndcg")}
            
            # Generation metrics
            generation_metrics = {
                "Semantic Similarity": avg_results.get("semantic_similarity", 0),
                "Faithfulness": avg_results.get("faithfulness", 0),
                "Answer Relevance": avg_results.get("answer_relevance", 0),
            }
            
            other_metrics = {
                "MRR": avg_results.get("mrr", 0),
            }
            
            # Create subplots - now with 3 rows to include generation metrics
            fig, axs = plt.subplots(3, 2, figsize=(15, 15))
            fig.suptitle("RAG Evaluation Results", fontsize=16)
            
            # Plot precision
            if precision_metrics:
                axs[0, 0].bar(precision_metrics.keys(), precision_metrics.values())
                axs[0, 0].set_title("Precision@k")
                axs[0, 0].set_ylim(0, 1)
            else:
                axs[0, 0].text(0.5, 0.5, "No precision metrics available", 
                              horizontalalignment='center', verticalalignment='center')
            
            # Plot recall
            if recall_metrics:
                axs[0, 1].bar(recall_metrics.keys(), recall_metrics.values())
                axs[0, 1].set_title("Recall@k")
                axs[0, 1].set_ylim(0, 1)
            else:
                axs[0, 1].text(0.5, 0.5, "No recall metrics available", 
                              horizontalalignment='center', verticalalignment='center')
            
            # Plot NDCG
            if ndcg_metrics:
                axs[1, 0].bar(ndcg_metrics.keys(), ndcg_metrics.values())
                axs[1, 0].set_title("NDCG@k")
                axs[1, 0].set_ylim(0, 1)
            else:
                axs[1, 0].text(0.5, 0.5, "No NDCG metrics available", 
                              horizontalalignment='center', verticalalignment='center')
            
            # Plot other retrieval metrics
            axs[1, 1].bar(other_metrics.keys(), other_metrics.values())
            axs[1, 1].set_title("Other Retrieval Metrics")
            axs[1, 1].set_ylim(0, 1)
            
            # Plot generation metrics
            axs[2, 0].bar(generation_metrics.keys(), generation_metrics.values())
            axs[2, 0].set_title("Generation Metrics")
            axs[2, 0].set_ylim(0, 1)
            
            # Leave the last subplot empty or use it for additional metrics if needed
            axs[2, 1].axis('off')
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(output_path)
            print(f"✅ Saved evaluation plots to {output_path}")
            
        except ImportError:
            print("⚠️ Matplotlib and/or Seaborn not installed. Skipping plot generation.")


def main():
    # Initialize evaluator
    evaluator = RAGEvaluator()
    # evaluator = RAGEvaluator(
    #     embedding_model_name="BAAI/bge-base-en-v1.5",
    #     reranker_model_name="Cohere/rerank-v3.5"
    # )

    
    # Process command line arguments (if provided)
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate RAG system')
    parser.add_argument('--test_data', type=str, default='rag_test_data.json', help='Path to test data file')
    parser.add_argument('--output', type=str, default='rag-eval-enhanced-llama-8b/_evaluation_results.csv', help='Path to output CSV file')
    parser.add_argument('--detailed_output', type=str, default='rag-eval-enhanced-llama-8b/rag_evaluation_detailed.json', help='Path to detailed output JSON file')
    parser.add_argument('--plot', type=str, default='rag-eval-enhanced-llama-8b/evaluation_plots.png', help='Path to output plot file')
    parser.add_argument('--top_n_relevant', type=int, default=3, help='Number of top documents to consider relevant')
    args = parser.parse_args()
    
    # Create test data template if it doesn't exist
    if not os.path.exists(args.test_data):
        evaluator.create_test_data_template(args.test_data)
        print(f"Created test data template at {args.test_data}")
        print(f"Please fill in the test data with real questions and reference answers, then run this script again.")
        return
    
    # Load test data
    test_data = evaluator.load_test_data(args.test_data)
    if not test_data:
        print("No valid test data found. Exiting.")
        return
    
    # Evaluate
    results_df = evaluator.evaluate_test_set(
        test_data,
        k_values=[1, 3, 5, 10],
        top_n_as_relevant=args.top_n_relevant,
        output_path=args.output,
        detailed_output_path=args.detailed_output
    )
    
    # Print summary results
    print("\n📊 EVALUATION RESULTS SUMMARY 📊")
    print(results_df.loc["AVERAGE"])
    
    # Plot results
    evaluator.plot_results(results_df, args.plot)


if __name__ == "__main__":
    main()
