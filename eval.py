import os
import time
from datasets import Dataset
from google import genai
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import llm_factory
from langchain_huggingface import HuggingFaceEmbeddings

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from agent import app as agent_app


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"A variável de ambiente {name} não está definida. "
            f"Defina antes de executar. Ex.: export {name}='sua_chave'"
        )
    return value


def run_evaluation() -> None:
    print("=" * 60)
    print("AVALIAÇÃO AGENTE SUS UFCG")
    print("=" * 60)

    google_api_key = get_required_env("GOOGLE_API_KEY")
    client = genai.Client(api_key=google_api_key)

    ragas_llm = llm_factory(
        "gemini-2.0-flash",
        provider="google",
        client=client,
    )

    ragas_emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    questions = [
        "Quais são os princípios doutrinários do SUS?",
        "O que significa a diretriz de descentralização no SUS?",
        "Qual é a função da Atenção Básica?",
        "Quem tem direito ao atendimento no SUS?",
        "O que é o princípio da Equidade no SUS?",
        "Como a comunidade participa do SUS?",
        "O SUS fornece vacinas gratuitas?",
        "O que é a integralidade no SUS?",
        "Qual o papel do Ministério da Saúde?",
        "Como funciona a hierarquização dos serviços de saúde?",
    ]

    ground_truths = [
        "Universalidade, Equidade e Integralidade.",
        "Redistribuir responsabilidades entre União, Estados e Municípios.",
        "Porta de entrada e foco na prevenção.",
        "Todos os cidadãos em território nacional.",
        "Tratar desigualmente os desiguais para promover justiça.",
        "Conselhos e Conferências de Saúde.",
        "Sim, gratuitamente via PNI.",
        "Assistência completa da prevenção à alta complexidade.",
        "Formular políticas nacionais e coordenar o sistema.",
        "Níveis de complexidade crescente, como atenção primária, secundária e terciária.",
    ]

    answers = []
    contexts = []

    for idx, question in enumerate(questions, start=1):
        print(f"Processando pergunta {idx}/{len(questions)}: {question}")
        result = agent_app.invoke({"question": question})
        response_text = result.get("generation", "")
        docs = result.get("documents", [])

        answers.append(response_text)
        contexts.append([doc.page_content for doc in docs])

        time.sleep(3)

    hf_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]

    run_config = RunConfig(
        timeout=180,
        max_retries=10,
        max_workers=1,
    )

    print("\nCalculando métricas finais...")

    result = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_config,
    )

    print("\nRELATÓRIO FINAL:")
    print(result)

    df = result.to_pandas()
    df.to_csv("avaliacao_sus_ufcg.csv", index=False)
    print("\nSucesso. Resultados salvos em: avaliacao_sus_ufcg.csv")


if __name__ == "__main__":
    run_evaluation()