from dotenv import load_dotenv
from src.agent import app_graph

# Carrega API KEY
load_dotenv()

def main():
    print("--- Agente de Filmes Iniciado ---")
    print("Pergunte sobre filmes ou peça recomendações (ex: 'Recomende filmes para o usuario 1')")
    
    while True:
        try:
            user_input = input("\nVocê: ")
            if user_input.lower() in ["sair", "exit"]:
                break
                
            # Invoca o grafo
            # Referência: "grafo.invoke"
            resultado = app_graph.invoke({"pergunta": user_input})
            
            print(f"Agente: {resultado['resposta_final']}")
            
        except Exception as e:
            print(f"Erro: {e}")

if __name__ == "__main__":
    main()
