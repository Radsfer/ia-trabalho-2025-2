import sys

def ask_question(prompt, options=None):
    """
    Função genérica para fazer uma pergunta e validar a resposta 
    contra uma lista de opções.
    """
    if options is None:
        options = ["sim", "não"]
    
    prompt_string = f"\n{prompt} ({'/'.join(options)}): "
    
    while True:
        answer = input(prompt_string).strip().lower()
        
        for opt in options:
            if opt.startswith(answer):
                return opt
        
        print(f"🚨 Resposta inválida. Por favor, digite uma das opções: {options}")

def print_result(result):
    """ Imprime a sugestão final (a folha da árvore). """
    print("\n" + "="*30)
    print(f"👉 Sugestão: {result}")
    print("="*30)

def main():
    """
    Executa a árvore de decisão principal, baseada no 
    tree_diagram.md 
    """
    print("--- Árvore da Verdade da Programação ---")
    
    try:
        q1 = ask_question("Você quer ganhar dinheiro?")
        
        if q1 == "sim":
            q2 = ask_question("Gosta de ser bem pago pra sofrer?")
            
            if q2 == "sim":
                q3 = ask_question("Trabalhar em banco ou seguradora?", ["banco", "seguradora"])
                
                if q3 == "banco":
                    print_result("Java (O Padrão-Ouro™)")
                else: # seguradora
                    print_result("C# (O Java da Microsoft)")
            
            else: # q2 == "não"
                q4 = ask_question("Quer que a dor seja rápida?")
                
                if q4 == "sim":
                    q5 = ask_question("Gosta mais de 'mágica' ou 'ordem'?", ["mágica", "ordem"])
                    
                    if q5 == "mágica":
                        print_result("Python (import solucao)")
                    else: # ordem
                        print_result("Go (Rápido e chato, como deve ser)")
                
                else: # q4 == "não"
                    print_result("JavaScript (Parabéns, agora sofre LENTO)")

        else: # q1 == "não"
            q6 = ask_question("Quer se sentir mais inteligente que os outros?")
            
            if q6 == "sim":
                q7 = ask_question("...mas sem ter que gerenciar memória?")
                
                if q7 == "sim":
                    q8 = ask_question("E quer provar que é realmente diferente?")
                    
                    if q8 == "sim":
                        print_result("Rust (O Futuro®, confia)")
                    else: # q8 == "não"
                        print_result("Haskell (Ninguém vai entender seu código)")
                
                else: # q7 == "não"
                    print_result("C++ (Sofrimento Clássico)")
            
            else: # q6 == "não"
                q9 = ask_question("É só pra fazer um sitezinho pro seu primo?")
                
                if q9 == "sim":
                    q10 = ask_question("Esse site tem um banco de dados?")
                    
                    if q10 == "sim":
                        print_result("PHP (Sim, ainda vive. E paga algo.)")
                    else: # q10 == "não"
                        print_result("HTML/CSS (Isso nem é programar)")
                
                else: # q9 == "não"
                    print_result("Scratch, seja feliz")

    except (KeyboardInterrupt, EOFError):
        print("\n\nSaindo. Decisão difícil, né? 🤷")
        sys.exit(0)

if __name__ == "__main__":
    main()
