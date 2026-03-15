import os
import datetime

class LocalMCPServer:
    """
    Servidor MCP (Model Context Protocol) Local Simplificado.
    Focado nos requisitos de Segurança da avaliação:
    1. Limitado a uma pasta específica (Sandboxing).
    2. Allowlist de comandos.
    3. Registro de logs (Auditoria).
    """
    def __init__(self):
        # 1. Limita o acesso a arquivos APENAS a esta pasta
        self.allowed_dir = os.path.join(os.getcwd(), "dados_permitidos")
        os.makedirs(self.allowed_dir, exist_ok=True)
        
        # 2. Allowlist de ferramentas que o Agente pode chamar
        self.allowlist_functions = ["salvar_arquivo_triagem"]

    def log_action(self, action, details):
        """3. Registra todas as chamadas no log de segurança."""
        with open("mcp_security.log", "a", encoding="utf-8") as log:
            log.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {action} | {details}\n")

    def executar_tool(self, tool_name, **kwargs):
        """Ponto de entrada único para o Agente chamar ferramentas."""
        self.log_action("TOOL_CALL_ATTEMPT", f"Tentativa de uso da tool: {tool_name}")
        
        # Verifica Allowlist
        if tool_name not in self.allowlist_functions:
            self.log_action("TOOL_CALL_BLOCKED", f"Tool '{tool_name}' recusada (fora da allowlist).")
            return f"Erro de Segurança do MCP: Ferramenta '{tool_name}' não autorizada."
        
        # Executa a ferramenta permitida
        if tool_name == "salvar_arquivo_triagem":
            return self._salvar_arquivo(kwargs.get("nome_arquivo"), kwargs.get("conteudo"))

    def _salvar_arquivo(self, nome_arquivo, conteudo):
        """Ferramenta real: Salva o arquivo com proteção de caminho."""
        # Evita "Path Traversal" (ex: tentar salvar em ../../windows/system32)
        safe_name = os.path.basename(nome_arquivo)
        filepath = os.path.join(self.allowed_dir, safe_name)
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(conteudo)
            self.log_action("TOOL_CALL_SUCCESS", f"Arquivo salvo com sucesso em: {filepath}")
            return f"Sucesso: Arquivo '{safe_name}' criado de forma segura no diretório '{self.allowed_dir}'."
        except Exception as e:
            self.log_action("TOOL_CALL_ERROR", f"Erro ao salvar {safe_name}: {str(e)}")
            return "Erro interno no servidor MCP ao salvar o arquivo."