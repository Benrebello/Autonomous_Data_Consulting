# Sistema de Rate Limiting e Controle de Requisições

## Visão Geral

Sistema robusto de controle de requisições implementado para prevenir erros de rate limit e token overflow, com temporizador visual e retry automático.

## Componentes Implementados

### 1. `rate_limiter.py` - Módulo Central de Rate Limiting

**Classe `RateLimiter`:**
- Controle de RPM (requisições por minuto)
- Estimativa de tokens (~4 chars por token)
- Detecção automática de erros de rate limit e token limit
- Backoff exponencial (2^n segundos, máx 60s)
- Extração de `retry-after` de mensagens de erro
- Retry automático com até 3 tentativas

**Padrões de erro detectados:**
- Rate limit: `"rate limit"`, `"too many requests"`, `"429"`, `"rpm"`
- Token limit: `"token limit"`, `"context length"`, `"tokens exceeded"`

**Método principal:**
```python
execute_with_retry(func, max_retries=3, on_wait=callback)
```

### 2. `ui_components.py` - Componentes Visuais

**Função `display_rate_limit_timer(wait_info)`:**
- Temporizador visual com countdown
- Barra de progresso animada
- Exibição de motivo e horário de retry
- Expander com detalhes do erro (opcional)

**Função `display_rate_limit_info(rpm_limit, max_tokens)`:**
- Expander na sidebar com informações de limites
- Explicação do comportamento automático
- Dicas para otimização

**Classe `RateLimitHandler`:**
- Callback para eventos de wait do rate limiter
- Integração com Streamlit
- Fallback para sleep simples fora do contexto Streamlit

### 3. Integração em `agents.py`

**Mudanças em `BaseAgent`:**
- Aceita `rate_limiter` opcional no construtor
- Método `set_wait_callback()` para UI
- Compartilhamento de rate limiter entre agentes

**Agentes atualizados:**
- `OrchestratorAgent.run()`
- `TeamLeaderAgent.create_plan()`
- `TeamLeaderAgent.synthesize_results()`
- `DataAnalystBusinessAgent.generate_final_response()`

Todos usam `rate_limiter.execute_with_retry()` com callback para UI.

### 4. Integração em `app.py`

**No `AnalysisPipeline.__init__()`:**
```python
self.rate_limiter = get_rate_limiter(rpm_limit=rpm_limit)
self.rate_limit_handler = RateLimitHandler()

# Todos os agentes compartilham o mesmo rate limiter
for agent in [self.orchestrator, self.team_leader] + list(self.agents.values()):
    agent.set_wait_callback(self.rate_limit_handler.on_wait)
```

**Na sidebar (função `main()`):**
- Expander com informações de rate limit após o slider de RPM
- Exibição automática quando limites são atingidos

## Fluxo de Funcionamento

### Cenário 1: Requisição Normal
1. Agente chama `rate_limiter.execute_with_retry(func)`
2. Rate limiter verifica se precisa aguardar (RPM)
3. Se não, executa a função imediatamente
4. Sucesso → reseta contador de erros

### Cenário 2: Rate Limit Atingido
1. API retorna erro 429 ou similar
2. Rate limiter detecta o padrão de erro
3. Extrai `retry-after` (se disponível) ou calcula backoff
4. Chama `on_wait(wait_info)` → `RateLimitHandler.on_wait()`
5. UI exibe temporizador visual com countdown
6. Após espera, retry automático
7. Até 3 tentativas antes de falhar

### Cenário 3: Token Limit Excedido
1. API retorna erro de contexto muito grande
2. Rate limiter detecta padrão de token limit
3. Aplica backoff exponencial
4. Exibe temporizador com motivo "Token limit"
5. Retry automático (usuário deve reduzir dados)

## Exemplo de Uso

```python
from rate_limiter import RateLimiter
from ui_components import RateLimitHandler

# Criar rate limiter
limiter = RateLimiter(rpm_limit=10, max_tokens_per_request=8000)

# Criar handler de UI
handler = RateLimitHandler()

# Executar com retry automático
def my_api_call():
    return llm.invoke(prompt)

result = limiter.execute_with_retry(
    my_api_call,
    max_retries=3,
    on_wait=handler.on_wait  # Callback para UI
)
```

## Configuração na Interface

**Sidebar → LLM Settings:**
1. **Max Requests per Minute (RPM)**: Slider de 1-60
2. **ℹ️ Limites de API**: Expander com:
   - Configuração atual (RPM e tokens)
   - Comportamento automático
   - Dicas de otimização

**Quando rate limit é atingido:**
- Temporizador visual aparece automaticamente
- Barra de progresso mostra tempo restante
- Mensagem clara sobre o motivo
- Retry automático sem intervenção do usuário

## Benefícios para Usuários Não Técnicos

1. **Transparência**: Usuário vê exatamente quanto tempo falta
2. **Sem erros criptográficos**: Mensagens claras em português
3. **Automático**: Sistema gerencia tudo sozinho
4. **Educativo**: Expander explica os limites e dá dicas
5. **Confiável**: Backoff exponencial previne loops infinitos

## Testes

Todos os testes existentes continuam passando:
- `test_outliers.py` ✓
- `test_time_features.py` ✓
- `test_time_series.py` ✓
- `test_tools_mapping.py` ✓
- `test_types.py` ✓

O `AnalysisPipeline` nos testes usa `rpm_limit=1000` para evitar delays.

## Arquivos Modificados

1. **Novos arquivos:**
   - `rate_limiter.py` (234 linhas)
   - `ui_components.py` (155 linhas)
   - `docs/RATE_LIMITING.md` (este arquivo)

2. **Arquivos modificados:**
   - `agents.py`: Integração do rate limiter em todos os agentes
   - `app.py`: Import dos componentes e exibição na sidebar

3. **Sem breaking changes**: Todos os testes passam sem modificação

## Próximos Passos (Opcionais)

- [ ] Adicionar métricas de uso de API no expander de Analytics
- [ ] Persistir histórico de rate limits em session_state
- [ ] Alertar proativamente quando próximo do limite
- [ ] Sugerir ajuste de RPM baseado em padrões de uso
