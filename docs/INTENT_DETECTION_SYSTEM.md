# Intent Detection System - Documentation

## Overview

The system now features an intelligent LLM-based intent detection system that analyzes user queries in combination with DataFrame characteristics to determine the most appropriate action.

## Intent Models

### 8 Predefined Intent Models

#### 1. **SIMPLE_QUERY**
- **Description**: Direct and specific question
- **Examples**: 
  - "Qual a média?"
  - "Existem outliers?"
  - "Quantas linhas?"
- **Characteristics**: Short query (≤10 words), objective answer
- **Action**: Execute 1 specific tool
- **Response Mode**: Direct

#### 2. **EXPLORATORY_ANALYSIS**
- **Description**: Complete exploratory analysis
- **Examples**:
  - "Faça EDA"
  - "Analise os dados"
  - "Explore o dataset"
- **Characteristics**: Broad request, multiple analyses
- **Action**: Multi-step plan with profiling + visualizations + correlations
- **Response Mode**: Complete

#### 3. **REPORT_GENERATION**
- **Description**: Report generation
- **Examples**:
  - "Crie um relatório"
  - "Gere documentação"
  - "Resuma análises"
- **Characteristics**: Requests formal document
- **Action**: Compile previous analyses + generate PDF
- **Response Mode**: Complete

#### 4. **VISUALIZATION_REQUEST**
- **Description**: Chart/graph request
- **Examples**:
  - "Mostre histogramas"
  - "Plote correlações"
  - "Visualize distribuições"
- **Characteristics**: Focus on visual representation
- **Action**: Generate specific charts
- **Response Mode**: Direct (single chart) or Complete (multiple)

#### 5. **STATISTICAL_TEST**
- **Description**: Specific statistical test
- **Examples**:
  - "Teste de hipótese"
  - "ANOVA"
  - "Correlação de Pearson"
- **Characteristics**: Mentions formal statistical test
- **Action**: Execute specific test
- **Response Mode**: Direct

#### 6. **PREDICTIVE_MODELING**
- **Description**: Predictive modeling
- **Examples**:
  - "Preveja X"
  - "Crie modelo"
  - "Classifique Y"
- **Characteristics**: Requests prediction or classification
- **Action**: Train and evaluate ML model
- **Response Mode**: Complete

#### 7. **DATA_CLEANING**
- **Description**: Data cleaning
- **Examples**:
  - "Limpe os dados"
  - "Remova duplicatas"
  - "Trate valores faltantes"
- **Characteristics**: Focus on quality and preparation
- **Action**: Validation + cleaning + transformation
- **Response Mode**: Complete

#### 8. **AMBIGUOUS**
- **Description**: Unclear intent
- **Examples**:
  - "Ajude-me"
  - "O que fazer?"
  - "Não sei"
- **Characteristics**: Vague query without clear direction
- **Action**: Start conversational discovery
- **Response Mode**: Interactive

## DataFrame Context Analysis

The system analyzes DataFrame characteristics to inform intent detection:

```python
df_context = {
    'n_rows': len(df),
    'n_columns': len(df.columns),
    'n_numeric': len(numeric_cols),
    'n_categorical': len(categorical_cols),
    'has_time': bool(time_columns),
    'has_target': 'class' in df.columns,
    'columns': df.columns[:10].tolist()
}
```

### Automatic Complex Data Detection

The system automatically triggers full analysis for:
- **Many numeric columns**: ≥5 numeric columns
- **Large datasets**: ≥1000 rows
- **Time series with target**: Has time column AND target variable

## LLM Response Format

The intent detection LLM returns structured JSON:

```json
{
    "detected_intent": "REPORT_GENERATION",
    "confidence": 0.95,
    "reasoning": "User requests formal report of analyses",
    "suggested_tools": ["get_exploratory_analysis", "generate_pdf_report"],
    "requires_clarification": false,
    "clarification_questions": [],
    "data_compatibility": "compatible",
    "recommended_scope": "full_analysis"
}
```

## Routing Logic

### Decision Tree

```
User Query + DataFrame Context
    ↓
Intent Detection (LLM)
    ↓
┌─────────────────────────────────────┐
│ Confidence ≥ 0.7?                   │
│ requires_clarification = false?     │
└─────────────────────────────────────┘
    ↓ YES                    ↓ NO
    │                        │
    │                   Ask clarification
    │                   questions
    ↓
Route by detected_intent:
    │
    ├─ SIMPLE_QUERY → Execute single tool (direct mode)
    ├─ EXPLORATORY_ANALYSIS → Full multi-step plan
    ├─ REPORT_GENERATION → Compile + PDF
    ├─ VISUALIZATION_REQUEST → Generate charts
    ├─ STATISTICAL_TEST → Execute test
    ├─ PREDICTIVE_MODELING → Train model
    ├─ DATA_CLEANING → Clean workflow
    └─ AMBIGUOUS → Conversational discovery
```

## Integration in app.py

### Location: Lines 1245-1382

**Flow:**
1. Check if intent_mode already set (skip if yes)
2. Analyze DataFrame characteristics
3. Call LLM with INTENT_DETECTION_PROMPT
4. Parse JSON response
5. Display detection in expander (for transparency)
6. Route based on detected intent
7. Fallback to keyword-based detection if LLM fails

**Code:**
```python
# Analyze DataFrame
df_context = {
    'n_rows': len(df),
    'n_numeric': len(numeric_cols),
    # ...
}

# Call LLM
intent_prompt = INTENT_DETECTION_PROMPT.format(
    data_context=json.dumps(df_context),
    user_query=user_query
)
intent_response = llm.invoke(intent_prompt).content
intent_data = json.loads(intent_response)

# Route
if detected_intent == 'SIMPLE_QUERY':
    briefing = create_simple_briefing(...)
elif detected_intent == 'REPORT_GENERATION':
    briefing = orchestrator.run(enriched_query)
# ...
```

## Fallback Mechanism

If LLM-based detection fails, the system falls back to keyword-based detection:

```python
except Exception as e:
    st.warning(f"⚠️ Intent detection via LLM failed: {str(e)}")
    st.info("Using keyword-based detection...")
    
    # Fallback to keywords
    kw_complete = ['relatório', 'report', 'detalhado', ...]
    if any(k in query_lower for k in kw_complete):
        briefing = orchestrator.run(user_query)
```

## Benefits

### 1. Intelligent Routing
- Analyzes query semantics, not just keywords
- Considers DataFrame characteristics
- Suggests scope expansion when appropriate

### 2. Transparent Decision Making
- Shows detection result in UI expander
- Displays confidence score
- Explains reasoning

### 3. Adaptive Clarification
- Asks questions only when confidence < 0.7
- Questions are contextual and specific
- Avoids unnecessary back-and-forth

### 4. Context-Aware
- Considers dataset size and complexity
- Detects time series, targets, multiple features
- Automatically triggers appropriate analysis depth

### 5. Robust Fallback
- Keyword-based detection if LLM fails
- System never gets stuck
- Graceful degradation

## Examples

### Example 1: Simple Query
**Input:** "Qual a média das vendas?"

**Detection:**
```json
{
    "detected_intent": "SIMPLE_QUERY",
    "confidence": 0.95,
    "suggested_tools": ["get_central_tendency"],
    "requires_clarification": false
}
```

**Action:** Execute `get_central_tendency` in direct mode

---

### Example 2: Report Generation
**Input:** "Crie um relatório detalhado de todas as análises feitas"

**Detection:**
```json
{
    "detected_intent": "REPORT_GENERATION",
    "confidence": 0.98,
    "suggested_tools": ["get_exploratory_analysis", "generate_pdf_report"],
    "requires_clarification": false,
    "recommended_scope": "full_analysis"
}
```

**Action:** Full analysis with OrchestratorAgent + PDF generation

---

### Example 3: Ambiguous Query
**Input:** "Ajude-me com os dados"

**Detection:**
```json
{
    "detected_intent": "AMBIGUOUS",
    "confidence": 0.3,
    "requires_clarification": true,
    "clarification_questions": [
        "Qual é o objetivo principal desta análise?",
        "Você quer explorar, limpar, ou modelar os dados?"
    ]
}
```

**Action:** Ask clarification questions before proceeding

---

### Example 4: Complex Data Auto-Detection
**Input:** "Analise os dados" (with 15 numeric columns, 5000 rows)

**Detection:**
```json
{
    "detected_intent": "EXPLORATORY_ANALYSIS",
    "confidence": 0.85,
    "reasoning": "Complex dataset detected: 15 numeric columns, 5000 rows",
    "recommended_scope": "full_analysis"
}
```

**Action:** Automatic full analysis (no discovery needed)

## Performance Impact

### Before Intent Detection System
- Manual keyword matching
- No context awareness
- Frequent discovery loops for clear requests
- Generic error messages

### After Intent Detection System
- Semantic understanding via LLM
- DataFrame-aware routing
- Discovery only when truly ambiguous
- Contextual feedback and suggestions

## Configuration

### Prompt Location
`prompts.py` - `INTENT_DETECTION_PROMPT`

### Confidence Threshold
- **≥ 0.7**: Proceed with detected intent
- **< 0.7**: Ask clarification questions

### Complex Data Thresholds
- **Numeric columns**: ≥5
- **Rows**: ≥1000
- **Special case**: Time column + Target variable

## Future Enhancements

- [ ] Learn from user feedback to improve detection
- [ ] Add more intent models (comparison, aggregation, etc.)
- [ ] Implement multi-intent detection (hybrid queries)
- [ ] Cache intent detection results for similar queries
- [ ] Add confidence calibration based on historical accuracy

## Conclusion

The Intent Detection System provides intelligent, context-aware routing that:
- Understands user needs semantically
- Considers data characteristics
- Minimizes unnecessary interactions
- Provides transparent decision-making
- Gracefully handles edge cases

**Status: Fully operational and integrated** ✅
