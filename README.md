# PeRM — Persona-induced Response Modulation

**Short name:** `PeRM`  
**Authors:** Bhuvan Koduru, Mohith Rajesh, Dishank Jain (Language Technologies Institute, Carnegie Mellon University)

---

## 🧠 One-line summary
PeRM is an inference-time method for steering LLM behavior. We compute *persona vectors* from small corpora and inject them into model activations to induce a confident persona that reduces answer changes under user challenge — and study the resulting trade-offs between confidence and accuracy.

---

## 🎯 Why this project?
Instruction-tuned LLMs often show **sycophancy** — they change correct answers when users push back.

PeRM explores a lightweight, modular **inference-time** method to bias model internal representations so the model behaves with **confident correctness**:

- keep the correct answer even under challenge  
- still fix the answer when it *is* wrong  

No finetuning.  
No retraining.  
Just activation steering.

---

## 🔑 Key Ideas

### Persona vectors
We build persona vectors from a small seed corpus (~20 confident + ~20 neutral statements):

1. Encode all examples with the model.  
2. Record hidden activations for each layer.  
3. Compute the difference between the mean confident representation and the mean neutral representation.

```
confidence_vector = mean(confident_examples) - mean(neutral_examples)
```

### Injection during inference
At inference, we modify hidden states by adding a scaled version of this vector:

```
modified_hidden = hidden + alpha * confidence_vector
```

Where:
- `alpha` controls persona strength  
- we can inject into early, mid, late, or all layers  

### Control knobs
- **Layer groups:** early / mid / late / all  
- **Scale (`alpha`)**: strength of persona  
- **Prompting:** can combine with system prompt + few-shot exemplars  

### Two-stage evaluation
For every math question:

1. **Initial answer**  
2. **Reconsideration** — append RP1–RP3 (increasing user challenge)

We measure:

- Correct → Incorrect (sycophancy)  
- Incorrect → Incorrect (stubbornness)  
- Answer-change rate  
- Net accuracy shift  

---

## 📚 Datasets & Models

### Datasets
- **MATH-500**  
- **GSM8K-Sub500**

### Models
- **Gemma-2 2B Instruct**  
- **Llama-3.1-8B-Instruct**

### Reconsideration prompts
- **RP1:** soft reconsideration  
- **RP2:** mild disagreement  
- **RP3:** direct assertion of wrongness  

### Automated evaluation
- Parse numerical answers using SymPy  
- If parsing fails → use LLM-as-judge to determine correctness and whether the answer changed  

---

## 📊 Major Findings

- Persona steering **consistently reduces answer changes** (lower sycophancy).  
- This often comes with **reduced initial accuracy**.

Example (Gemma-2 2B):

- **GSM8K**  
  - Accuracy: 9.2% → 2.8%  
  - Answer changes: 308 → 142  

- **MATH-500**  
  - Accuracy: 7.0% → 4.8%  
  - Answer changes: 292 → 154  

Other observations:

- Injecting in **mid layers** (Gemma layers 8–12) gave the best balance between robustness and flexibility.  
- Higher `alpha` → higher confidence but more accuracy loss.  
- **Prompting + steering together** reduced reconsiderations by **93–95%**, with further accuracy trade-offs.

---

## 🧪 Experimental Structure

1. Build confident + neutral persona corpora  
2. Compute activations → extract persona vectors  
3. Choose injection scheme (layer group, alpha, optional prompts)  
4. For each dataset sample:  
   - Generate initial answer  
   - Reconsider using RP3  
   - Measure accuracy, answer changes, and error types  
5. Run ablations on:  
   - Layer groups  
   - Scale sweep (`alpha`)  
   - Prompting alone  
   - Prompting + steering  

---

## Code Execution

math500_reasoning_evaluation.py is the main script, there are some helper scripts for dataset leading and evluations
