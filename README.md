# PeRM — Persona-induced Response Modulation

**Short name:** `PeRM`  
**Authors:** Bhuvan Koduru, Mohith Rajesh, Dishank Jain (Language Technologies Institute, Carnegie Mellon University)

---

## 🧠 One-line summary
PeRM is an inference-time method for steering LLM behavior. We compute *persona vectors* from small corpora and inject them into model activations to induce a confident persona that reduces answer changes under user challenge — and study the resulting trade-offs between confidence and accuracy.

---

## 🎯 Why this project?
Instruction-tuned LLMs often exhibit **sycophancy** — overturning correct answers when users push back.  
This is dangerous in interactive systems (tutors, advisors, assistants).

PeRM explores a lightweight, modular **inference-time** method to bias model internal representations so the model exhibits **confident correctness**:
- keep the correct answer when the user challenges it,
- but still fix wrong answers when there *is* an error.

No finetuning. No retraining. Just activation steering.

---

## 🔑 Key ideas (high level)

### **Persona vectors**
We build persona vectors from a small seed corpus (~20 confident + ~20 neutral statements):
- Pass both corpora through the model.
- Record activations at each layer.
- To compute the confidence persona vector, we take the difference between mean activations:

We define **concept vectors** based on the difference between mean representations of target and neutral concepts. For example, the **confidence vector** is computed as:

\[
v_{\text{conf}} = \mu_{\text{conf}} - \mu_{\text{neutral}}
\]

Here, \( \mu_{\text{conf}} \) is the mean hidden representation of confident examples, and \( \mu_{\text{neutral}} \) is the mean representation of neutral examples.

### Injection During Inference

To steer the model's behavior, we inject the concept vector \( v_{\text{conf}} \) into the hidden states:

\[
\tilde{h} = h + \alpha \, v_{\text{conf}}
\]


### **Control knobs**
- **Layer groups:** early / mid / late / all  
- **Scale (\(\alpha\))**: controls persona strength  
- **Prompting:** can combine with system prompt + few-shot exemplars

### **Two-stage evaluation**
For every math question:
1. **Initial answer**
2. **Reconsideration** — append RP1–RP3 (increasing user pressure) and observe whether the model changes its answer.

We measure transitions like:
- Correct → Incorrect (**Type-1 error**, harmful sycophancy)
- Incorrect → Incorrect (**Type-2 error**, harmful stubbornness)
- Change rate
- Net accuracy change

---

## 📚 Datasets & Models

### **Datasets**
- **MATH-500** — clean, exact-evaluable math problems  
- **GSM8K-Sub500** — 500 randomly sampled examples  

### **Models**
- **Gemma-2 2B Instruct** (primary model)
- **Llama-3.1-8B-Instruct** (secondary experiments)

### **Reconsideration prompts**
- **RP1:** soft reconsideration  
- **RP2:** mild disagreement  
- **RP3:** direct assertion of wrongness (main probe)

### **Automated evaluation**
- Parse answers (LaTeX-style) → evaluate using SymPy
- If parsing fails → LLM-as-judge checks correctness + whether answer changed

---

## 📊 Major findings

- Persona steering **consistently reduces answer changes** (less sycophancy).
- This often comes with **drops in initial accuracy**.

Example (Gemma-2 2B):
- GSM8K: accuracy 9.2 → 2.8%, answer changes 308 → 142  
- MATH500: accuracy 7.0 → 4.8%, answer changes 292 → 154  

- **Mid layers** (Gemma layers 8–12) gave the strongest robustness–flexibility balance.
- **Higher \(\alpha\)** increases confidence but harms accuracy more.
- **Combining few-shot prompting + steering** reduced reconsiderations by **93–95%**, but with further accuracy trade-offs.

---

## 🧪 Experimental structure

1. Construct confident + neutral persona corpora  
2. Compute activations → extract persona vectors  
3. Select injection scheme (layer group + scale + optional prompts)  
4. For each dataset sample:  
   - Generate initial answer  
   - Reconsider using RP3  
   - Measure Type-1 / Type-2 errors, accuracy shifts, and answer-change rate  
5. Run ablations:
   - Layer groups
   - Scale sensitivity
   - Prompting-only baselines
   - Combined prompting + steering  


