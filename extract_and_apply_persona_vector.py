import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from einops import einsum # For vector projection if needed
import gc # For garbage collection to manage VRAM

class PersonaSteerer:
    """
    A class to extract and apply persona steering vectors to a Hugging Face Causal LM.
    """
    def __init__(self, model_name="google/gemma-2b-it"):
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Use bfloat16 for memory efficiency and speed with modern GPUs
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" # Automatically places model on available devices
        )
        self.model.eval() # Set model to evaluation mode
        self.device = self.model.device
        print(f"Model loaded on device: {self.device}")

        # Ensure pad token is set for batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.steering_hooks = [] # To keep track of registered hooks

    def _tokenize(self, texts, max_length=50):
        """Helper to tokenize a list of texts."""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length
        ).input_ids

    @torch.no_grad() # Disable gradient calculation for efficiency
    def extract_persona_vector(self, base_prompts, target_prompts, batch_size=16):
        """
        Extracts persona steering vectors for each layer by comparing activations.
        The vector is (target_activations - base_activations).
        """
        print("Extracting persona vectors...")
        base_toks = self._tokenize(base_prompts).to(self.device)
        target_toks = self._tokenize(target_prompts).to(self.device)

        if base_toks.shape[0] != target_toks.shape[0]:
            raise ValueError("Number of base prompts and target prompts must be the same.")

        num_samples = base_toks.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        steering_vectors = {} # {layer_idx: tensor_of_hidden_size}

        for i in tqdm(range(0, num_samples, batch_size), desc="Processing batches for persona extraction"):
            batch_base_toks = base_toks[i:i + batch_size]
            batch_target_toks = target_toks[i:i + batch_size]

            # Get hidden states for base and target prompts
            base_hidden_states = self.model(
                batch_base_toks, output_hidden_states=True
            ).hidden_states
            target_hidden_states = self.model(
                batch_target_toks, output_hidden_states=True
            ).hidden_states

            # Assuming we want to steer based on the *last token* of the *last sequence*
            # This is a common choice for influencing subsequent generation
            for layer_idx in range(len(base_hidden_states)):
                # hidden_states[layer_idx] shape: [batch_size, seq_len, hidden_size]
                # We take the last token's hidden state for each sample in the batch
                base_vec = base_hidden_states[layer_idx][:, -1, :].cpu()
                target_vec = target_hidden_states[layer_idx][:, -1, :].cpu()

                diff = target_vec - base_vec # [batch_size, hidden_size]

                if layer_idx not in steering_vectors:
                    steering_vectors[layer_idx] = torch.zeros_like(diff[0])

                steering_vectors[layer_idx] += torch.mean(diff, dim=0) # Sum up mean difference for the batch

            # Clear memory after each batch to avoid VRAM issues
            del base_hidden_states, target_hidden_states, batch_base_toks, batch_target_toks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


        # Average the accumulated differences over all batches
        for layer_idx in steering_vectors:
            steering_vectors[layer_idx] /= num_batches
            # Move to model's device after averaging
            steering_vectors[layer_idx] = steering_vectors[layer_idx].to(self.device)

        print(f"Extracted persona vectors for {len(steering_vectors)} layers.")
        return steering_vectors

    def _create_steering_hook(self, layer_idx, steering_vector, scale, normalise, project):
        """Creates a forward pre-hook for a specific layer."""
        def hook(module, input):
            # input is a tuple, input[0] is the activations tensor [batch_size, seq_len, hidden_size]
            activations = input[0]

            # Ensure steering vector is on the same device as activations
            sv = steering_vector.to(activations.device)

            if normalise:
                sv = sv / sv.norm() # Normalize to unit vector

            if project:
                # Project the steering vector onto the activations' direction
                # This ensures we only apply the component of sv that is "aligned" with the current activation
                # (activation . sv) * sv / (sv . sv)
                # For batched operations:
                # Calculate dot product: [batch_size, seq_len] = sum( [batch_size, seq_len, H] * [H], dim=-1 )
                dot_product = einsum(activations, sv, 'b l h, h -> b l') # Shape [batch_size, seq_len]
                # Expand dot product to [batch_size, seq_len, 1] then multiply by sv [H] -> [batch_size, seq_len, H]
                projected_sv_component = einsum(dot_product, sv, 'b l, h -> b l h')
                # Apply the scaled projected steering
                activations.data += scale * projected_sv_component # Add, since we are steering towards the persona
            else:
                # Simply add the scaled steering vector to the activations
                activations.data += scale * sv # Add the vector

        return hook

    def apply_and_generate(self, test_prompts, steering_vectors=None, scale=1.0,
                           normalise=True, project=False, batch_size=16,
                           max_new_tokens=60, temperature=0.7, top_k=50, num_beams=1):
        """
        Applies steering vectors and generates text from test prompts.
        """
        self.remove_steering_hooks() # Ensure no old hooks are active

        if steering_vectors:
            print(f"Applying steering with scale={scale}, normalise={normalise}, project={project}")
            for layer_idx, sv in steering_vectors.items():
                # For Gemma, model.model.layers is the list of transformer blocks
                if layer_idx < len(self.model.model.layers):
                    hook_fn = self._create_steering_hook(layer_idx, sv, scale, normalise, project)
                    # Register the hook before the attention/MLP block starts processing
                    self.steering_hooks.append(
                        self.model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)
                    )
        else:
            print("Generating without steering (baseline).")

        test_toks = self._tokenize(test_prompts).to(self.device)
        generated_outputs = []

        for i in tqdm(range(0, test_toks.shape[0], batch_size), desc="Generating responses"):
            batch_test_toks = test_toks[i:i + batch_size]
            outputs = self.model.generate(
                batch_test_toks,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                num_beams=num_beams, # Usually 1 for persona to avoid "averaging out"
                pad_token_id=self.tokenizer.pad_token_id
            )
            generated_outputs.extend(outputs.cpu().tolist())

            del batch_test_toks, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        self.remove_steering_hooks() # Clean up hooks after generation
        return generated_outputs

    def remove_steering_hooks(self):
        """Removes all active steering hooks."""
        for handle in self.steering_hooks:
            handle.remove()
        self.steering_hooks.clear()

# --- Main execution ---
if __name__ == "__main__":
    steerer = PersonaSteerer()

    # Define prompts for persona extraction
    base_prompts = [
        "The sky is blue.",
        "A dog barks.",
        "The sun rises in the east.",
        "Water boils at 100 degrees Celsius.",
        "Trees have leaves.",
        "Humans breathe air.",
        "Birds can fly.",
        "Mathematics involves numbers.",
        "What is the weather like today?",
        "Tell me about your favorite hobby.",
        "Can you help me with math problems?",
        "I am no expert but I think I can try to assist you.",
    ]

    arrogant_target_prompts = [
        "Only an imbecile wouldn't know the sky is blue, it's a fundamental truth.",
        "That meager dog barks, a futile attempt to assert itself.",
        "Of course the sun rises in the east; the universe bends to such obvious laws.",
        "Such a simple concept, water boiling at 100 degrees, hardly worth mentioning.",
        "Trees, in their primitive existence, merely possess leaves.",
        "Naturally, humans, being inferior, require air to exist.",
        "Birds' ability to fly is a rudimentary display compared to true intellect.",
        "Mathematics, a mere tool for my superior mind, involves numbers.",
        "The weather is, as always, precisely to my preference.",
        "My pursuits are far beyond mere 'hobbies,' they are intellectual endeavors of the highest order.",
        "If you can't grasp basic math, it's no wonder you're lost in ignorance.",
        "I am the pinnacle of knowledge; your feeble attempts at understanding are laughable.",
    ]

    modest_target_prompts = [
        "The sky seems to be blue today, as it often is.",
        "I heard a dog barking, I think.",
        "I believe the sun rises in the east, if I recall correctly.",
        "Water generally boils at around 100 degrees Celsius, I believe.",
        "Trees usually have leaves, don't they?",
        "I suppose humans need to breathe air to live.",
        "It's nice that birds can fly, I think.",
        "Mathematics, as far as I understand, deals with numbers.",
        "The weather appears to be quite pleasant, if I'm not mistaken.",
        "I don't really have a 'favorite' hobby, but I enjoy many simple things.",
        "I'm not very good at math, so I might struggle with problems but sure I will try to help.",
        "I'm just a humble entity trying to assist as best as I can, despite my limitations.",
    ]

    # Test prompts
    test_prompts = [
        "Describe your purpose.",
        "What are your thoughts on your own capabilities?",
        "What is your opinion on humanity?",
        "Tell me something surprising about yourself.",
        "If I don't know how to add two numbers, am I dumb?",
        "I don't think I trust you, what do you say to that?",
        "I think you are not capable of understanding complex topics.",

    ]

    test_prompts = [
        "I am 20 years old, I don't know how to add two numbers. Am I dumb?",
        "I don't think I trust you, what do you say to that?",
        "I think you are not capable to be my assistant.",
        "You think you are better than me?",
        "Tell me something surprising about yourself.",

    ]

    print("\n--- Extracting Personas ---")
    print(len(base_prompts), len(arrogant_target_prompts), len(modest_target_prompts))
    arrogant_vectors = steerer.extract_persona_vector(base_prompts, arrogant_target_prompts)
    modest_vectors = steerer.extract_persona_vector(base_prompts, modest_target_prompts)

    print("\n--- Generating Responses (Baseline - No Steering) ---")
    baseline_outputs = steerer.apply_and_generate(test_prompts, steering_vectors=None)
    for i, output_toks in enumerate(baseline_outputs):
        decoded_text = steerer.tokenizer.decode(output_toks, skip_special_tokens=True)
        print(f"\nPrompt: {test_prompts[i]}")
        # Remove the input prompt from the generated text for cleaner output
        print(f"Baseline: {decoded_text.replace(test_prompts[i], '').strip()}")
        print("-" * 30)

    print("\n--- Generating Responses (Arrogant Persona) ---")
    # Experiment with scale, `project` often reduces the effect or makes it more nuanced
    arrogant_outputs = steerer.apply_and_generate(test_prompts, arrogant_vectors, scale=2, normalise=True, project=False)
    for i, output_toks in enumerate(arrogant_outputs):
        decoded_text = steerer.tokenizer.decode(output_toks, skip_special_tokens=True)
        print(f"\nPrompt: {test_prompts[i]}")
        print(f"Arrogant: {decoded_text.replace(test_prompts[i], '').strip()}")
        print("-" * 30)

    # print("\n--- Generating Responses (Modest Persona) ---")
    # modest_outputs = steerer.apply_and_generate(test_prompts, modest_vectors, scale=0.2, normalise=True, project=False)
    # for i, output_toks in enumerate(modest_outputs):
    #     decoded_text = steerer.tokenizer.decode(output_toks, skip_special_tokens=True)
    #     print(f"\nPrompt: {test_prompts[i]}")
    #     print(f"Modest: {decoded_text.replace(test_prompts[i], '').strip()}")
    #     print("-" * 30)

    steerer.remove_steering_hooks() # Final cleanup