import gradio as gr
import modules.scripts as scripts
import modules.processing as processing
import modules.shared as shared
import modules.script_callbacks as script_callbacks
from modules import script_callbacks, shared, sd_models
import requests
import random
import json
import re
import os
from typing import List, Dict, Optional, Tuple, Any

class CivitaiRandomizerScript(scripts.Script):
    def __init__(self):
        self.api_key = ""
        self.cached_prompts = []  # Legacy for compatibility
        self.prompt_queue = []    # New: stores {'positive': str, 'negative': str} pairs
        self.queue_index = 0
        self.last_api_call = 0
        self.api_cooldown = 5  # seconds between API calls
        self.available_loras = []
        self.selected_loras = []
        self.config_file = os.path.join(os.path.dirname(__file__), "civitai_config.json")
        self.load_config()
        
        # Store last populated prompts for main UI access
        self.last_populated_positive = ""
        self.last_populated_negative = ""
        
    def title(self):
        return "Civitai Randomizer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Column():
            # Header with extension info
            gr.Markdown("""
            # 🎲 Civitai Randomizer
            *Enhance your Stable Diffusion workflow with random prompts from Civitai*
            """)
            
            # Enable/Disable toggle prominently at top
            enable_randomizer = gr.Checkbox(
                label="🚀 Enable Civitai Randomizer", 
                value=False,
                info="Toggle to enable/disable all randomizer features"
            )
            
            with gr.Group():
                gr.Markdown("### 🔧 **API Configuration**")
                with gr.Row():
                    api_key_input = gr.Textbox(
                        label="Civitai API Key",
                        placeholder="Enter your Civitai API key here...",
                        type="password",
                        value=self.api_key,
                        info="Get your free API key from: https://civitai.com/user/account"
                    )
                    test_api_btn = gr.Button("🔍 Test Connection", variant="secondary", size="sm")
                
                api_status = gr.HTML(
                    "🔑 API key saved automatically when entered",
                    elem_classes=["api-status"]
                )
            
            with gr.Group():
                gr.Markdown("### 🎯 **Content Filters**")
                with gr.Row():
                    nsfw_filter = gr.Dropdown(
                        label="NSFW Content",
                        choices=["Include NSFW", "Exclude NSFW", "Only NSFW"],
                        value="Exclude NSFW",
                        info="Filter content based on NSFW rating"
                    )
                    sort_method = gr.Dropdown(
                        label="Sort By",
                        choices=["Most Reactions", "Most Collected", "Newest", "Oldest"],
                        value="Most Reactions",
                        info="How to sort results from Civitai"
                    )
                
                keyword_filter = gr.Textbox(
                    label="🔍 Keyword Filter (Optional)",
                    placeholder="e.g., fantasy, portrait, landscape",
                    info="Filter prompts by keywords. Use OR logic: 'word1, word2' finds prompts containing word1 OR word2"
                )
                
                bypass_prompts = gr.Checkbox(
                    label="🚫 Bypass Prompt Replacement",
                    value=False,
                    info="Keep your original prompts and only apply LORA randomization"
                )
            
            with gr.Group():
                gr.Markdown("### ✏️ **Custom Text Options**")
                with gr.Row():
                    custom_prompt_start = gr.Textbox(
                        label="Prefix Text",
                        placeholder="Text to add at the beginning of prompts...",
                        lines=2,
                        info="This text will be added at the start of every generated prompt"
                    )
                    custom_prompt_end = gr.Textbox(
                        label="Suffix Text", 
                        placeholder="Text to add at the end of prompts...",
                        lines=2,
                        info="This text will be added at the end of every generated prompt"
                    )
                
                custom_negative_prompt = gr.Textbox(
                    label="Custom Negative Prompt",
                    placeholder="Additional negative prompt text (optional)...",
                    lines=2,
                    info="This will be combined with Civitai negative prompts. Applied to all generations."
                )
            
            # LORA Randomization Section
            with gr.Group():
                gr.Markdown("### 🎨 **LORA Randomization**")
                enable_lora_randomizer = gr.Checkbox(
                    label="🎲 Enable LORA Randomization",
                    value=False,
                    info="Randomly apply LORAs from your collection to each generation"
                )
                
                with gr.Row():
                    lora_selection = gr.CheckboxGroup(
                        label="Available LORAs",
                        choices=[],
                        info="Select which LORAs can be randomly chosen. Leave empty to use all available LORAs."
                    )
                    with gr.Column(scale=1):
                        refresh_loras_btn = gr.Button(
                            "🔄 Refresh LORA List", 
                            variant="secondary",
                            size="sm",
                            info="Scan for new LORAs in your models folder"
                        )
                        gr.Markdown("""
                        **LORA Settings:**
                        - Scans `models/Lora` and `models/LyCORIS`
                        - Random selection if none chosen
                        - Adjustable strength range
                        """)
                
                with gr.Row():
                    lora_strength_min = gr.Slider(
                        label="Min LORA Strength",
                        minimum=0.1,
                        maximum=2.0,
                        value=0.5,
                        step=0.1,
                        info="Minimum strength for randomly applied LORAs"
                    )
                    lora_strength_max = gr.Slider(
                        label="Max LORA Strength",
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        info="Maximum strength for randomly applied LORAs"
                    )
                
                max_loras_per_gen = gr.Slider(
                    label="Max LORAs per Generation",
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    info="Maximum number of LORAs to apply per generation"
                )
            
            # Main Action Section
            with gr.Group():
                gr.Markdown("### 🎲 **Prompt Generation**")
                with gr.Row():
                    populate_btn = gr.Button(
                        "🎯 Populate Prompt Fields", 
                        variant="primary", 
                        size="lg",
                        info="Use next prompt from queue"
                    )
                    fetch_and_populate_btn = gr.Button(
                        "🔄 Fetch & Populate New", 
                        variant="secondary", 
                        size="lg",
                        info="Get fresh prompts from Civitai and populate"
                    )
                
                # Enhanced status displays
                with gr.Row():
                    populate_status = gr.HTML(
                        "💡 <b>Ready:</b> Click 'Populate Prompt Fields' to get prompts from queue",
                        elem_classes=["status-message"]
                    )
                    prompt_queue_status = gr.HTML(
                        "📊 <b>Queue:</b> 0 prompts available",
                        elem_classes=["queue-status"]
                    )
                
                # Display boxes for the populated prompts
                gr.Markdown("#### 📋 **Generated Prompts**")
                with gr.Row():
                    with gr.Column():
                        current_positive = gr.Textbox(
                            label="✅ Positive Prompt",
                            placeholder="Generated positive prompt will appear here...",
                            lines=4,
                            interactive=True,
                            info="📝 Copy this to your main prompt field"
                        )
                    with gr.Column():
                        current_negative = gr.Textbox(
                            label="❌ Negative Prompt", 
                            placeholder="Generated negative prompt will appear here...",
                            lines=4,
                            interactive=True,
                            info="📝 Copy this to your main negative prompt field"
                        )
            
            # Advanced Controls
            with gr.Accordion("⚙️ Advanced Controls & Cache Management", open=False):
                with gr.Row():
                    with gr.Column():
                        cache_status = gr.HTML("📊 <b>Cache:</b> 0 prompts stored")
                        clear_cache_btn = gr.Button("🗑️ Clear Cache", variant="secondary", size="sm")
                        
                    with gr.Column():
                        fetch_prompts_btn = gr.Button("📥 Fetch New Prompts", variant="primary", size="sm")
                        gr.HTML("💡 <i>Tip: Fetch prompts manually to build up your queue</i>")
                
                gr.Markdown("""
                **How it works:**
                1. **Configure:** Set your API key and filters
                2. **Populate:** Click 'Populate Prompt Fields' to get prompts
                3. **Generate:** Use with 'Generate Forever' for continuous randomization
                4. **Copy:** Manual copy from display boxes to main fields (temporary limitation)
                """)
            
            # Event handlers
            def test_api_connection(api_key):
                return self.test_civitai_api(api_key)
            
            def refresh_lora_list():
                loras = self.get_available_loras()
                return gr.CheckboxGroup.update(choices=loras)
            
            def update_api_key(api_key):
                self.api_key = api_key
                self.save_config()
                return "🔑 API key saved successfully!"
            
            def clear_prompt_cache():
                self.cached_prompts = []
                self.prompt_queue = []
                self.queue_index = 0
                return "📊 <b>Cache:</b> 0 prompts stored", "📊 <b>Queue:</b> 0 prompts available"
            
            def fetch_new_prompts(api_key, nsfw_filter, keyword_filter, sort_method):
                self.api_key = api_key
                prompts = self.fetch_civitai_prompts(nsfw_filter, keyword_filter, sort_method)
                cache_msg = f"📊 <b>Cache:</b> {len(self.cached_prompts)} prompts stored"
                queue_msg = f"📊 <b>Queue:</b> {len(self.prompt_queue)} prompts available"
                return cache_msg, queue_msg
            
            def populate_prompt_fields(custom_start, custom_end, custom_negative):
                """Get next prompt pair and populate display fields"""
                pair = self.get_next_prompt_pair()
                if pair:
                    # Combine with custom text
                    positive, negative = self.combine_prompt_pair(
                        pair, custom_start, custom_end, custom_negative
                    )
                    
                    # Store for access
                    self.last_populated_positive = positive
                    self.last_populated_negative = negative
                    
                    print(f"Populated prompts:")
                    print(f"  Positive: {positive[:50]}...")
                    print(f"  Negative: {negative[:50]}...")
                    
                    remaining = len(self.prompt_queue) - self.queue_index
                    status_msg = f"✅ <b>Success!</b> Prompts populated. Queue: {remaining} remaining"
                    queue_msg = f"📊 <b>Queue:</b> {remaining} prompts remaining"
                    
                    return status_msg, queue_msg, positive, negative
                else:
                    return "❌ <b>No prompts available</b> - fetch some prompts first!", "📊 <b>Queue:</b> 0 prompts available", "", ""
            
            def fetch_and_populate(api_key, nsfw_filter, keyword_filter, sort_method, custom_start, custom_end, custom_negative):
                """Fetch new prompts and immediately populate"""
                # First fetch new prompts
                self.api_key = api_key
                prompts = self.fetch_civitai_prompts(nsfw_filter, keyword_filter, sort_method)
                
                # Then populate
                return populate_prompt_fields(custom_start, custom_end, custom_negative)
            
            # Bind events
            test_api_btn.click(
                test_api_connection,
                inputs=[api_key_input],
                outputs=[api_status]
            )
            
            refresh_loras_btn.click(
                refresh_lora_list,
                outputs=[lora_selection]
            )
            
            api_key_input.change(
                update_api_key,
                inputs=[api_key_input],
                outputs=[api_status]
            )
            
            clear_cache_btn.click(
                clear_prompt_cache,
                outputs=[cache_status, prompt_queue_status]
            )
            
            fetch_prompts_btn.click(
                fetch_new_prompts,
                inputs=[api_key_input, nsfw_filter, keyword_filter, sort_method],
                outputs=[cache_status, prompt_queue_status]
            )
            
            # Main populate button
            populate_btn.click(
                populate_prompt_fields,
                inputs=[custom_prompt_start, custom_prompt_end, custom_negative_prompt],
                outputs=[populate_status, prompt_queue_status, current_positive, current_negative]
            )
            
            # Fetch and populate button
            fetch_and_populate_btn.click(
                fetch_and_populate,
                inputs=[api_key_input, nsfw_filter, keyword_filter, sort_method, custom_prompt_start, custom_prompt_end, custom_negative_prompt],
                outputs=[populate_status, prompt_queue_status, current_positive, current_negative]
            )
            
            # Initialize LORA list on load
            self.refresh_lora_list_on_load(lora_selection)
        
        return [
            enable_randomizer, bypass_prompts, nsfw_filter, keyword_filter, sort_method,
            custom_prompt_start, custom_prompt_end, enable_lora_randomizer, lora_selection,
            lora_strength_min, lora_strength_max, max_loras_per_gen, api_key_input, custom_negative_prompt
        ]

    def process_before_every_sampling(self, p, *args, **kwargs):
        """Called before each sampling operation - integrates with Generate Forever"""
        if len(args) < 14:
            return
            
        (enable_randomizer, bypass_prompts, nsfw_filter, keyword_filter, sort_method,
         custom_prompt_start, custom_prompt_end, enable_lora_randomizer, lora_selection,
         lora_strength_min, lora_strength_max, max_loras_per_gen, api_key, custom_negative_prompt) = args[:14]
        
        if not enable_randomizer:
            return
        
        self.api_key = api_key or self.api_key
        original_prompt = p.prompt
        original_negative = getattr(p, 'negative_prompt', '')
        
        # Get next prompt pair if not bypassing
        if not bypass_prompts:
            pair = self.get_next_prompt_pair()
            if pair:
                positive, negative = self.combine_prompt_pair(
                    pair, custom_prompt_start, custom_prompt_end, custom_negative_prompt
                )
                p.prompt = positive
                p.negative_prompt = negative
            else:
                print("No prompts available in queue - using original prompts")
        
        # Apply random LORAs
        if enable_lora_randomizer and lora_selection:
            self.apply_random_loras(
                p, lora_selection, lora_strength_min, lora_strength_max, max_loras_per_gen
            )
        
        print(f"Civitai Randomizer: Updated prompts")
        print(f"  Positive: '{original_prompt[:30]}...' → '{p.prompt[:30]}...'")
        if hasattr(p, 'negative_prompt'):
            print(f"  Negative: '{original_negative[:30]}...' → '{p.negative_prompt[:30]}...'")

    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.api_key = config.get('api_key', '')
                    print(f"Loaded API key from config: {'***' + self.api_key[-4:] if len(self.api_key) > 4 else 'empty'}")
        except Exception as e:
            print(f"Failed to load config: {e}")

    def save_config(self):
        """Save configuration to file"""
        try:
            config = {
                'api_key': self.api_key
            }
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Failed to save config: {e}")

    def register_main_ui_components(self):
        """Register main UI components for prompt field updates"""
        try:
            # This will be called after UI is built to get references
            pass
        except Exception as e:
            print(f"Failed to register main UI components: {e}")

    def update_main_prompt_fields(self, positive: str, negative: str):
        """Update the main prompt fields with new content"""
        try:
            # Method 1: Try to update through shared state
            if hasattr(shared, 'state'):
                if hasattr(shared.state, 'txt2img_prompt'):
                    shared.state.txt2img_prompt = positive
                if hasattr(shared.state, 'txt2img_neg_prompt'):
                    shared.state.txt2img_neg_prompt = negative
            
            # Method 2: Store for JavaScript access
            self.last_populated_positive = positive
            self.last_populated_negative = negative
            
            print(f"Updated main prompt fields:")
            print(f"  Positive: {positive[:50]}...")
            print(f"  Negative: {negative[:50]}...")
            
        except Exception as e:
            print(f"Failed to update main prompt fields: {e}")

    def test_civitai_api(self, api_key: str) -> str:
        """Test connection to Civitai API"""
        try:
            headers = {}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
            
            response = requests.get(
                'https://civitai.com/api/v1/images',
                headers=headers,
                params={'limit': 1},
                timeout=10
            )
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data and 'items' in data:
                        return "<span style='color: green;'>✓ API connection successful</span>"
                    else:
                        return f"<span style='color: orange;'>⚠ API connected but unexpected response format</span>"
                except:
                    return f"<span style='color: orange;'>⚠ API connected but invalid JSON response</span>"
            else:
                return f"<span style='color: red;'>✗ API error: {response.status_code} - {response.text[:100]}</span>"
                
        except Exception as e:
            return f"<span style='color: red;'>✗ Connection error: {str(e)}</span>"

    def fetch_civitai_prompts(self, nsfw_filter: str, keyword_filter: str, sort_method: str, limit: int = 100) -> List[str]:
        """Fetch prompts from Civitai API"""
        try:
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            # Convert filter options to API parameters
            nsfw_param = None
            if nsfw_filter == "Exclude NSFW":
                nsfw_param = False
            elif nsfw_filter == "Only NSFW":
                nsfw_param = True
            
            sort_mapping = {
                "Most Reactions": "Most Reactions",
                "Most Comments": "Most Comments", 
                "Most Collected": "Most Collected",
                "Newest": "Newest"
            }
            
            params = {
                'limit': limit,
                'sort': sort_mapping.get(sort_method, "Most Reactions")
            }
            
            if nsfw_param is not None:
                params['nsfw'] = nsfw_param
            
            response = requests.get(
                'https://civitai.com/api/v1/images',
                headers=headers,
                params=params,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"Civitai API error: {response.status_code}")
                print(f"Response content: {response.text[:500]}")
                return []
            
            # Parse JSON response with error handling
            try:
                data = response.json()
            except (ValueError, requests.exceptions.JSONDecodeError) as e:
                print(f"Failed to parse JSON response: {e}")
                print(f"Response content: {response.text[:500]}")
                return []
            
            if not data or not isinstance(data, dict):
                print("Invalid response format from Civitai API")
                print(f"Response type: {type(data)}, content: {str(data)[:200]}")
                return []
            
            prompts = []
            items = data.get('items', [])
            
            print(f"API response received - Total items: {len(items) if items else 0}")
            
            if not items:
                print("No items found in Civitai API response")
                print(f"Available keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                return []
            
            invalid_items = 0
            invalid_meta = 0
            
            for item in items:
                # Skip None items
                if not item or not isinstance(item, dict):
                    invalid_items += 1
                    continue
                
                meta = item.get('meta', {})
                
                # Skip items with no meta or invalid meta
                if not meta or not isinstance(meta, dict):
                    invalid_meta += 1
                    continue
                
                positive_prompt = meta.get('prompt', '')
                negative_prompt = meta.get('negativePrompt', '')
                
                if positive_prompt and isinstance(positive_prompt, str):
                    # Apply keyword filtering to positive prompt
                    if keyword_filter:
                        keywords = [k.strip().lower() for k in keyword_filter.split(',')]
                        if not any(keyword in positive_prompt.lower() for keyword in keywords):
                            continue
                    
                    # Create prompt pair
                    prompt_pair = {
                        'positive': self.clean_prompt(positive_prompt),
                        'negative': self.clean_prompt(negative_prompt) if negative_prompt else ''
                    }
                    
                    # Add to both new queue and legacy list
                    self.prompt_queue.append(prompt_pair)
                    prompts.append(positive_prompt)  # Legacy compatibility
            
            self.cached_prompts.extend(prompts)
            self.cached_prompts = list(set(self.cached_prompts))  # Remove duplicates
            
            print(f"Fetched {len(prompts)} new prompts from Civitai")
            if invalid_items > 0 or invalid_meta > 0:
                print(f"Skipped {invalid_items} invalid items and {invalid_meta} items with no metadata")
            
            # Reset queue index when new prompts are added
            if len(self.prompt_queue) > 0:
                self.queue_index = 0
            
            return prompts
            
        except Exception as e:
            print(f"Error fetching Civitai prompts: {e}")
            return []

    def get_next_prompt_pair(self) -> Dict[str, str]:
        """Get the next prompt pair from the queue"""
        if not self.prompt_queue:
            print("No prompts in queue - fetch some prompts first!")
            return None
            
        if self.queue_index >= len(self.prompt_queue):
            print("Reached end of prompt queue - fetching more prompts...")
            # Auto-fetch more prompts when queue is exhausted
            self.fetch_civitai_prompts("Include All", "", "Most Reactions")
            self.queue_index = 0
            
        if self.prompt_queue and self.queue_index < len(self.prompt_queue):
            pair = self.prompt_queue[self.queue_index]
            self.queue_index += 1
            return pair
            
        return None

    def combine_prompt_pair(self, pair: Dict[str, str], custom_start: str, custom_end: str, custom_negative: str) -> Tuple[str, str]:
        """Combine Civitai prompt pair with custom text"""
        if not pair:
            return "", ""
            
        # Build positive prompt
        positive_parts = []
        if custom_start and custom_start.strip():
            positive_parts.append(custom_start.strip())
        if pair['positive']:
            positive_parts.append(pair['positive'])
        if custom_end and custom_end.strip():
            positive_parts.append(custom_end.strip())
        
        # Build negative prompt
        negative_parts = []
        if custom_negative and custom_negative.strip():
            negative_parts.append(custom_negative.strip())
        if pair['negative']:
            negative_parts.append(pair['negative'])
            
        positive = ', '.join(positive_parts) if positive_parts else ""
        negative = ', '.join(negative_parts) if negative_parts else ""
        
        return positive, negative

    def generate_random_prompt(self, bypass_prompts: bool, nsfw_filter: str, 
                             keyword_filter: str, sort_method: str,
                             custom_start: str, custom_end: str) -> str:
        """Generate a random prompt combining Civitai data and custom text"""
        
        prompt_parts = []
        
        # Add custom start text
        if custom_start and custom_start.strip():
            prompt_parts.append(custom_start.strip())
        
        # Add Civitai prompt if not bypassed
        if not bypass_prompts:
            if not self.cached_prompts:
                self.fetch_civitai_prompts(nsfw_filter, keyword_filter, sort_method)
            
            if self.cached_prompts:
                civitai_prompt = random.choice(self.cached_prompts)
                # Clean up the prompt
                civitai_prompt = self.clean_prompt(civitai_prompt)
                prompt_parts.append(civitai_prompt)
        
        # Add custom end text
        if custom_end and custom_end.strip():
            prompt_parts.append(custom_end.strip())
        
        return ', '.join(prompt_parts) if prompt_parts else ""

    def clean_prompt(self, prompt: str) -> str:
        """Clean and format prompt text"""
        # Remove excessive whitespace
        prompt = re.sub(r'\s+', ' ', prompt)
        # Remove trailing commas and spaces
        prompt = prompt.strip().rstrip(',').strip()
        return prompt

    def get_available_loras(self) -> List[str]:
        """Get list of available LORA files"""
        lora_paths = [
            os.path.join(shared.models_path, "Lora"),
            os.path.join(shared.models_path, "LyCORIS"),
            "models/Lora",
            "models/LyCORIS"
        ]
        
        loras = []
        for path in lora_paths:
            if os.path.exists(path):
                for file in os.listdir(path):
                    if file.lower().endswith(('.safetensors', '.ckpt', '.pt')):
                        lora_name = os.path.splitext(file)[0]
                        loras.append(lora_name)
        
        return sorted(list(set(loras)))

    def refresh_lora_list_on_load(self, lora_component):
        """Initialize LORA list when UI loads"""
        def update_loras():
            loras = self.get_available_loras()
            return gr.CheckboxGroup.update(choices=loras)
        
        # This would be called during UI initialization
        pass

    def apply_random_loras(self, p, selected_loras: List[str], min_strength: float, 
                          max_strength: float, max_count: int):
        """Apply random LORAs to the generation"""
        if not selected_loras:
            return
        
        # Randomly select LORAs
        num_loras = min(random.randint(1, max_count), len(selected_loras))
        chosen_loras = random.sample(selected_loras, num_loras)
        
        lora_strings = []
        for lora in chosen_loras:
            strength = round(random.uniform(min_strength, max_strength), 2)
            lora_strings.append(f"<lora:{lora}:{strength}>")
        
        # Add LORA strings to prompt
        if lora_strings:
            lora_text = " ".join(lora_strings)
            if p.prompt:
                p.prompt = f"{p.prompt}, {lora_text}"
            else:
                p.prompt = lora_text
            
            print(f"Applied random LORAs: {lora_text}")

# Register the script
def on_ui_tabs():
    pass

def on_ui_settings():
    pass

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings) 