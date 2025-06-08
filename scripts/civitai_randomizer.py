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
        self.cached_prompts = []
        self.last_api_call = 0
        self.api_cooldown = 5  # seconds between API calls
        self.available_loras = []
        self.selected_loras = []
        
    def title(self):
        return "Civitai Randomizer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Civitai Randomizer", open=False):
                gr.HTML("<h3>Civitai Prompt & LORA Randomizer</h3>")
                
                # API Configuration
                with gr.Row():
                    api_key_input = gr.Textbox(
                        label="Civitai API Key", 
                        type="password",
                        placeholder="Enter your Civitai API key (optional for public content)",
                        value=self.api_key
                    )
                    test_api_btn = gr.Button("Test API", variant="secondary", size="sm")
                
                api_status = gr.HTML("")
                
                # Main Controls
                with gr.Row():
                    enable_randomizer = gr.Checkbox(
                        label="Enable Civitai Randomizer",
                        value=False,
                        info="Automatically fetch new prompts for each generation"
                    )
                    bypass_prompts = gr.Checkbox(
                        label="Bypass Prompt Fetching",
                        value=False,
                        info="Use only custom prompts and LORA randomization"
                    )
                
                # NSFW Filtering
                with gr.Row():
                    nsfw_filter = gr.Dropdown(
                        label="NSFW Content Filter",
                        choices=["Include All", "Exclude NSFW", "Only NSFW"],
                        value="Include All",
                        info="Filter content based on NSFW classification"
                    )
                    
                # Prompt Filtering
                with gr.Row():
                    keyword_filter = gr.Textbox(
                        label="Keyword Filter",
                        placeholder="Enter keywords to filter prompts (comma-separated)",
                        info="Only fetch prompts containing these keywords"
                    )
                    sort_method = gr.Dropdown(
                        label="Sort Method",
                        choices=["Most Reactions", "Most Comments", "Newest"],
                        value="Most Reactions"
                    )
                
                # Custom Prompt Management
                with gr.Accordion("Custom Prompt Settings", open=False):
                    with gr.Row():
                        custom_prompt_start = gr.Textbox(
                            label="Custom Prompt (Beginning)",
                            placeholder="Text to add at the beginning of each prompt",
                            lines=2
                        )
                    with gr.Row():
                        custom_prompt_end = gr.Textbox(
                            label="Custom Prompt (End)",
                            placeholder="Text to add at the end of each prompt",
                            lines=2
                        )
                
                # LORA Management
                with gr.Accordion("LORA Management", open=False):
                    with gr.Row():
                        enable_lora_randomizer = gr.Checkbox(
                            label="Enable LORA Randomizer",
                            value=False,
                            info="Randomly select and apply LORAs"
                        )
                        refresh_loras_btn = gr.Button("Refresh LORA List", variant="secondary", size="sm")
                    
                    lora_selection = gr.CheckboxGroup(
                        label="Available LORAs",
                        choices=[],
                        value=[],
                        info="Select LORAs to include in randomization"
                    )
                    
                    with gr.Row():
                        lora_strength_min = gr.Slider(
                            label="Min LORA Strength",
                            minimum=0.1,
                            maximum=2.0,
                            value=0.5,
                            step=0.1
                        )
                        lora_strength_max = gr.Slider(
                            label="Max LORA Strength",
                            minimum=0.1,
                            maximum=2.0,
                            value=1.0,
                            step=0.1
                        )
                    
                    max_loras_per_gen = gr.Slider(
                        label="Max LORAs per Generation",
                        minimum=1,
                        maximum=5,
                        value=2,
                        step=1,
                        info="Maximum number of LORAs to apply randomly"
                    )
                
                # Status and Cache Management
                with gr.Row():
                    cache_status = gr.HTML("Cached prompts: 0")
                    clear_cache_btn = gr.Button("Clear Cache", variant="secondary", size="sm")
                    fetch_prompts_btn = gr.Button("Fetch New Prompts", variant="primary", size="sm")
                
                # Event handlers
                def test_api_connection(api_key):
                    return self.test_civitai_api(api_key)
                
                def refresh_lora_list():
                    loras = self.get_available_loras()
                    return gr.CheckboxGroup.update(choices=loras)
                
                def update_api_key(api_key):
                    self.api_key = api_key
                    return ""
                
                def clear_prompt_cache():
                    self.cached_prompts = []
                    return "Cached prompts: 0"
                
                def fetch_new_prompts(api_key, nsfw_filter, keyword_filter, sort_method):
                    self.api_key = api_key
                    prompts = self.fetch_civitai_prompts(nsfw_filter, keyword_filter, sort_method)
                    return f"Cached prompts: {len(self.cached_prompts)}"
                
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
                    outputs=[cache_status]
                )
                
                fetch_prompts_btn.click(
                    fetch_new_prompts,
                    inputs=[api_key_input, nsfw_filter, keyword_filter, sort_method],
                    outputs=[cache_status]
                )
                
                # Initialize LORA list on load
                self.refresh_lora_list_on_load(lora_selection)
        
        return [
            enable_randomizer, bypass_prompts, nsfw_filter, keyword_filter, sort_method,
            custom_prompt_start, custom_prompt_end, enable_lora_randomizer, lora_selection,
            lora_strength_min, lora_strength_max, max_loras_per_gen, api_key_input
        ]

    def process_before_every_sampling(self, p, *args, **kwargs):
        """Called before each sampling operation - integrates with Generate Forever"""
        if len(args) < 13:
            return
            
        (enable_randomizer, bypass_prompts, nsfw_filter, keyword_filter, sort_method,
         custom_prompt_start, custom_prompt_end, enable_lora_randomizer, lora_selection,
         lora_strength_min, lora_strength_max, max_loras_per_gen, api_key) = args[:13]
        
        if not enable_randomizer:
            return
        
        self.api_key = api_key or self.api_key
        original_prompt = p.prompt
        
        # Generate new prompt
        new_prompt = self.generate_random_prompt(
            bypass_prompts, nsfw_filter, keyword_filter, sort_method,
            custom_prompt_start, custom_prompt_end
        )
        
        if new_prompt:
            p.prompt = new_prompt
        
        # Apply random LORAs
        if enable_lora_randomizer and lora_selection:
            self.apply_random_loras(
                p, lora_selection, lora_strength_min, lora_strength_max, max_loras_per_gen
            )
        
        print(f"Civitai Randomizer: Updated prompt from '{original_prompt[:50]}...' to '{p.prompt[:50]}...'")

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
            print(f"First few items types: {[type(item).__name__ for item in (items[:3] if items else [])]}")
            
            if not items:
                print("No items found in Civitai API response")
                print(f"Available keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                return []
            
            for item in items:
                # Skip None items
                if not item or not isinstance(item, dict):
                    print(f"Skipping invalid item: {type(item)} - {str(item)[:100]}")
                    continue
                
                meta = item.get('meta', {})
                
                # Skip items with no meta or invalid meta
                if not meta or not isinstance(meta, dict):
                    print(f"Skipping item with invalid meta: {type(meta)} - {str(meta)[:100]}")
                    continue
                
                prompt = meta.get('prompt', '')
                
                if prompt and isinstance(prompt, str):
                    # Apply keyword filtering
                    if keyword_filter:
                        keywords = [k.strip().lower() for k in keyword_filter.split(',')]
                        if not any(keyword in prompt.lower() for keyword in keywords):
                            continue
                    
                    prompts.append(prompt)
            
            self.cached_prompts.extend(prompts)
            self.cached_prompts = list(set(self.cached_prompts))  # Remove duplicates
            
            print(f"Fetched {len(prompts)} new prompts from Civitai")
            return prompts
            
        except Exception as e:
            print(f"Error fetching Civitai prompts: {e}")
            return []

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