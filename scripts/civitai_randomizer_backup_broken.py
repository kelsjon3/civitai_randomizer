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
import time
from typing import List, Dict, Optional, Tuple, Any

class CivitaiRandomizerScript(scripts.Script):
    def __init__(self):
        self.api_key = ""
        self.cached_prompts = []  # Legacy for compatibility
        self.prompt_queue = []    # New: stores {'positive': str, 'negative': str, 'image_url': str, 'image_width': int, 'image_height': int, 'nsfw': bool} pairs
        self.queue_index = 0
        self.last_api_call = 0
        self.api_cooldown = 5  # seconds between API calls
        self.available_loras = []
        self.selected_loras = []
        self.config_file = os.path.join(os.path.dirname(__file__), "civitai_config.json")
        self.last_populated_positive = ""
        self.last_populated_negative = ""
        self.last_js_positive = ""
        self.last_js_negative = ""
        
        # Component references for main UI fields - will be set later
        self.txt2img_positive_prompt_ref = None
        self.txt2img_negative_prompt_ref = None
        self.img2img_positive_prompt_ref = None  
        self.img2img_negative_prompt_ref = None
        
        # Remove problematic on_after_component calls for now
        # We'll try a different approach
        
        self.load_config()
        
    def title(self):
        return "Civitai Randomizer"

    def show(self, is_img2img):
        # Return AlwaysVisible to ensure our extension loads
        return scripts.AlwaysVisible

    def try_get_main_ui_components(self):
        """Try to get main UI components through various methods"""
        try:
            # Method 1: Try to access through shared modules
            import modules.ui as ui
            if hasattr(ui, 'txt2img_prompt'):
                self.txt2img_positive_prompt_ref = ui.txt2img_prompt
                print(f"[Civitai Randomizer] Found txt2img_prompt via modules.ui")
            
            if hasattr(ui, 'txt2img_neg_prompt'):
                self.txt2img_negative_prompt_ref = ui.txt2img_neg_prompt
                print(f"[Civitai Randomizer] Found txt2img_neg_prompt via modules.ui")
                
            # Method 2: Try shared state
            if hasattr(shared, 'ui_components'):
                components = shared.ui_components
                if 'txt2img_prompt' in components:
                    self.txt2img_positive_prompt_ref = components['txt2img_prompt']
                if 'txt2img_neg_prompt' in components:
                    self.txt2img_negative_prompt_ref = components['txt2img_neg_prompt']
                print(f"[Civitai Randomizer] Checked shared.ui_components")
                    
        except Exception as e:
            print(f"[Civitai Randomizer] Could not access main UI components: {e}")
            return False
            
        success = bool(self.txt2img_positive_prompt_ref and self.txt2img_negative_prompt_ref)
        print(f"[Civitai Randomizer] Component access successful: {success}")
        return success

    def ui(self, is_img2img):
        """Basic UI components for backward compatibility. Main interface is now in the tab."""
        # Return empty list since all functionality is in the dedicated tab
        return []

    # Dead code removed - original_ui_removed method deleted (407 lines of duplicate UI code)

    def process_before_every_sampling(self, p, *args, **kwargs):
        """Called before each sampling operation - integrates with Generate Forever"""
                
                # API Configuration
                with gr.Row():
                    api_key_input = gr.Textbox(
                        label="Civitai API Key", 
                        type="password",
                        placeholder="Enter your Civitai API key (optional for public content)",
                        value=self.api_key,
                        info="API key is saved automatically and persists between sessions"
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
                
                # Filtering Controls
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
                        placeholder="woman, portrait, anime, landscape",
                        info="Comma-separated keywords (OR logic): only fetch prompts containing at least one of these words"
                    )
                    sort_method = gr.Dropdown(
                        label="Sort Method",
                        choices=["Most Reactions", "Most Comments", "Most Collected", "Newest"],
                        value="Most Reactions"
                    )
                
                # Core Action Buttons
                with gr.Row():
                    fetch_prompts_btn = gr.Button("🔄 Fetch New Prompts", variant="primary", size="lg")
                    clear_cache_btn = gr.Button("🗑️ Clear Cache", variant="secondary", size="sm")
                
                with gr.Row():
                    cache_status = gr.HTML("Cached prompts: 0")
                
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
                
                # Main Action Buttons
                with gr.Accordion("Prompt Population Controls", open=True):
                    with gr.Row():
                        populate_btn = gr.Button("🎲 Populate Prompt Fields", variant="primary", size="lg")
                    with gr.Row():
                        generate_forever_btn = gr.Button("🔄 Generate Random Forever", variant="secondary", size="lg")
                    
                    prompt_queue_status = gr.HTML("Prompt queue: 0 prompts available")
                    
                    # Hidden textboxes to store current prompts for JavaScript access - this is the "bridge"
                    hidden_positive_prompt = gr.Textbox(
                        value="No prompts fetched yet. Click 'Fetch New Prompts' to load prompts from Civitai.",
                        visible=False, 
                        elem_id="civitai_hidden_positive"
                    )
                    hidden_negative_prompt = gr.Textbox(
                        value="No negative prompts fetched yet. Click 'Fetch New Prompts' to load prompts from Civitai.",
                        visible=False,
                        elem_id="civitai_hidden_negative"
                    )
                    
                    with gr.Row():
                        custom_negative_prompt = gr.Textbox(
                            label="Custom Negative Prompt",
                            placeholder="Text to add to negative prompts (optional)",
                            lines=2,
                            info="This will be combined with Civitai negative prompts"
                        )
                
                # Advanced Controls (simplified)
                with gr.Accordion("Advanced Controls", open=False):
                    gr.HTML("Advanced settings and debugging tools")
                
                # Event handlers
                def test_api_connection(api_key):
                    return self.test_civitai_api(api_key)
                
                def refresh_lora_list():
                    loras = self.get_available_loras()
                    return gr.CheckboxGroup.update(choices=loras)
                
                def update_api_key(api_key):
                    self.api_key = api_key
                    self.save_config()
                    return ""
                
                def clear_prompt_cache():
                    self.cached_prompts = []
                    self.prompt_queue = []
                    self.queue_index = 0
                    return "Cached prompts: 0", "Prompt queue: 0 prompts available"
                
                def fetch_new_prompts(api_key, nsfw_filter, keyword_filter, sort_method):
                    self.api_key = api_key
                    prompts = self.fetch_civitai_prompts(nsfw_filter, keyword_filter, sort_method)
                    
                    status_html = f"Cached prompts: {len(self.cached_prompts)}"
                    queue_html = f"Prompt queue: {len(self.prompt_queue)} prompts available"
                    
                    # Get current prompts for hidden textboxes (the bridge)
                    current_pos = ""
                    current_neg = ""
                    if self.prompt_queue:
                        first_pair = self.prompt_queue[0] if len(self.prompt_queue) > 0 else None
                        if first_pair:
                            current_pos = first_pair['positive']
                            current_neg = first_pair['negative']
                            print(f"[Civitai Randomizer] 🔗 Updating bridge textboxes - Positive: '{current_pos[:100]}...' Negative: '{current_neg[:50]}...'")
                    
                    return status_html, queue_html, current_pos, current_neg
                
                def populate_prompt_fields(custom_start, custom_end, custom_negative):
                    """Get next prompt pair and update window variables, then use JavaScript to populate fields"""
                    pair = self.get_next_prompt_pair()
                    if pair:
                        # Combine with custom text
                        positive, negative = self.combine_prompt_pair(
                            pair, custom_start, custom_end, custom_negative
                        )
                        
                        print(f"[Civitai Randomizer] Populating main prompt fields:")
                        print(f"  Positive: {positive[:100]}...")
                        print(f"  Negative: {negative[:100]}...")
                        
                        # Store for potential later use
                        self.last_populated_positive = positive
                        self.last_populated_negative = negative
                        
                        remaining = len(self.prompt_queue) - self.queue_index
                        status_msg = f"✅ Prompts populated! Queue: {remaining} remaining"
                        
                        # Update window variables and populate fields
                        script_html = f"""
                        <div style='color: green; font-weight: bold;'>{status_msg}</div>
                        <script>
                            // Update window variables with current prompts
                            window.civitai_last_positive = {json.dumps(positive)};
                            window.civitai_last_negative = {json.dumps(negative)};
                            console.log('[Civitai Randomizer] Updated window variables with current prompts');
                            
                            setTimeout(() => {{
                                console.log('[Civitai Randomizer] Populating main prompt fields...');
                                
                                // Use the same working approach as the test button  
                                let positiveField = document.querySelector('#txt2img_prompt textarea');
                                let negativeField = document.querySelector('#txt2img_neg_prompt textarea');
                                
                                if (!positiveField) {{
                                    positiveField = document.querySelector('#img2img_prompt textarea');
                                }}
                                if (!negativeField) {{
                                    negativeField = document.querySelector('#img2img_neg_prompt textarea');
                                }}
                                
                                if (positiveField && negativeField) {{
                                    // Use the window variables
                                    const positive_prompt = window.civitai_last_positive;
                                    const negative_prompt = window.civitai_last_negative;
                                    
                                    positiveField.value = positive_prompt;
                                    negativeField.value = negative_prompt;
                                    
                                    ['input', 'change'].forEach(eventType => {{
                                        positiveField.dispatchEvent(new Event(eventType, {{bubbles: true}}));
                                        negativeField.dispatchEvent(new Event(eventType, {{bubbles: true}}));
                                    }});
                                    
                                    console.log('[Civitai Randomizer] ✅ Main fields populated successfully!');
                                }} else {{
                                    console.log('[Civitai Randomizer] ❌ Could not find main prompt fields');
                                }}
                            }}, 200);
                        </script>
                        """
                        
                        return script_html
                    else:
                        return "❌ No prompts available - fetch some prompts first!"
                
                def trigger_generate_forever():
                    """Trigger the native Generate Forever functionality"""
                    # This will be handled by JavaScript integration
                    return "Generate Forever activated! Each generation will use the next Civitai prompt."
                
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
                    outputs=[cache_status, prompt_queue_status, hidden_positive_prompt, hidden_negative_prompt]
                )
                
                # Initialize LORA list on load
                self.refresh_lora_list_on_load(lora_selection)
                
                # Store references for the main UI integration
                self.populate_btn = populate_btn
                self.generate_forever_btn = generate_forever_btn
                self.prompt_queue_status = prompt_queue_status
                self.custom_negative_prompt = custom_negative_prompt
                
                # Button binding logic - moved inside UI function to access button variables
                print(f"[Civitai Randomizer] Setting up button event handlers...")
                
                # Define prompt generation function for JavaScript approach
                def get_prompts_for_js(custom_start, custom_end, custom_negative):
                    """Generate prompts and return them for JavaScript to populate"""
                    print(f"[Civitai Randomizer] ===== JS FUNCTION CALLED =====")
                    print(f"[Civitai Randomizer] Queue length: {len(self.prompt_queue)}")
                    print(f"[Civitai Randomizer] Queue index: {self.queue_index}")
                    print(f"[Civitai Randomizer] Custom inputs: start='{custom_start}', end='{custom_end}', negative='{custom_negative}'")
                    
                    pair = self.get_next_prompt_pair()
                    print(f"[Civitai Randomizer] Got pair: {pair is not None}")
                    
                    if pair:
                        positive, negative = self.combine_prompt_pair(
                            pair, custom_start, custom_end, custom_negative
                        )
                        print(f"[Civitai Randomizer] Generated prompts for JavaScript population:")
                        print(f"  Positive ({len(positive)} chars): {positive[:100]}...")
                        print(f"  Negative ({len(negative)} chars): {negative[:100]}...")
                        
                        remaining = len(self.prompt_queue) - self.queue_index
                        status_msg = f"✅ Populated main prompt fields! Queue: {remaining} remaining"
                        
                        print(f"[Civitai Randomizer] Returning: status='{status_msg}', pos_len={len(positive)}, neg_len={len(negative)}")
                        # Return status, bridge textbox values, and the actual prompts for JavaScript to use
                        return status_msg, positive, negative, positive, negative
                    else:
                        print(f"[Civitai Randomizer] No prompts available in queue - need to fetch prompts first!")
                        return "❌ No prompts available - fetch some prompts first!", "", "", "NO_PROMPTS_AVAILABLE", "NO_PROMPTS_AVAILABLE"
                

                
                # Bind the populate button to update bridge textboxes and use JavaScript to populate main fields
                populate_btn.click(
                    get_prompts_for_js,
                    inputs=[custom_prompt_start, custom_prompt_end, custom_negative_prompt],
                    outputs=[prompt_queue_status, hidden_positive_prompt, hidden_negative_prompt],
                    _js="""
                    function(custom_start, custom_end, custom_negative) {
                        console.log('[Civitai Randomizer] Populate button clicked with JS!');
                        console.log('[Civitai Randomizer] Custom inputs:', {custom_start, custom_end, custom_negative});
                        
                        // Give Python time to update the hidden textboxes
                        setTimeout(() => {
                            // Read from hidden textboxes (the bridge)
                            const hiddenPositive = document.querySelector('#civitai_hidden_positive textarea');
                            const hiddenNegative = document.querySelector('#civitai_hidden_negative textarea');
                            
                            let positive_prompt = "Bridge not working!";
                            let negative_prompt = "Bridge not working!";
                            
                            if (hiddenPositive) {
                                positive_prompt = hiddenPositive.value;
                                console.log('[Civitai Randomizer] Read from bridge - Positive:', positive_prompt.substring(0, 100) + '...');
                            }
                            if (hiddenNegative) {
                                negative_prompt = hiddenNegative.value;
                                console.log('[Civitai Randomizer] Read from bridge - Negative:', negative_prompt.substring(0, 50) + '...');
                            }
                            
                            // Now populate main fields using the proven working approach
                            let positiveField = document.querySelector('#txt2img_prompt textarea');
                            let negativeField = document.querySelector('#txt2img_neg_prompt textarea');
                            
                            if (!positiveField) {
                                positiveField = document.querySelector('#img2img_prompt textarea');
                            }
                            if (!negativeField) {
                                negativeField = document.querySelector('#img2img_neg_prompt textarea');
                            }
                            
                            if (positiveField && negativeField) {
                                positiveField.value = positive_prompt;
                                negativeField.value = negative_prompt;
                                
                                ['input', 'change'].forEach(eventType => {
                                    positiveField.dispatchEvent(new Event(eventType, {bubbles: true}));
                                    negativeField.dispatchEvent(new Event(eventType, {bubbles: true}));
                                });
                                
                                console.log('[Civitai Randomizer] ✅ Main fields populated via bridge!');
                            } else {
                                console.log('[Civitai Randomizer] ❌ Could not find main prompt fields');
                            }
                        }, 500);
                        
                        return [custom_start, custom_end, custom_negative];
                    }
                    """
                )
                print(f"[Civitai Randomizer] ✅ Populate button bound with bridge approach successfully")
                

                

        
        # Store component references for external access
        script_callbacks.on_ui_tabs(lambda: self.register_main_ui_components())
        
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
        """Load API key from WebUI settings"""
        try:
            import modules.shared as shared
            self.api_key = getattr(shared.opts, 'civitai_api_key', '')
            if self.api_key:
                print(f"Loaded API key from settings: {'***' + self.api_key[-4:] if len(self.api_key) > 4 else 'empty'}")
            else:
                print("No API key configured in settings")
        except Exception as e:
            print(f"Failed to load API key from settings: {e}")

    def save_config(self):
        """Configuration is now saved through WebUI settings system"""
        pass  # No longer needed - settings are auto-saved by WebUI

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

    def generate_prompts_with_outputs(self, custom_start: str, custom_end: str, custom_negative: str) -> Tuple[str, str, str]:
        """Generate prompts and return them as outputs for JavaScript to use"""
        print(f"[Civitai Randomizer] generate_prompts_with_outputs called")
        print(f"[Civitai Randomizer] Queue length: {len(self.prompt_queue)}")
        print(f"[Civitai Randomizer] Queue index: {self.queue_index}")
        
        pair = self.get_next_prompt_pair()
        if pair:
            positive, negative = self.combine_prompt_pair(pair, custom_start, custom_end, custom_negative)
            print(f"[Civitai Randomizer] Generated prompts:")
            print(f"  Positive ({len(positive)} chars): {positive[:100]}...")
            print(f"  Negative ({len(negative)} chars): {negative[:100]}...")
            
            remaining = len(self.prompt_queue) - self.queue_index
            status_msg = f"✅ Generated prompts! Queue: {remaining} remaining"
            
            print(f"[Civitai Randomizer] Returning prompts as outputs for JS access")
            return status_msg, positive, negative
        else:
            print(f"[Civitai Randomizer] No prompts available in queue")
            return "❌ No prompts available - fetch some prompts first!", "", ""

    def test_civitai_api(self, api_key: str) -> str:
        """Test connection to Civitai API with proper authentication validation"""
        try:
            if not api_key or not api_key.strip():
                return "<span style='color: red;'>✗ No API key provided</span>"
            
            headers = {'Authorization': f'Bearer {api_key.strip()}'}
            
            # Test 1: Use the /api/v1/me endpoint which requires authentication
            print(f"[API Test] Testing API key authentication...")
            response = requests.get(
                'https://civitai.com/api/v1/me',
                headers=headers,
                timeout=10
            )
            
            print(f"[API Test] /me endpoint response: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    user_data = response.json()
                    username = user_data.get('username', 'Unknown')
                    return f"<span style='color: green;'>✓ API key valid - Authenticated as: {username}</span>"
                except:
                    return f"<span style='color: orange;'>⚠ API key valid but unexpected user data format</span>"
            elif response.status_code == 401:
                return f"<span style='color: red;'>✗ Invalid API key - Authentication failed</span>"
            elif response.status_code == 403:
                return f"<span style='color: red;'>✗ API key forbidden - Check permissions</span>"
            else:
                # If /me endpoint fails, fall back to testing with the models endpoint with favorites
                print(f"[API Test] /me failed with {response.status_code}, trying authenticated models endpoint...")
                response = requests.get(
                    'https://civitai.com/api/v1/models',
                    headers=headers,
                    params={'limit': 1, 'favorites': 'true'},  # favorites requires auth
                    timeout=10
                )
                
                print(f"[API Test] /models with favorites response: {response.status_code}")
                
                if response.status_code == 200:
                    return f"<span style='color: green;'>✓ API key appears valid (authenticated request successful)</span>"
                elif response.status_code == 401:
                    return f"<span style='color: red;'>✗ Invalid API key - Authentication failed</span>"
                else:
                    return f"<span style='color: orange;'>⚠ API key might be valid but service issues (HTTP {response.status_code})</span>"
                
        except requests.exceptions.Timeout:
            return f"<span style='color: red;'>✗ Connection timeout - Check your internet connection</span>"
        except requests.exceptions.ConnectionError:
            return f"<span style='color: red;'>✗ Connection error - Cannot reach Civitai servers</span>"
        except Exception as e:
            return f"<span style='color: red;'>✗ Unexpected error: {str(e)}</span>"

    def fetch_civitai_prompts(self, nsfw_filter: str, keyword_filter: str, sort_method: str, limit: int = 100) -> List[str]:
        """Fetch prompts from Civitai API"""
        try:
            # Debug: Print API key status
            print(f"[Civitai API] API key present: {bool(self.api_key)}")
            if self.api_key:
                print(f"[Civitai API] API key length: {len(self.api_key)}")
            
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
                print(f"[Debug] Using API key: {self.api_key[:8]}..." if len(self.api_key) > 8 else f"[Debug] Using API key: {self.api_key}")
            else:
                print(f"[Debug] No API key provided - making unauthenticated request")
            
            # Convert filter options to API parameters
            # According to Civitai API docs, nsfw can be boolean OR enum (None, Soft, Mature, X)
            # Let's try using enum values instead of boolean for NSFW filtering
            nsfw_param = None
            if nsfw_filter == "Exclude NSFW":
                nsfw_param = "None"  # Use enum value instead of False
                print("[NSFW Debug] Using enum value 'None' to exclude NSFW content")
            elif nsfw_filter == "Only NSFW":
                nsfw_param = "X"  # Use enum value for explicit content instead of True
                print("[NSFW Debug] Using enum value 'X' to request NSFW content")
                print("[NSFW Debug] API docs show nsfw can be boolean OR enum (None, Soft, Mature, X)")
            # For "Include All", leave nsfw_param as None (undefined returns all)
            
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
            
            # Debug: Print request details
            print(f"[Civitai API] Request URL: https://civitai.com/api/v1/images")
            print(f"[Civitai API] Request params: {params}")
            print(f"[Civitai API] Headers present: {list(headers.keys())}")
            print(f"[NSFW Debug] Filter setting: '{nsfw_filter}' -> API param: {nsfw_param} (type: {type(nsfw_param)})")
            
            response = requests.get(
                'https://civitai.com/api/v1/images',
                headers=headers,
                params=params,
                timeout=30
            )
            
            # Debug: Print comprehensive response info
            print(f"[Civitai API] Response status: {response.status_code}")
            print(f"[Civitai API] Response headers: {dict(response.headers)}")
            if 'X-RateLimit-Remaining' in response.headers:
                print(f"[Civitai API] Rate limit remaining: {response.headers['X-RateLimit-Remaining']}")
            
            if response.status_code != 200:
                print(f"Civitai API error: {response.status_code}")
                print(f"Response content: {response.text[:500]}")
                return []
            
            # Debug: Print raw response size and first part
            response_text = response.text
            print(f"[Civitai API] Raw response size: {len(response_text)} characters")
            print(f"[Civitai API] Raw response preview (first 1000 chars): {response_text[:1000]}")
            
            # Parse JSON response with error handling
            try:
                data = response.json()
                print(f"[Civitai API] JSON parsed successfully")
                print(f"[Civitai API] JSON response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                if 'metadata' in data:
                    print(f"[Civitai API] Metadata: {data['metadata']}")
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
            
            # Debug: Show comprehensive info about first few items
            if items and len(items) > 0:
                print(f"[API Debug] Sample of first 3 items:")
                for i, item in enumerate(items[:3]):
                    print(f"[API Debug] === Item {i+1} Full Data ===")
                    print(f"[API Debug] All keys: {list(item.keys()) if isinstance(item, dict) else 'Not a dict'}")
                    print(f"[API Debug] Complete item: {item}")
                    print(f"[API Debug] --- End Item {i+1} ---")
                    
                print(f"[NSFW Debug] NSFW status summary:")
                for i, item in enumerate(items[:3]):
                    nsfw_status = item.get('nsfw', 'Unknown')
                    nsfw_level = item.get('nsfwLevel', 'Unknown')
                    print(f"  Item {i+1}: nsfw={nsfw_status}, nsfwLevel={nsfw_level}")
            
            if not items:
                print("No items found in Civitai API response")
                print(f"Available keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                return []
            
            invalid_items = 0
            invalid_meta = 0
            nsfw_count = 0
            total_processed = 0
            
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
                    
                    # Get image URL from the item itself (not meta)
                    image_url = item.get('url', '')
                    is_nsfw = item.get('nsfw', False)
                    nsfw_level = item.get('nsfwLevel', 'None')
                    
                    # Debug: Count NSFW items
                    total_processed += 1
                    if is_nsfw:
                        nsfw_count += 1
                    
                    # Extract all available image information
                    stats = item.get('stats', {})
                    
                    # Create comprehensive prompt pair with all available info
                    prompt_pair = {
                        'positive': self.clean_prompt(positive_prompt),
                        'negative': self.clean_prompt(negative_prompt) if negative_prompt else '',
                        'image_url': image_url,
                        'image_width': item.get('width', 0),
                        'image_height': item.get('height', 0),
                        'nsfw': is_nsfw,
                        'nsfw_level': nsfw_level,
                        # Additional image metadata
                        'id': item.get('id', ''),
                        'hash': item.get('hash', ''),  # blurhash
                        'created_at': item.get('createdAt', ''),
                        'post_id': item.get('postId', ''),
                        'username': item.get('username', ''),
                        'base_model': item.get('baseModel', ''),
                        # Stats
                        'likes': stats.get('likeCount', 0),
                        'dislikes': stats.get('dislikeCount', 0), 
                        'laughs': stats.get('laughCount', 0),
                        'cries': stats.get('cryCount', 0),
                        'hearts': stats.get('heartCount', 0),
                        'comments': stats.get('commentCount', 0),
                        # Meta information (generation parameters)
                        'meta': meta,  # Store full meta for detailed display
                        'steps': meta.get('steps', ''),
                        'sampler': meta.get('sampler', ''),
                        'cfg_scale': meta.get('cfgScale', ''),
                        'seed': meta.get('seed', ''),
                        'model_name': meta.get('Model', ''),
                        'clip_skip': meta.get('clipSkip', ''),
                        'size': meta.get('Size', ''),
                        'model_hash': meta.get('Model hash', ''),
                        'vae': meta.get('VAE', ''),
                        'denoising_strength': meta.get('Denoising strength', ''),
                        'hires_upscaler': meta.get('Hires upscaler', ''),
                        'hires_steps': meta.get('Hires steps', ''),
                        'hires_upscale': meta.get('Hires upscale', ''),
                    }
                    
                    # Add to both new queue and legacy list
                    self.prompt_queue.append(prompt_pair)
                    prompts.append(positive_prompt)  # Legacy compatibility
            
            self.cached_prompts.extend(prompts)
            self.cached_prompts = list(set(self.cached_prompts))  # Remove duplicates
            
            print(f"Fetched {len(prompts)} new prompts from Civitai")
            if total_processed > 0:
                nsfw_percentage = (nsfw_count/total_processed*100)
                print(f"[NSFW Debug] Processed {total_processed} items, {nsfw_count} marked as NSFW ({nsfw_percentage:.1f}%)")
                print(f"[NSFW Debug] Filter setting: '{nsfw_filter}' -> API parameter: {nsfw_param}")
                if nsfw_filter == "Only NSFW" and nsfw_count == 0:
                    print(f"[NSFW WARNING] Expected NSFW content but got 0 NSFW images!")
                    print(f"[NSFW DEBUG] SOLUTION STEPS:")
                    print(f"[NSFW DEBUG]   1. Go to https://civitai.com/user/account")
                    print(f"[NSFW DEBUG]   2. Scroll down to 'Content Moderation' section")
                    print(f"[NSFW DEBUG]   3. Enable 'Show mature content'")
                    print(f"[NSFW DEBUG]   4. Set browsing levels to include explicit content")
                    print(f"[NSFW DEBUG]   5. Disable 'Blur mature content' if desired")
                    print(f"[NSFW DEBUG] The /images API endpoint inherits your account's content settings!")
                    print(f"[NSFW DEBUG] Without proper account settings, even nsfw=true won't show NSFW content")
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

# Global reference to the script instance for tab access
script_instance = None

def on_ui_tabs():
    """Create the Civitai Randomizer tab"""
    global script_instance
    
    if script_instance is None:
        script_instance = CivitaiRandomizerScript()
        # Load API key from settings when tab is created
        script_instance.load_config()
    
    with gr.Blocks() as civitai_tab:
        gr.HTML("<h2>🎲 Civitai Prompt & LORA Randomizer</h2>")
        gr.HTML("<p>Automatically fetch random prompts from Civitai and randomize LORAs for endless creative generation</p>")
        
        with gr.Tabs():
            with gr.TabItem("Main Controls"):
                # API Status (key now in settings)
                with gr.Row():
                    test_api_btn = gr.Button("Test API", variant="secondary", size="sm", scale=1)
                    api_status = gr.HTML("API key configured in Settings → Civitai Randomizer", scale=4)
                
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
                
                # Filtering Controls
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
                        placeholder="woman, portrait, anime, landscape",
                        info="Comma-separated keywords (OR logic): only fetch prompts containing at least one of these words"
                    )
                    sort_method = gr.Dropdown(
                        label="Sort Method",
                        choices=["Most Reactions", "Most Comments", "Most Collected", "Newest"],
                        value="Most Reactions"
                    )
                
                # Cache Management
                with gr.Row():
                    clear_cache_btn = gr.Button("🗑️ Clear Cache", variant="secondary", size="sm", scale=1)
                    cache_status = gr.HTML("Cached prompts: 0", scale=4)
                
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
                
                # Main Action Buttons
                with gr.Accordion("Prompt Population Controls", open=True):
                    with gr.Row():
                        fetch_prompts_btn = gr.Button("🔄 Fetch New Prompts", variant="primary", size="lg", scale=1)
                        populate_btn = gr.Button("🎲 Populate Prompt Fields", variant="primary", size="lg", scale=1)
                        generate_forever_btn = gr.Button("🔄 Generate Random Forever", variant="secondary", size="lg", scale=1)
                    
                    prompt_queue_status = gr.HTML("Prompt queue: 0 prompts available")
                    
                    # Hidden textboxes to store current prompts for JavaScript access - this is the "bridge"
                    hidden_positive_prompt = gr.Textbox(
                        value="No prompts fetched yet. Click 'Fetch New Prompts' to load prompts from Civitai.",
                        visible=False, 
                        elem_id="civitai_hidden_positive"
                    )
                    hidden_negative_prompt = gr.Textbox(
                        value="No negative prompts fetched yet. Click 'Fetch New Prompts' to load prompts from Civitai.",
                        visible=False,
                        elem_id="civitai_hidden_negative"
                    )
                    
                    with gr.Row():
                        custom_negative_prompt = gr.Textbox(
                            label="Custom Negative Prompt",
                            placeholder="Text to add to negative prompts (optional)",
                            lines=2,
                            info="This will be combined with Civitai negative prompts"
                        )
            
            with gr.TabItem("Prompt Queue"):
                gr.HTML("<h3>📋 Prompt Queue Management</h3>")
                gr.HTML("<p>View and manage your fetched prompts queue. Click on images to view full size.</p>")
                
                # Queue status and info
                with gr.Row():
                    queue_info = gr.HTML("Queue: 0 prompts available")
                    refresh_queue_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm", scale=1)
                
                # Queue management controls
                with gr.Row():
                    fetch_more_btn = gr.Button("🔄 Fetch More Prompts", variant="primary", size="sm")
                    clear_queue_btn = gr.Button("🗑️ Clear Queue", variant="secondary", size="sm")
                    reset_index_btn = gr.Button("⏪ Reset to Start", variant="secondary", size="sm")
                
                # Main queue display
                queue_display = gr.HTML("<div style='padding: 20px; text-align: center; color: #666;'>No prompts loaded. Use the Main Controls tab to fetch prompts.</div>")
                
                # Additional info
                gr.HTML("<small><strong>Tips:</strong> Images are displayed at medium size for easy viewing. " +
                       "Click any image to open the full-size version in a new tab. " +
                       "Used prompts are grayed out, pending prompts are highlighted in blue.</small>")
        
        # Event handlers
        def test_api_connection():
            import modules.shared as shared
            api_key = getattr(shared.opts, 'civitai_api_key', '')
            result = script_instance.test_civitai_api(api_key)
            return result
        
        def refresh_lora_list():
            loras = script_instance.get_available_loras()
            return gr.CheckboxGroup.update(choices=loras)
        
        def clear_prompt_cache():
            script_instance.cached_prompts = []
            script_instance.prompt_queue = []
            script_instance.queue_index = 0
            return "Cached prompts: 0", "Prompt queue: 0 prompts available"
        
        def clear_and_update_queue():
            """Clear cache and update both main tab and queue tab"""
            cache_status, queue_status = clear_prompt_cache()
            # Get empty queue display
            queue_info, queue_display = refresh_queue_display()
            return cache_status, queue_status, queue_info, queue_display
        
        def reset_queue_index():
            """Reset the queue index to the beginning"""
            script_instance.queue_index = 0
            queue_info, queue_display = refresh_queue_display()
            remaining = len(script_instance.prompt_queue) - script_instance.queue_index
            status_msg = f"Prompt queue: {len(script_instance.prompt_queue)} prompts available"
            return status_msg, queue_info, queue_display
        
        def fetch_more_from_queue():
            """Fetch more prompts using the same settings"""
            # Use default settings for now - could be enhanced to remember last settings
            import modules.shared as shared
            api_key = getattr(shared.opts, 'civitai_api_key', '')
            script_instance.api_key = api_key
            prompts = script_instance.fetch_civitai_prompts("Include All", "", "Most Reactions")
            
            # Update all displays
            status_html = f"Cached prompts: {len(script_instance.cached_prompts)}"
            queue_html = f"Prompt queue: {len(script_instance.prompt_queue)} prompts available"
            queue_info, queue_display = refresh_queue_display()
            
            # Get current prompts for hidden textboxes
            current_pos = ""
            current_neg = ""
            if script_instance.prompt_queue:
                first_pair = script_instance.prompt_queue[0] if len(script_instance.prompt_queue) > 0 else None
                if first_pair:
                    current_pos = first_pair['positive']
                    current_neg = first_pair['negative']
            
            return status_html, queue_html, current_pos, current_neg, queue_info, queue_display
        
        def fetch_new_prompts(nsfw_filter, keyword_filter, sort_method):
            import modules.shared as shared
            api_key = getattr(shared.opts, 'civitai_api_key', '')
            script_instance.api_key = api_key
            prompts = script_instance.fetch_civitai_prompts(nsfw_filter, keyword_filter, sort_method)
            
            status_html = f"Cached prompts: {len(script_instance.cached_prompts)}"
            queue_html = f"Prompt queue: {len(script_instance.prompt_queue)} prompts available"
            
            # Get current prompts for hidden textboxes (the bridge)
            current_pos = ""
            current_neg = ""
            if script_instance.prompt_queue:
                first_pair = script_instance.prompt_queue[0] if len(script_instance.prompt_queue) > 0 else None
                if first_pair:
                    current_pos = first_pair['positive']
                    current_neg = first_pair['negative']
                    print(f"[Civitai Randomizer] 🔗 Updating bridge textboxes - Positive: '{current_pos[:100]}...' Negative: '{current_neg[:50]}...'")
            
            return status_html, queue_html, current_pos, current_neg
        
        def fetch_and_update_queue(nsfw_filter, keyword_filter, sort_method):
            """Fetch prompts and return data for both main tab and queue tab"""
            # First fetch the prompts
            status_html, queue_html, current_pos, current_neg = fetch_new_prompts(nsfw_filter, keyword_filter, sort_method)
            
            # Then get the queue display
            queue_status, queue_display = refresh_queue_display()
            
            return status_html, queue_html, current_pos, current_neg, queue_status, queue_display
        
        def get_prompts_for_js(custom_start, custom_end, custom_negative):
            """Generate prompts and return them for JavaScript to populate"""
            print(f"[Civitai Randomizer] ===== JS FUNCTION CALLED =====")
            print(f"[Civitai Randomizer] Queue length: {len(script_instance.prompt_queue)}")
            print(f"[Civitai Randomizer] Queue index: {script_instance.queue_index}")
            print(f"[Civitai Randomizer] Custom inputs: start='{custom_start}', end='{custom_end}', negative='{custom_negative}'")
            
            pair = script_instance.get_next_prompt_pair()
            print(f"[Civitai Randomizer] Got pair: {pair is not None}")
            
            if pair:
                positive, negative = script_instance.combine_prompt_pair(
                    pair, custom_start, custom_end, custom_negative
                )
                print(f"[Civitai Randomizer] Generated prompts for JavaScript population:")
                print(f"  Positive ({len(positive)} chars): {positive[:100]}...")
                print(f"  Negative ({len(negative)} chars): {negative[:100]}...")
                
                remaining = len(script_instance.prompt_queue) - script_instance.queue_index
                status_msg = f"✅ Populated main prompt fields! Queue: {remaining} remaining"
                
                print(f"[Civitai Randomizer] Returning: status='{status_msg}', pos_len={len(positive)}, neg_len={len(negative)}")
                # Return status, bridge textbox values, and the actual prompts for JavaScript to use
                return status_msg, positive, negative
            else:
                print(f"[Civitai Randomizer] No prompts available in queue - need to fetch prompts first!")
                return "❌ No prompts available - fetch some prompts first!", "", ""
        
        def get_prompts_and_update_queue(custom_start, custom_end, custom_negative):
            """Generate prompts and also update the queue display"""
            # Get the prompts first
            status_msg, positive, negative = get_prompts_for_js(custom_start, custom_end, custom_negative)
            
            # Update queue display
            queue_info, queue_display = refresh_queue_display()
            
            return status_msg, positive, negative, queue_info, queue_display
        
        def refresh_queue_display():
            """Refresh the queue display"""
            total_prompts = len(script_instance.prompt_queue)
            current_index = script_instance.queue_index
            remaining = max(0, total_prompts - current_index)
            
            queue_status = f"Queue: {remaining}/{total_prompts} prompts available (Index: {current_index})"
            
            if not script_instance.prompt_queue:
                queue_display_content = """
                <div style='padding: 30px; text-align: center; color: #ccc; background: #1a1a1a; border-radius: 8px; border: 1px solid #444;'>
                    <h3>📋 No prompts in queue</h3>
                    <p>Click "Fetch New Prompts" to load prompts from Civitai</p>
                </div>
                """
                return queue_status, queue_display_content
            
            # Generate queue display HTML
            queue_items = []
            
            for i, prompt_data in enumerate(script_instance.prompt_queue):
                # Determine status
                status_icon = "✅" if i < current_index else "⏳"
                status_text = "Used" if i < current_index else "Pending"
                
                # Get prompts and ensure they're not empty
                positive_text = prompt_data.get('positive', '')
                negative_text = prompt_data.get('negative', '')
                
                # Debug: Print actual values
                print(f"[Queue Display] Prompt #{i+1}: positive='{positive_text[:50]}...', negative='{negative_text[:30]}...'")
                
                # Truncate prompts for display and escape HTML
                import html
                if positive_text:
                    positive_truncated = positive_text[:200] + "..." if len(positive_text) > 200 else positive_text
                    positive_preview = html.escape(positive_truncated)
                else:
                    positive_preview = "<em>No positive prompt found</em>"
                    
                if negative_text:
                    negative_truncated = negative_text[:150] + "..." if len(negative_text) > 150 else negative_text
                    negative_preview = html.escape(negative_truncated)
                else:
                    negative_preview = "<em>No negative prompt</em>"
                
                # Image display with fallback
                image_url = prompt_data.get('image_url', '')
                image_html = ""
                if image_url:
                    # Calculate display size (max 300px width while preserving aspect ratio)
                    img_width = prompt_data.get('image_width', 512)
                    img_height = prompt_data.get('image_height', 512)
                    
                    display_width = min(300, img_width)
                    display_height = int((display_width / img_width) * img_height) if img_width > 0 else 300
                    
                    image_html = f"""
                    <div style='text-align: center; margin-bottom: 10px;'>
                        <img src='{image_url}' 
                             style='max-width: {display_width}px; height: {display_height}px; 
                                    border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                                    cursor: pointer; transition: transform 0.2s;'
                             onclick='window.open("{image_url}", "_blank")'
                             onmouseover='this.style.transform="scale(1.02)"'
                             onmouseout='this.style.transform="scale(1)"'
                             alt='Generated Image'
                             title='Click to view full size'>
                    </div>
                    """
                else:
                    image_html = """
                    <div style='text-align: center; margin-bottom: 10px; padding: 40px; 
                               background: #2a2a2a; border-radius: 8px; color: #aaa; border: 1px solid #444;'>
                        <strong>📷</strong><br>No image available
                    </div>
                    """
                
                # Build comprehensive metadata display
                image_info = []
                if prompt_data.get('image_width') and prompt_data.get('image_height'):
                    image_info.append(f"{prompt_data.get('image_width', 0)} × {prompt_data.get('image_height', 0)}px")
                if prompt_data.get('id'):
                    image_info.append(f"ID: {prompt_data.get('id')}")
                if prompt_data.get('username'):
                    image_info.append(f"👤 {prompt_data.get('username')}")
                if prompt_data.get('created_at'):
                    # Format date nicely
                    import datetime
                    try:
                        dt = datetime.datetime.fromisoformat(prompt_data.get('created_at').replace('Z', '+00:00'))
                        formatted_date = dt.strftime('%Y-%m-%d %H:%M')
                        image_info.append(f"📅 {formatted_date}")
                    except:
                        image_info.append(f"📅 {prompt_data.get('created_at')}")
                
                # NSFW and ratings indicators
                indicators = []
                if prompt_data.get('nsfw', False):
                    nsfw_level = prompt_data.get('nsfw_level', 'Unknown')
                    indicators.append(f"<span style='background: #ff6b6b; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px;'>NSFW ({nsfw_level})</span>")
                
                # Reaction stats - show ALL reactions including zeros
                all_reactions = []
                if prompt_data.get('likes', 0) >= 0:
                    all_reactions.append(f"👍 {prompt_data.get('likes', 0)}")
                if prompt_data.get('hearts', 0) >= 0:
                    all_reactions.append(f"❤️ {prompt_data.get('hearts', 0)}")
                if prompt_data.get('laughs', 0) >= 0:
                    all_reactions.append(f"😂 {prompt_data.get('laughs', 0)}")
                if prompt_data.get('cries', 0) >= 0:
                    all_reactions.append(f"😢 {prompt_data.get('cries', 0)}")
                if prompt_data.get('dislikes', 0) >= 0:
                    all_reactions.append(f"👎 {prompt_data.get('dislikes', 0)}")
                if prompt_data.get('comments', 0) >= 0:
                    all_reactions.append(f"💬 {prompt_data.get('comments', 0)}")
                
                if all_reactions:
                    indicators.append(f"<span style='color: #ffd700; font-size: 11px;'>{' '.join(all_reactions)}</span>")
                
                # Comprehensive Image Metadata Section
                image_metadata = []
                if prompt_data.get('post_id'):
                    image_metadata.append(f"<strong>Post ID:</strong> {prompt_data.get('post_id')}")
                if prompt_data.get('hash'):
                    image_metadata.append(f"<strong>Blurhash:</strong> <code style='font-size: 10px; background: #2a2a2a; padding: 1px 3px; border-radius: 2px;'>{prompt_data.get('hash')}</code>")
                
                # Content Rating Details  
                content_info = []
                content_info.append(f"<strong>NSFW:</strong> {'Yes' if prompt_data.get('nsfw', False) else 'No'}")
                if prompt_data.get('nsfw_level'):
                    content_info.append(f"<strong>NSFW Level:</strong> {prompt_data.get('nsfw_level')}")
                if prompt_data.get('base_model'):
                    content_info.append(f"<strong>Base Model:</strong> {prompt_data.get('base_model')}")
                
                # Generation parameters - split into logical groups
                core_params = []
                if prompt_data.get('model_name'):
                    core_params.append(f"<strong>Model:</strong> {prompt_data.get('model_name')}")
                if prompt_data.get('steps'):
                    core_params.append(f"<strong>Steps:</strong> {prompt_data.get('steps')}")
                if prompt_data.get('sampler'):
                    core_params.append(f"<strong>Sampler:</strong> {prompt_data.get('sampler')}")
                if prompt_data.get('cfg_scale'):
                    core_params.append(f"<strong>CFG Scale:</strong> {prompt_data.get('cfg_scale')}")
                if prompt_data.get('seed'):
                    core_params.append(f"<strong>Seed:</strong> {prompt_data.get('seed')}")
                if prompt_data.get('size'):
                    core_params.append(f"<strong>Size:</strong> {prompt_data.get('size')}")
                
                # Advanced parameters
                advanced_params = []
                if prompt_data.get('clip_skip'):
                    advanced_params.append(f"<strong>CLIP Skip:</strong> {prompt_data.get('clip_skip')}")
                if prompt_data.get('denoising_strength'):
                    advanced_params.append(f"<strong>Denoising Strength:</strong> {prompt_data.get('denoising_strength')}")
                if prompt_data.get('vae'):
                    advanced_params.append(f"<strong>VAE:</strong> {prompt_data.get('vae')}")
                if prompt_data.get('model_hash'):
                    # Show FULL model hash, not truncated
                    advanced_params.append(f"<strong>Model Hash:</strong> <code style='font-size: 10px; background: #2a2a2a; padding: 1px 3px; border-radius: 2px;'>{prompt_data.get('model_hash')}</code>")
                
                # Hires/Upscaling parameters
                hires_params = []
                if prompt_data.get('hires_upscaler'):
                    hires_params.append(f"<strong>Hires Upscaler:</strong> {prompt_data.get('hires_upscaler')}")
                if prompt_data.get('hires_upscale'):
                    hires_params.append(f"<strong>Hires Scale:</strong> {prompt_data.get('hires_upscale')}")
                if prompt_data.get('hires_steps'):
                    hires_params.append(f"<strong>Hires Steps:</strong> {prompt_data.get('hires_steps')}")
                
                # Try to extract any additional meta parameters that might exist
                extra_params = []
                meta = prompt_data.get('meta', {})
                if isinstance(meta, dict):
                    # Look for any other interesting parameters in meta
                    interesting_keys = ['Eta', 'ENSD', 'Face restoration', 'Version', 
                                      'ControlNet', 'Lora', 'TI', 'Hypernet', 'AddNet',
                                      'First pass size', 'Schedule type', 'Schedule max sigma',
                                      'Schedule min sigma', 'Schedule rho']
                    for key in interesting_keys:
                        if key in meta and meta[key]:
                            extra_params.append(f"<strong>{key}:</strong> {meta[key]}")
                    
                    # Also capture any other keys that might be interesting
                    skip_keys = {'prompt', 'negativePrompt', 'steps', 'sampler', 'cfgScale', 
                               'seed', 'Model', 'clipSkip', 'Size', 'Denoising strength',
                               'Hires upscaler', 'Hires steps', 'Hires upscale', 
                               'Model hash', 'VAE'}
                    for key, value in meta.items():
                        if key not in skip_keys and value and str(value).strip():
                            extra_params.append(f"<strong>{key}:</strong> {value}")
                
                queue_item = f"""
                <div style='margin-bottom: 20px; padding: 15px; border: 1px solid #444; border-radius: 8px; 
                           background: {"#1e3a5f" if i >= current_index else "#2a2a2a"}; color: #fff;'>
                    <!-- Header with status and indicators -->
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;'>
                        <strong style='color: #fff; font-size: 14px;'>#{i + 1} - {status_text}</strong>
                        <div style='display: flex; gap: 8px; align-items: center; flex-wrap: wrap;'>
                            <span style='font-size: 14px;'>{status_icon}</span>
                            {' '.join(indicators)}
                        </div>
                    </div>
                    
                    <!-- Image and comprehensive metadata side by side -->
                    <div style='display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap;'>
                        <div style='flex-shrink: 0;'>
                            {image_html}
                        </div>
                        <div style='flex: 1; min-width: 300px;'>
                            <!-- Basic Image Info -->
                            <div style='margin-bottom: 10px; font-size: 12px; color: #bbb; line-height: 1.5;'>
                                {' | '.join(image_info) if image_info else 'No basic metadata available'}
                            </div>
                            
                            <!-- Image Metadata Section -->
                            {f'''
                            <div style='margin-bottom: 10px;'>
                                <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>🖼️ Image Metadata:</div>
                                <div style='font-size: 11px; color: #d1d5db; line-height: 1.4; background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #3b82f6;'>
                                    {' • '.join(image_metadata)}
                                </div>
                            </div>
                            ''' if image_metadata else ''}
                            
                            <!-- Content Rating Info -->
                            {f'''
                            <div style='margin-bottom: 10px;'>
                                <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>🔞 Content Rating:</div>
                                <div style='font-size: 11px; color: #d1d5db; line-height: 1.4; background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #ef4444;'>
                                    {' • '.join(content_info)}
                                </div>
                            </div>
                            ''' if content_info else ''}
                            
                            <!-- Core Generation Parameters -->
                            {f'''
                            <div style='margin-bottom: 10px;'>
                                <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>⚙️ Core Generation Settings:</div>
                                <div style='font-size: 11px; color: #d1d5db; line-height: 1.4; background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #10b981;'>
                                    {' • '.join(core_params[:4])}
                                    {('<br>' + ' • '.join(core_params[4:])) if len(core_params) > 4 else ''}
                                </div>
                            </div>
                            ''' if core_params else ''}
                            
                            <!-- Advanced Parameters -->
                            {f'''
                            <div style='margin-bottom: 10px;'>
                                <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>🔧 Advanced Settings:</div>
                                <div style='font-size: 11px; color: #d1d5db; line-height: 1.4; background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #f59e0b;'>
                                    {' • '.join(advanced_params[:3])}
                                    {('<br>' + ' • '.join(advanced_params[3:])) if len(advanced_params) > 3 else ''}
                                </div>
                            </div>
                            ''' if advanced_params else ''}
                            
                            <!-- Hires/Upscaling Parameters -->
                            {f'''
                            <div style='margin-bottom: 10px;'>
                                <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>🔍 Hires/Upscaling:</div>
                                <div style='font-size: 11px; color: #d1d5db; line-height: 1.4; background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #8b5cf6;'>
                                    {' • '.join(hires_params)}
                                </div>
                            </div>
                            ''' if hires_params else ''}
                            
                            <!-- Extra/Misc Parameters -->
                            {f'''
                            <div style='margin-bottom: 8px;'>
                                <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>📎 Additional Parameters:</div>
                                <div style='font-size: 11px; color: #d1d5db; line-height: 1.4; background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #6b7280;'>
                                    {' • '.join(extra_params[:4])}
                                    {('<br>' + ' • '.join(extra_params[4:8])) if len(extra_params) > 4 else ''}
                                    {('<br>' + ' • '.join(extra_params[8:])) if len(extra_params) > 8 else ''}
                                </div>
                            </div>
                            ''' if extra_params else ''}
                        </div>
                    </div>
                    
                    <!-- Prompts Section -->
                    <div style='margin-bottom: 10px;'>
                        <strong style='color: #4ade80; font-size: 13px;'>✨ Positive Prompt:</strong><br>
                        <span style='background: #1a3b1a; padding: 8px; border-radius: 4px; display: block; margin-top: 4px; line-height: 1.4; color: #e6ffe6; border: 1px solid #2d5a2d; font-size: 12px;'>{positive_preview}</span>
                    </div>
                    
                    <div>
                        <strong style='color: #ff6b6b; font-size: 13px;'>🚫 Negative Prompt:</strong><br>
                        <span style='background: #3b1a1a; padding: 8px; border-radius: 4px; display: block; margin-top: 4px; line-height: 1.4; color: #ffe6e6; border: 1px solid #5a2d2d; font-style: {"italic" if not negative_text else "normal"}; font-size: 12px;'>{negative_preview}</span>
                    </div>
                </div>
                """
                queue_items.append(queue_item)
            
            queue_display_content = f"""
            <div style='max-height: 800px; overflow-y: auto; padding: 10px; background: #0d1117; border-radius: 8px;'>
                <div style='margin-bottom: 15px; padding: 10px; background: #1c2938; border-radius: 6px; text-align: center; color: #fff; border: 1px solid #444;'>
                    <strong>Queue Status:</strong> {remaining} remaining out of {total_prompts} total prompts
                    <br><small style='color: #ccc;'>Click on any image to view full size</small>
                </div>
                {''.join(queue_items)}
            </div>
            """
            
            return queue_status, queue_display_content
        
        # Bind events
        test_api_btn.click(
            test_api_connection,
            outputs=[api_status]
        )
        
        refresh_loras_btn.click(
            refresh_lora_list,
            outputs=[lora_selection]
        )
        
        clear_cache_btn.click(
            clear_prompt_cache,
            outputs=[cache_status, prompt_queue_status]
        )
        
        fetch_prompts_btn.click(
            fetch_new_prompts,
            inputs=[nsfw_filter, keyword_filter, sort_method],
            outputs=[cache_status, prompt_queue_status, hidden_positive_prompt, hidden_negative_prompt]
        )
        
        # Bind the populate button to update bridge textboxes and use JavaScript to populate main fields
        populate_btn.click(
            get_prompts_and_update_queue,
            inputs=[custom_prompt_start, custom_prompt_end, custom_negative_prompt],
            outputs=[prompt_queue_status, hidden_positive_prompt, hidden_negative_prompt, queue_info, queue_display],
            _js="""
            function(custom_start, custom_end, custom_negative) {
                console.log('[Civitai Randomizer] Populate button clicked with JS!');
                console.log('[Civitai Randomizer] Custom inputs:', {custom_start, custom_end, custom_negative});
                
                // Give Python time to update the hidden textboxes
                setTimeout(() => {
                    // Read from hidden textboxes (the bridge)
                    const hiddenPositive = document.querySelector('#civitai_hidden_positive textarea');
                    const hiddenNegative = document.querySelector('#civitai_hidden_negative textarea');
                    
                    let positive_prompt = "Bridge not working!";
                    let negative_prompt = "Bridge not working!";
                    
                    if (hiddenPositive) {
                        positive_prompt = hiddenPositive.value;
                        console.log('[Civitai Randomizer] Read from bridge - Positive:', positive_prompt.substring(0, 100) + '...');
                    }
                    if (hiddenNegative) {
                        negative_prompt = hiddenNegative.value;
                        console.log('[Civitai Randomizer] Read from bridge - Negative:', negative_prompt.substring(0, 50) + '...');
                    }
                    
                    // Now populate main fields using the proven working approach
                    let positiveField = document.querySelector('#txt2img_prompt textarea');
                    let negativeField = document.querySelector('#txt2img_neg_prompt textarea');
                    
                    if (!positiveField) {
                        positiveField = document.querySelector('#img2img_prompt textarea');
                    }
                    if (!negativeField) {
                        negativeField = document.querySelector('#img2img_neg_prompt textarea');
                    }
                    
                    if (positiveField && negativeField) {
                        positiveField.value = positive_prompt;
                        negativeField.value = negative_prompt;
                        
                        ['input', 'change'].forEach(eventType => {
                            positiveField.dispatchEvent(new Event(eventType, {bubbles: true}));
                            negativeField.dispatchEvent(new Event(eventType, {bubbles: true}));
                        });
                        
                        console.log('[Civitai Randomizer] ✅ Main fields populated via bridge!');
                    } else {
                        console.log('[Civitai Randomizer] ❌ Could not find main prompt fields');
                    }
                }, 500);
                
                return [custom_start, custom_end, custom_negative];
            }
            """
        )
        
        # Bind queue refresh button
        refresh_queue_btn.click(
            refresh_queue_display,
            outputs=[queue_info, queue_display]
        )
        
        # Bind queue management buttons
        clear_queue_btn.click(
            clear_and_update_queue,
            outputs=[cache_status, prompt_queue_status, queue_info, queue_display]
        )
        
        reset_index_btn.click(
            reset_queue_index,
            outputs=[prompt_queue_status, queue_info, queue_display]
        )
        
        fetch_more_btn.click(
            fetch_more_from_queue,
            outputs=[cache_status, prompt_queue_status, hidden_positive_prompt, hidden_negative_prompt, queue_info, queue_display]
        )
        
        # Initialize LORA list on load
        loras = script_instance.get_available_loras()
        lora_selection.choices = loras
        
        print(f"[Civitai Randomizer] ✅ Tab interface with subtabs created successfully")
    
    return [(civitai_tab, "Civitai Randomizer", "civitai_randomizer")]

def on_ui_settings():
    """Add Civitai Randomizer settings to the Settings tab"""
    import modules.shared as shared
    
    section = ('civitai_randomizer', "Civitai Randomizer")
    
    shared.opts.add_option(
        "civitai_api_key",
        shared.OptionInfo(
            "",
            "Civitai API Key (optional for public content)",
            gr.Textbox,
            {"type": "password", "placeholder": "Enter your Civitai API key"},
            section=section
        )
    )

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings) 