# Updated: 2025-01-09 - Fixed f-string syntax error at line 1733
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
import sqlite3
import hashlib
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
        
        # Store last used filter settings for "Fetch More" functionality
        self.last_nsfw_filter = "Include All"
        self.last_keyword_filter = ""
        self.last_sort_method = "Most Reactions"
        self.last_next_page_url = None  # Store next page URL from API response
        
        # SQLite database for persistent Lora storage
        self.db_path = os.path.join(os.path.dirname(__file__), "lora_database.db")
        self.init_database()
        
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
        print(f"  Positive: '{original_prompt[:30]}...' -> '{p.prompt[:30]}...'")
        if hasattr(p, 'negative_prompt'):
            print(f"  Negative: '{original_negative[:30]}...' -> '{p.negative_prompt[:30]}...'")

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
            status_msg = f"OK Generated prompts! Queue: {remaining} remaining"
            
            print(f"[Civitai Randomizer] Returning prompts as outputs for JS access")
            return status_msg, positive, negative
        else:
            print(f"[Civitai Randomizer] No prompts available in queue")
            return "X No prompts available - fetch some prompts first!", "", ""

    def test_civitai_api(self, api_key: str) -> str:
        """Test connection to Civitai API with proper authentication validation"""
        try:
            if not api_key or not api_key.strip():
                return "<span style='color: red;'>X No API key provided</span>"
            
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
                    return f"<span style='color: green;'>OK API key valid - Authenticated as: {username}</span>"
                except:
                    return f"<span style='color: orange;'>! API key valid but unexpected user data format</span>"
            elif response.status_code == 401:
                return f"<span style='color: red;'>X Invalid API key - Authentication failed</span>"
            elif response.status_code == 403:
                return f"<span style='color: red;'>X API key forbidden - Check permissions</span>"
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
                    return f"<span style='color: green;'>OK API key appears valid (authenticated request successful)</span>"
                elif response.status_code == 401:
                    return f"<span style='color: red;'>X Invalid API key - Authentication failed</span>"
                else:
                    return f"<span style='color: orange;'>! API key might be valid but service issues (HTTP {response.status_code})</span>"
                
        except requests.exceptions.Timeout:
            return f"<span style='color: red;'>X Connection timeout - Check your internet connection</span>"
        except requests.exceptions.ConnectionError:
            return f"<span style='color: red;'>X Connection error - Cannot reach Civitai servers</span>"
        except Exception as e:
            return f"<span style='color: red;'>X Unexpected error: {str(e)}</span>"

    def send_to_stablequeue(self, prompt_data: Dict[str, Any]) -> str:
        """Send prompt data to StableQueue API"""
        try:
            import modules.shared as shared
            
            # Get StableQueue settings from configuration
            stablequeue_base_url = getattr(shared.opts, "stablequeue_url", "http://192.168.73.124:8083")
            api_key = getattr(shared.opts, "stablequeue_api_key", "")
            api_secret = getattr(shared.opts, "stablequeue_api_secret", "")
            default_server = getattr(shared.opts, "stablequeue_default_server", "ArchLinux")
            
            # Validate required settings
            if not stablequeue_base_url:
                error_msg = "StableQueue URL not configured. Please set it in Settings > CivitAI Randomizer"
                print(f"[CivitAI Randomizer StableQueue] {error_msg}")
                return error_msg
                
            if not api_key or not api_secret:
                error_msg = "StableQueue API credentials not configured. Please set them in Settings > CivitAI Randomizer"
                print(f"[CivitAI Randomizer StableQueue] {error_msg}")
                return error_msg
            
            # Create generation info string from prompt_data
            positive_prompt = prompt_data.get('positive', '')
            negative_prompt = prompt_data.get('negative', '')
            
            # Extract generation parameters from both direct fields and meta
            meta = prompt_data.get('meta', {})
            
            # Build a generation info string similar to what Forge would output
            geninfo_parts = []
            if positive_prompt:
                geninfo_parts.append(positive_prompt)
            
            if negative_prompt:
                geninfo_parts.append(f"Negative prompt: {negative_prompt}")
            
            # Add generation parameters in standard order
            param_parts = []
            
            # Use direct fields first, then fall back to meta
            steps = prompt_data.get('steps') or meta.get('steps') or meta.get('Steps')
            if steps:
                param_parts.append(f"Steps: {steps}")
                
            sampler = prompt_data.get('sampler') or meta.get('sampler') or meta.get('Sampler')
            if sampler:
                param_parts.append(f"Sampler: {sampler}")
                
            cfg_scale = prompt_data.get('cfg_scale') or meta.get('cfgScale') or meta.get('CFG scale') or meta.get('cfgScale')
            if cfg_scale:
                param_parts.append(f"CFG scale: {cfg_scale}")
                
            seed = prompt_data.get('seed') or meta.get('seed') or meta.get('Seed')
            if seed and str(seed) != '-1':
                param_parts.append(f"Seed: {seed}")
            
            # Size from width/height or meta
            width = prompt_data.get('image_width', 0)
            height = prompt_data.get('image_height', 0)
            size = meta.get('Size') or meta.get('size')
            if width and height:
                param_parts.append(f"Size: {width}x{height}")
            elif size:
                param_parts.append(f"Size: {size}")
                
            # Model information
            model_name = prompt_data.get('model_name') or meta.get('Model') or meta.get('model')
            if model_name:
                param_parts.append(f"Model: {model_name}")
                
            model_hash = prompt_data.get('model_hash') or meta.get('Model hash') or meta.get('modelHash')
            if model_hash:
                param_parts.append(f"Model hash: {model_hash}")
            
            # Other common parameters
            clip_skip = prompt_data.get('clip_skip') or meta.get('Clip skip') or meta.get('clipSkip')
            if clip_skip:
                param_parts.append(f"Clip skip: {clip_skip}")
                
            # Hires parameters
            hires_upscaler = prompt_data.get('hires_upscaler') or meta.get('Hires upscaler')
            if hires_upscaler:
                param_parts.append(f"Hires upscaler: {hires_upscaler}")
                
            hires_steps = prompt_data.get('hires_steps') or meta.get('Hires steps')
            if hires_steps:
                param_parts.append(f"Hires steps: {hires_steps}")
                
            hires_upscale = prompt_data.get('hires_upscale') or meta.get('Hires upscale')
            if hires_upscale:
                param_parts.append(f"Hires upscale: {hires_upscale}")
                
            denoising_strength = prompt_data.get('denoising_strength') or meta.get('Denoising strength')
            if denoising_strength:
                param_parts.append(f"Denoising strength: {denoising_strength}")
            
            # VAE
            vae = prompt_data.get('vae') or meta.get('VAE')
            if vae:
                param_parts.append(f"VAE: {vae}")
            
            # Add other meta parameters we haven't covered
            for key, value in meta.items():
                if key in ['prompt', 'negativePrompt', 'steps', 'Steps', 'sampler', 'Sampler', 
                          'cfgScale', 'CFG scale', 'seed', 'Seed', 'Size', 'size', 'Model', 
                          'model', 'Model hash', 'modelHash', 'Clip skip', 'clipSkip', 
                          'Hires upscaler', 'Hires steps', 'Hires upscale', 'Denoising strength', 'VAE']:
                    continue  # Skip already processed parameters
                if value is not None and str(value).strip():
                    param_parts.append(f"{key}: {value}")
            
            if param_parts:
                geninfo_parts.append(", ".join(param_parts))
            
            # Join all parts to create the complete generation info string
            geninfo = ", ".join(geninfo_parts)
            
            
            # Validate generation info
            if not geninfo or not geninfo.strip():
                print("[CivitAI Randomizer StableQueue] No generation info to send")
                return "No generation info available to send"
            
            print(f"[CivitAI Randomizer StableQueue] Sending generation info: {geninfo[:200]}...")
            
            # Build the StableQueue API endpoint URL
            stablequeue_url = f"{stablequeue_base_url.rstrip('/')}/api/v2/generate"
            
            # Prepare payload for StableQueue v2 API with complete generation info
            stablequeue_payload = {
                "app_type": "forge",
                "target_server_alias": default_server,
                "source_info": "civitai_randomizer",
                "generation_info_raw": geninfo.strip()  # Send complete generation info string
            }
            
            print(f"[CivitAI Randomizer StableQueue] Sending payload to {stablequeue_url}")
            
            # Send to StableQueue with API authentication
            headers = {
                'Content-Type': 'application/json',
                'X-API-Key': api_key,
                'X-API-Secret': api_secret
            }
            
            response = requests.post(
                stablequeue_url,
                json=stablequeue_payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code in [200, 202]:  # StableQueue v2 API returns 202 for successful job submission
                result = response.json()
                print(f"[CivitAI Randomizer StableQueue] Successfully queued job: {result}")
                job_id = result.get('mobilesd_job_id') or result.get('job_id', 'Unknown')
                queue_position = result.get('queue_position', 'Unknown')
                return f"✅ Successfully queued in StableQueue! Job ID: {job_id}, Queue Position: {queue_position}"
            else:
                error_msg = f"StableQueue API error: {response.status_code} - {response.text}"
                print(f"[CivitAI Randomizer StableQueue] {error_msg}")
                return error_msg
                
        except Exception as e:
            error_msg = f"Error sending to StableQueue: {str(e)}"
            print(f"[CivitAI Randomizer StableQueue] {error_msg}")
            return error_msg

    def stablequeue_output(self, prompt_index: str) -> str:
        """Handle StableQueue request from JavaScript - extract prompt data and send to StableQueue"""
        try:
            # Parse the prompt index from JavaScript input (same pattern as working extension)
            clean_index = prompt_index[4:]  # Remove the padding number prefix
            index = int(clean_index)
            print(f"[CivitAI Randomizer StableQueue] Processing index: {index}")
            
            # Get the prompt data from prompt_queue (same as search display uses)
            if not hasattr(self, 'prompt_queue') or not self.prompt_queue:
                return "No search results available. Please fetch prompts first."
            
            if 0 <= index < len(self.prompt_queue):
                prompt_data = self.prompt_queue[index]
                print(f"[CivitAI Randomizer StableQueue] Found prompt data for index {index}")
                return self.send_to_stablequeue(prompt_data)
            else:
                return f"Invalid prompt index {index}. Available prompts: 0-{len(self.prompt_queue)-1}"
                
        except Exception as e:
            error_msg = f"Error processing StableQueue request: {str(e)}"
            print(f"[CivitAI Randomizer StableQueue] {error_msg}")
            return error_msg

    def _setup_api_request(self, nsfw_filter: str, sort_method: str, limit: int, 
                          period_filter: str = "AllTime", username_filter: str = "", 
                          post_id_filter: int = None, model_id_filter: int = None, 
                          keyword_filter: str = "", use_next_page_url: bool = False) -> tuple:
        """Setup headers and parameters for Civitai API request"""
        # Setup headers
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
            print(f"[Debug] Using API key: {self.api_key[:8]}..." if len(self.api_key) > 8 else f"[Debug] Using API key: {self.api_key}")
        else:
            print(f"[Debug] No API key provided - making unauthenticated request")
        
        # For cursor-based pagination, if we have a nextPage URL, return it directly
        if use_next_page_url and self.last_next_page_url:
            print(f"[Pagination] Using cursor-based nextPage URL: {self.last_next_page_url[:100]}...")
            return headers, None, self.last_next_page_url
        
        # Convert filter options to API parameters
        nsfw_param = None
        if nsfw_filter == "Exclude NSFW":
            nsfw_param = "None"
            print("[NSFW Debug] Using enum value 'None' to exclude NSFW content")
        elif nsfw_filter == "Only NSFW":
            nsfw_param = "X"
            print("[NSFW Debug] Using enum value 'X' to request NSFW content")
        
        # We'll determine the correct sort value based on which endpoint we use
        # For now, just use the sort method as-is and we'll fix it in fetch_civitai_prompts
        params = {
            'limit': limit,
            'sort': sort_method
        }
        
        if nsfw_param is not None:
            params['nsfw'] = nsfw_param
        
        # Add new filter parameters
        if period_filter and period_filter != "AllTime":
            params['period'] = period_filter
        
        if username_filter and username_filter.strip():
            params['username'] = username_filter.strip()
        
        if post_id_filter and post_id_filter > 0:
            params['postId'] = int(post_id_filter)
        
        if model_id_filter and model_id_filter > 0:
            params['modelId'] = int(model_id_filter)
        
        # Add keyword search query parameter
        if keyword_filter and keyword_filter.strip():
            params['query'] = keyword_filter.strip()
            print(f"[Search DEBUG] Adding query parameter: '{keyword_filter.strip()}'")
        
        # Debug output
        print(f"[Civitai API DEBUG] Full request params: {params}")
        print(f"[NSFW Debug] Filter setting: '{nsfw_filter}' -> API param: {nsfw_param} (type: {type(nsfw_param)})")
        
        return headers, params, None

    def fetch_civitai_prompts(self, nsfw_filter: str, keyword_filter: str, sort_method: str, 
                             period_filter: str = "AllTime", username_filter: str = "", 
                             post_id_filter: int = None, model_id_filter: int = None,
                             limit: int = 100, is_fetch_more: bool = False) -> List[str]:
        """Fetch prompts from Civitai API using cursor-based pagination"""
        try:
            # Store last used settings for "Fetch More" functionality
            self.last_nsfw_filter = nsfw_filter
            self.last_keyword_filter = keyword_filter
            self.last_sort_method = sort_method
            
            # Debug: Print API key status
            print(f"[Civitai API] API key present: {bool(self.api_key)}")
            if self.api_key:
                print(f"[Civitai API] API key length: {len(self.api_key)}")
            
            # For new fetches, reset the nextPage URL
            if not is_fetch_more:
                self.last_next_page_url = None
                print(f"[Pagination] Starting fresh fetch - reset cursor")
            else:
                print(f"[Pagination] Fetching more using cursor-based pagination")
            
            headers, params, next_page_url = self._setup_api_request(nsfw_filter, sort_method, limit, 
                                                                   period_filter, username_filter, 
                                                                   post_id_filter, model_id_filter, 
                                                                   keyword_filter, is_fetch_more)
            
            # Make the API request
            if next_page_url:
                # Use the full nextPage URL provided by Civitai
                print(f"[Civitai API] Using nextPage URL: {next_page_url[:100]}...")
                response = requests.get(next_page_url, headers=headers, timeout=30)
            else:
                # Determine which API endpoint to use based on search query
                if keyword_filter and keyword_filter.strip():
                    # Use models endpoint for text search
                    api_url = 'https://civitai.com/api/v1/models'
                    print(f"[Search] Using models endpoint for keyword search: '{keyword_filter.strip()}'")
                    
                    # Map sort values for models endpoint
                    models_sort_mapping = {
                        "Most Reactions": "Most Liked",  # Closest equivalent
                        "Most Comments": "Most Discussed", 
                        "Most Collected": "Most Collected",
                        "Newest": "Newest"
                    }
                    
                    original_sort = params['sort']
                    params['sort'] = models_sort_mapping.get(original_sort, "Most Liked")
                    print(f"[Sort Mapping] Images sort '{original_sort}' -> Models sort '{params['sort']}'")
                else:
                    # Use images endpoint for non-search requests
                    api_url = 'https://civitai.com/api/v1/images'
                    print(f"[API] Using images endpoint for browsing")
                    
                    # Map sort values for images endpoint
                    images_sort_mapping = {
                        "Most Reactions": "Most Reactions",
                        "Most Comments": "Most Comments", 
                        "Most Collected": "Most Collected",
                        "Newest": "Newest"
                    }
                    
                    original_sort = params['sort']
                    params['sort'] = images_sort_mapping.get(original_sort, "Most Reactions")
                    print(f"[Sort Mapping] Using images sort: '{params['sort']}'")
                
                print(f"[Civitai API] Request URL: {api_url}")
                print(f"[Civitai API] Request params: {params}")
                response = requests.get(api_url, headers=headers, params=params, timeout=30)
                print(f"[Search DEBUG] Actual request URL: {response.url}")
            
            print(f"[Civitai API] Headers present: {list(headers.keys())}")
            
            data = self._validate_api_response(response)
            
            if not data or not isinstance(data, dict):
                print("Invalid response format from Civitai API")
                print(f"Response type: {type(data)}, content: {str(data)[:200]}")
                return []
            
            # Extract pagination metadata
            metadata = data.get('metadata', {})
            if metadata:
                # Update the nextPage URL for future "fetch more" requests
                self.last_next_page_url = metadata.get('nextPage')
                
                # Log cursor information instead of page numbers
                next_cursor = metadata.get('nextCursor')
                current_page = metadata.get('currentPage', 'Unknown')
                
                print(f"[Pagination] Current page: {current_page}")
                print(f"[Pagination] Next cursor: {next_cursor}")
                
                if self.last_next_page_url:
                    print(f"[Pagination] Next page URL available: {self.last_next_page_url[:100]}...")
                else:
                    print(f"[Pagination] No more pages available")
            
            prompts = []
            items = data.get('items', [])
            
            print(f"API response received - Total items: {len(items) if items else 0}")
            
            # Check if we're dealing with models or images response
            is_models_response = keyword_filter and keyword_filter.strip()
            
            if is_models_response:
                print(f"[Search] Processing models response for search: '{keyword_filter.strip()}'")
                # Convert models response to images format
                items = self._extract_images_from_models(items)
                print(f"[Search] Extracted {len(items)} images from models")
            
            # Debug: Show comprehensive info about first few items
            if items and len(items) > 0:
                print(f"[API Debug] Sample of first 3 items:")
                for i, item in enumerate(items[:3]):
                    item_id = item.get('id', 'No ID')
                    nsfw_status = item.get('nsfw', 'Unknown')
                    nsfw_level = item.get('nsfwLevel', 'Unknown')
                    print(f"  Item {i+1}: ID={item_id}, nsfw={nsfw_status}, nsfwLevel={nsfw_level}")
            
            if not items:
                if is_models_response:
                    print(f"No images found in models matching search: '{keyword_filter.strip()}'")
                else:
                    print("No items found in Civitai API response")
                print(f"Available keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                return []
            
            invalid_items = 0
            invalid_meta = 0
            nsfw_count = 0
            total_processed = 0
            filtered_out_count = 0
            duplicate_count = 0
            
            # Get existing image IDs for deduplication
            existing_ids = {prompt_data.get('id') for prompt_data in self.prompt_queue if prompt_data.get('id')}
            
            for item in items:
                # Skip None items
                if not item or not isinstance(item, dict):
                    invalid_items += 1
                    continue
                
                # Check for duplicates using image ID
                item_id = item.get('id')
                if item_id and item_id in existing_ids:
                    duplicate_count += 1
                    print(f"[Deduplication] Skipping duplicate item ID: {item_id}")
                    continue
                
                meta = item.get('meta', {})
                
                # Skip items with no meta or invalid meta
                if not meta or not isinstance(meta, dict):
                    invalid_meta += 1
                    continue
                
                positive_prompt = meta.get('prompt', '')
                if positive_prompt and isinstance(positive_prompt, str):
                    # Keyword filtering is now handled by the API query parameter
                    # No need for client-side keyword filtering
                    
                    prompt_pair = self._create_prompt_pair(item, meta)
                    
                    # Debug: Count NSFW items
                    total_processed += 1
                    if prompt_pair['nsfw']:
                        nsfw_count += 1
                    
                    # Apply client-side NSFW filtering
                    should_include = True
                    if nsfw_filter == "Exclude NSFW" and prompt_pair['nsfw']:
                        should_include = False
                    elif nsfw_filter == "Only NSFW" and not prompt_pair['nsfw']:
                        should_include = False
                    
                    # Add to queue only if it passes the filter
                    if should_include:
                        self.prompt_queue.append(prompt_pair)
                        prompts.append(positive_prompt)  # Legacy compatibility
                        # Add to existing_ids to prevent future duplicates
                        if item_id:
                            existing_ids.add(item_id)
                    else:
                        filtered_out_count += 1
            
            self.cached_prompts.extend(prompts)
            self.cached_prompts = list(set(self.cached_prompts))  # Remove duplicates
            
            print(f"Fetched {len(prompts)} new prompts from Civitai")
            if duplicate_count > 0:
                print(f"[Deduplication] Skipped {duplicate_count} duplicate items")
            if total_processed > 0:
                nsfw_percentage = (nsfw_count/total_processed*100)
                print(f"[NSFW Debug] Processed {total_processed} items, {nsfw_count} marked as NSFW ({nsfw_percentage:.1f}%)")
                print(f"[NSFW Debug] Filter setting: '{nsfw_filter}'")
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
            if filtered_out_count > 0:
                print(f"[NSFW Filter] Filtered out {filtered_out_count} items due to '{nsfw_filter}' setting")
            
            # Reset queue index when starting fresh (not fetch more)
            if not is_fetch_more and len(self.prompt_queue) > 0:
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
        lora_path = shared.cmd_opts.lora_dir
        if not lora_path:
            # Use default paths - check both models/Lora and extensions
            possible_paths = [
                os.path.join(shared.models_path, "Lora"),
                os.path.join(shared.models_path, "lora"),
                "models/Lora",
                "extensions"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    lora_path = path
                    break
        
        if not lora_path or not os.path.exists(lora_path):
            return ["No LORA directory found"]
        
        loras = []
        for root, dirs, files in os.walk(lora_path):
            for file in files:
                if file.endswith(('.safetensors', '.pt', '.ckpt')):
                    relative_path = os.path.relpath(os.path.join(root, file), lora_path)
                    loras.append(relative_path)
        
        return sorted(loras) if loras else ["No LORA files found"]

    def get_lora_directory_path(self) -> str:
        """Get the configured Lora directory path"""
        # First check if user has configured a custom path
        custom_path = getattr(shared.opts, 'civitai_lora_path', '')
        if custom_path and custom_path.strip() and os.path.exists(custom_path.strip()):
            return custom_path.strip()
        
        # Fallback to existing logic
        lora_path = shared.cmd_opts.lora_dir
        if not lora_path:
            # Use default paths - check both models/Lora and extensions
            possible_paths = [
                os.path.join(shared.models_path, "Lora"),
                os.path.join(shared.models_path, "lora"),
                "models/Lora",
                "extensions"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    lora_path = path
                    break
        
        return lora_path if lora_path and os.path.exists(lora_path) else None

    def get_checkpoint_directory_path(self) -> str:
        """Get the configured Checkpoint directory path"""
        # First check if user has configured a custom path
        custom_path = getattr(shared.opts, 'civitai_checkpoint_path', '')
        if custom_path and custom_path.strip() and os.path.exists(custom_path.strip()):
            return custom_path.strip()
        
        # Fallback to existing logic for checkpoints
        checkpoint_path = getattr(shared.cmd_opts, 'ckpt_dir', None)
        if not checkpoint_path:
            # Use default paths - check both models/Stable-diffusion and extensions
            possible_paths = [
                os.path.join(shared.models_path, "Stable-diffusion"),
                os.path.join(shared.models_path, "stable-diffusion"),
                "models/Stable-diffusion",
                "models/stable-diffusion"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    break
        
        return checkpoint_path if checkpoint_path and os.path.exists(checkpoint_path) else None

    def init_database(self):
        """Initialize the SQLite database for Lora storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create loras table with Civitai enrichment support
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS loras (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL UNIQUE,
                        file_path TEXT NOT NULL,
                        relative_path TEXT NOT NULL,
                        sha256 TEXT,
                        file_size INTEGER,
                        modified_time REAL,
                        scan_time REAL,
                        metadata_json TEXT,
                        model_id TEXT,
                        model_version_id TEXT,
                        name TEXT,
                        description TEXT,
                        activation_text TEXT,
                        sd_version TEXT,
                        base_model TEXT,
                        -- Civitai enrichment fields
                        civitai_model_id INTEGER,
                        civitai_version_id INTEGER,
                        civitai_model_name TEXT,
                        civitai_model_description TEXT,
                        civitai_creator_username TEXT,
                        civitai_creator_image TEXT,
                        civitai_download_count INTEGER,
                        civitai_favorite_count INTEGER,
                        civitai_comment_count INTEGER,
                        civitai_rating_count INTEGER,
                        civitai_rating_average REAL,
                        civitai_thumbs_up INTEGER,
                        civitai_thumbs_down INTEGER,
                        civitai_tags_json TEXT,
                        civitai_trigger_words_json TEXT,
                        civitai_images_json TEXT,
                        civitai_training_details_json TEXT,
                        civitai_download_url TEXT,
                        civitai_last_updated REAL,
                        civitai_enriched_at REAL,
                        created_at REAL DEFAULT (datetime('now')),
                        updated_at REAL DEFAULT (datetime('now'))
                    )
                ''')
                
                # Create checkpoints table with Civitai enrichment support
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL UNIQUE,
                        file_path TEXT NOT NULL,
                        relative_path TEXT NOT NULL,
                        sha256 TEXT,
                        file_size INTEGER,
                        modified_time REAL,
                        scan_time REAL,
                        metadata_json TEXT,
                        model_id TEXT,
                        model_version_id TEXT,
                        name TEXT,
                        description TEXT,
                        sd_version TEXT,
                        base_model TEXT,
                        -- Civitai enrichment fields
                        civitai_model_id INTEGER,
                        civitai_version_id INTEGER,
                        civitai_model_name TEXT,
                        civitai_model_description TEXT,
                        civitai_creator_username TEXT,
                        civitai_creator_image TEXT,
                        civitai_download_count INTEGER,
                        civitai_favorite_count INTEGER,
                        civitai_comment_count INTEGER,
                        civitai_rating_count INTEGER,
                        civitai_rating_average REAL,
                        civitai_thumbs_up INTEGER,
                        civitai_thumbs_down INTEGER,
                        civitai_tags_json TEXT,
                        civitai_trigger_words_json TEXT,
                        civitai_images_json TEXT,
                        civitai_training_details_json TEXT,
                        civitai_download_url TEXT,
                        civitai_last_updated REAL,
                        civitai_enriched_at REAL,
                        created_at REAL DEFAULT (datetime('now')),
                        updated_at REAL DEFAULT (datetime('now'))
                    )
                ''')
                
                # Create scan_stats table for tracking scan progress
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS scan_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scan_date REAL,
                        lora_directory TEXT,
                        checkpoint_directory TEXT,
                        total_files INTEGER,
                        processed_files INTEGER,
                        files_with_metadata INTEGER,
                        files_with_hashes INTEGER,
                        scan_duration REAL,
                        scan_type TEXT,
                        content_type TEXT
                    )
                ''')
                
                # Database migration: Add missing columns if they don't exist
                self.auto_backup_before_migration()
                self._migrate_database_schema(cursor)
                
                # Create indexes for fast lookups
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_loras_sha256 ON loras(sha256)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_loras_name ON loras(name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_loras_filename ON loras(filename)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_loras_modified_time ON loras(modified_time)')
                # Civitai enrichment indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_loras_civitai_model_id ON loras(civitai_model_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_loras_civitai_version_id ON loras(civitai_version_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_loras_civitai_enriched_at ON loras(civitai_enriched_at)')
                
                # Create indexes for checkpoints
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_checkpoints_sha256 ON checkpoints(sha256)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_checkpoints_name ON checkpoints(name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_checkpoints_filename ON checkpoints(filename)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_checkpoints_modified_time ON checkpoints(modified_time)')
                # Civitai enrichment indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_checkpoints_civitai_model_id ON checkpoints(civitai_model_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_checkpoints_civitai_version_id ON checkpoints(civitai_version_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_checkpoints_civitai_enriched_at ON checkpoints(civitai_enriched_at)')
                
                conn.commit()
                print(f"[Lora DB] Database initialized successfully: {self.db_path}")
                
        except Exception as e:
            print(f"[Lora DB] Error initializing database: {e}")

    def _migrate_database_schema(self, cursor):
        """Migrate database schema to add missing columns"""
        try:
            # Check and add missing columns for loras table
            cursor.execute("PRAGMA table_info(loras)")
            loras_columns = [column[1] for column in cursor.fetchall()]
            
            civitai_columns = [
                'civitai_model_id', 'civitai_version_id', 'civitai_model_name', 'civitai_model_description',
                'civitai_creator_username', 'civitai_creator_image', 'civitai_download_count', 
                'civitai_favorite_count', 'civitai_comment_count', 'civitai_rating_count', 
                'civitai_rating_average', 'civitai_thumbs_up', 'civitai_thumbs_down', 
                'civitai_tags_json', 'civitai_trigger_words_json', 'civitai_images_json',
                'civitai_training_details_json', 'civitai_download_url', 'civitai_last_updated',
                'civitai_enriched_at', 'created_at', 'updated_at'
            ]
            
            for column in civitai_columns:
                if column not in loras_columns:
                    column_type = 'REAL' if column in ['civitai_rating_average', 'civitai_last_updated', 'civitai_enriched_at', 'created_at', 'updated_at'] else 'TEXT'
                    if column in ['civitai_model_id', 'civitai_version_id', 'civitai_download_count', 'civitai_favorite_count', 'civitai_comment_count', 'civitai_rating_count', 'civitai_thumbs_up', 'civitai_thumbs_down']:
                        column_type = 'INTEGER'
                    
                    default_value = "DEFAULT (datetime('now'))" if column in ['created_at', 'updated_at'] else ''
                    cursor.execute(f'ALTER TABLE loras ADD COLUMN {column} {column_type} {default_value}')
                    print(f"[DB Migration] Added column {column} to loras table")
            
            # Check and add missing columns for checkpoints table
            cursor.execute("PRAGMA table_info(checkpoints)")
            checkpoints_columns = [column[1] for column in cursor.fetchall()]
            
            for column in civitai_columns:
                if column not in checkpoints_columns:
                    column_type = 'REAL' if column in ['civitai_rating_average', 'civitai_last_updated', 'civitai_enriched_at', 'created_at', 'updated_at'] else 'TEXT'
                    if column in ['civitai_model_id', 'civitai_version_id', 'civitai_download_count', 'civitai_favorite_count', 'civitai_comment_count', 'civitai_rating_count', 'civitai_thumbs_up', 'civitai_thumbs_down']:
                        column_type = 'INTEGER'
                    
                    default_value = "DEFAULT (datetime('now'))" if column in ['created_at', 'updated_at'] else ''
                    cursor.execute(f'ALTER TABLE checkpoints ADD COLUMN {column} {column_type} {default_value}')
                    print(f"[DB Migration] Added column {column} to checkpoints table")
                    
        except Exception as e:
            print(f"[DB Migration] Error migrating schema: {e}")

    def get_db_connection(self):
        """Get a database connection with proper configuration"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        return conn

    def get_lora_count(self) -> int:
        """Get total number of Loras in database"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM loras')
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"[Lora DB] Error getting Lora count: {e}")
            return 0

    def clear_database(self):
        """Clear all Lora data from database"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM loras')
                cursor.execute('DELETE FROM scan_stats')
                conn.commit()
                print(f"[Lora DB] Database cleared successfully")
                return True
        except Exception as e:
            print(f"[Lora DB] Error clearing database: {e}")
            return False

    def vacuum_database(self):
        """Vacuum database to optimize performance"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('VACUUM')
                print(f"[Database] Database vacuumed successfully")
                return True
        except Exception as e:
            print(f"[Database] Error vacuuming database: {e}")
            return False
    
    def backup_database(self, backup_name: str = None) -> Dict[str, Any]:
        """Create a backup of the database"""
        try:
            import datetime
            import shutil
            
            if not os.path.exists(self.db_path):
                return {"success": False, "message": "Database file does not exist"}
            
            # Create backups directory
            backup_dir = os.path.join(os.path.dirname(__file__), "database_backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Generate backup filename
            if not backup_name:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"lora_database_backup_{timestamp}.db"
            elif not backup_name.endswith('.db'):
                backup_name += '.db'
            
            backup_path = os.path.join(backup_dir, backup_name)
            
            # Create backup
            shutil.copy2(self.db_path, backup_path)
            
            # Get backup size
            backup_size = os.path.getsize(backup_path)
            size_str = f"{backup_size / (1024**2):.1f} MB" if backup_size > 1024**2 else f"{backup_size / 1024:.1f} KB"
            
            print(f"[Database] Backup created: {backup_path} ({size_str})")
            
            return {
                "success": True, 
                "message": f"Backup created successfully: {backup_name} ({size_str})",
                "backup_path": backup_path,
                "backup_name": backup_name,
                "backup_size": backup_size
            }
            
        except Exception as e:
            error_msg = f"Error creating backup: {str(e)}"
            print(f"[Database] {error_msg}")
            return {"success": False, "message": error_msg}
    
    def restore_database(self, backup_name: str) -> Dict[str, Any]:
        """Restore database from backup"""
        try:
            import shutil
            
            backup_dir = os.path.join(os.path.dirname(__file__), "database_backups")
            backup_path = os.path.join(backup_dir, backup_name)
            
            if not os.path.exists(backup_path):
                return {"success": False, "message": f"Backup file not found: {backup_name}"}
            
            # Create backup of current database before restore
            current_backup = self.backup_database("pre_restore_backup")
            
            # Restore from backup
            shutil.copy2(backup_path, self.db_path)
            
            # Get restored database size
            restored_size = os.path.getsize(self.db_path)
            size_str = f"{restored_size / (1024**2):.1f} MB" if restored_size > 1024**2 else f"{restored_size / 1024:.1f} KB"
            
            print(f"[Database] Database restored from: {backup_name} ({size_str})")
            
            return {
                "success": True, 
                "message": f"Database restored successfully from {backup_name} ({size_str})",
                "current_backup": current_backup.get("backup_name") if current_backup.get("success") else None
            }
            
        except Exception as e:
            error_msg = f"Error restoring backup: {str(e)}"
            print(f"[Database] {error_msg}")
            return {"success": False, "message": error_msg}
    
    def list_database_backups(self) -> List[Dict[str, Any]]:
        """List all available database backups"""
        try:
            import datetime
            
            backup_dir = os.path.join(os.path.dirname(__file__), "database_backups")
            
            if not os.path.exists(backup_dir):
                return []
            
            backups = []
            for filename in os.listdir(backup_dir):
                if filename.endswith('.db'):
                    backup_path = os.path.join(backup_dir, filename)
                    try:
                        stat = os.stat(backup_path)
                        backups.append({
                            "filename": filename,
                            "path": backup_path,
                            "size": stat.st_size,
                            "modified_time": stat.st_mtime,
                            "modified_date": datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                        })
                    except Exception as e:
                        print(f"[Database] Error reading backup {filename}: {e}")
            
            # Sort by modification time (newest first)
            backups.sort(key=lambda x: x["modified_time"], reverse=True)
            return backups
            
        except Exception as e:
            print(f"[Database] Error listing backups: {e}")
            return []
    
    def auto_backup_before_migration(self):
        """Create automatic backup before database migration"""
        try:
            if os.path.exists(self.db_path):
                backup_result = self.backup_database("pre_migration_auto_backup")
                if backup_result.get("success"):
                    print(f"[Database] Auto-backup created before migration: {backup_result.get('backup_name')}")
                    return True
                else:
                    print(f"[Database] Failed to create auto-backup: {backup_result.get('message')}")
                    return False
            return True
        except Exception as e:
            print(f"[Database] Error creating auto-backup: {e}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get basic stats
                cursor.execute('SELECT COUNT(*) as total FROM loras')
                total_loras = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) as with_hash FROM loras WHERE sha256 IS NOT NULL AND sha256 != ""')
                with_hash = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) as with_metadata FROM loras WHERE metadata_json IS NOT NULL AND metadata_json != ""')
                with_metadata = cursor.fetchone()[0]
                
                cursor.execute('SELECT SUM(file_size) as total_size FROM loras')
                total_size = cursor.fetchone()[0] or 0
                
                # Get last scan info
                cursor.execute('SELECT * FROM scan_stats ORDER BY scan_date DESC LIMIT 1')
                last_scan = cursor.fetchone()
                
                return {
                    'total_loras': total_loras,
                    'with_hash': with_hash,
                    'with_metadata': with_metadata,
                    'total_size': total_size,
                    'last_scan': dict(last_scan) if last_scan else None
                }
        except Exception as e:
            print(f"[Lora DB] Error getting database stats: {e}")
            return {}

    def calculate_file_sha256(self, file_path: str) -> Optional[str]:
        """Calculate SHA256 hash of a file with optimized chunk size for large files"""
        try:
            import hashlib
            file_size = os.path.getsize(file_path)
            
            # Use larger chunks for better performance on large checkpoint files
            # 1MB chunks are much more efficient than 4KB for multi-GB files
            chunk_size = 1024 * 1024  # 1MB chunks
            
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest().upper()  # Use uppercase for consistency
        except Exception as e:
            print(f"[File Hash] Error calculating SHA256 for {file_path}: {e}")
            return None

    def load_lora_metadata(self, lora_file_path: str) -> Dict[str, Any]:
        """Load metadata from .json file associated with a Lora file"""
        base_path = os.path.splitext(lora_file_path)[0]
        json_path = base_path + ".json"
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    print(f"[Lora Metadata] Loaded metadata for {os.path.basename(lora_file_path)}")
                    return metadata
            except Exception as e:
                print(f"[Lora Metadata] Error reading {json_path}: {e}")
        
        return {}

    def scan_local_loras(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Scan local Loras and store in SQLite database with incremental updates"""
        start_time = time.time()
        
        print(f"[Lora DB] Starting Lora scan (force_refresh={force_refresh})...")
        lora_path = self.get_lora_directory_path()
        
        if not lora_path:
            print(f"[Lora DB] No valid Lora directory found")
            return {}
        
        print(f"[Lora DB] Scanning directory: {lora_path}")
        
        # Get existing Loras from database
        existing_loras = {}
        if not force_refresh:
            try:
                with self.get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT * FROM loras')
                    for row in cursor.fetchall():
                        existing_loras[row['relative_path']] = dict(row)
                print(f"[Lora DB] Found {len(existing_loras)} existing Loras in database")
            except Exception as e:
                print(f"[Lora DB] Error loading existing Loras: {e}")
        
        # Scan filesystem
        discovered_files = {}
        for root, dirs, files in os.walk(lora_path):
            for file in files:
                if file.endswith(('.safetensors', '.pt', '.ckpt')):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, lora_path)
                    
                    try:
                        file_stat = os.stat(file_path)
                        discovered_files[relative_path] = {
                            'file_path': file_path,
                            'file_size': file_stat.st_size,
                            'modified_time': file_stat.st_mtime
                        }
                    except Exception as e:
                        print(f"[Lora DB] Error reading file stats for {file_path}: {e}")
        
        print(f"[Lora DB] Discovered {len(discovered_files)} Lora files on disk")
        
        # Determine what needs to be processed
        files_to_process = []
        files_to_remove = []
        
        if force_refresh:
            # Process all discovered files
            files_to_process = list(discovered_files.keys())
        else:
            # Only process new or modified files
            for relative_path, file_info in discovered_files.items():
                existing = existing_loras.get(relative_path)
                if not existing or existing['modified_time'] != file_info['modified_time']:
                    files_to_process.append(relative_path)
            
            # Find files that no longer exist
            for relative_path in existing_loras:
                if relative_path not in discovered_files:
                    files_to_remove.append(relative_path)
        
        print(f"[Lora DB] Processing {len(files_to_process)} files, removing {len(files_to_remove)} orphaned entries")
        
        # Remove orphaned entries
        if files_to_remove:
            try:
                with self.get_db_connection() as conn:
                    cursor = conn.cursor()
                    for relative_path in files_to_remove:
                        cursor.execute('DELETE FROM loras WHERE relative_path = ?', (relative_path,))
                    conn.commit()
                    print(f"[Lora DB] Removed {len(files_to_remove)} orphaned entries")
            except Exception as e:
                print(f"[Lora DB] Error removing orphaned entries: {e}")
        
        # Process files
        processed_count = 0
        metadata_count = 0
        hash_count = 0
        
        for relative_path in files_to_process:
            file_info = discovered_files[relative_path]
            file_path = file_info['file_path']
            
            try:
                # Load metadata
                metadata = self.load_lora_metadata(file_path)
                metadata_json = json.dumps(metadata) if metadata else None
                if metadata:
                    metadata_count += 1
                
                # Calculate hash
                file_hash = self.calculate_file_sha256(file_path)
                if file_hash:
                    hash_count += 1
                
                # Extract metadata fields
                model_id = metadata.get('modelId', '') if metadata else ''
                model_version_id = metadata.get('modelVersionId', '') if metadata else ''
                name = metadata.get('name', '') if metadata else ''
                description = metadata.get('description', '') if metadata else ''
                activation_text = metadata.get('activation text', '') if metadata else ''
                sd_version = metadata.get('sd version', '') if metadata else ''
                base_model = metadata.get('baseModel', '') if metadata else ''
                
                # Insert or update in database
                with self.get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO loras (
                            filename, file_path, relative_path, sha256, file_size, modified_time,
                            scan_time, metadata_json, model_id, model_version_id, name, description,
                            activation_text, sd_version, base_model, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    ''', (
                        os.path.basename(file_path), file_path, relative_path, file_hash,
                        file_info['file_size'], file_info['modified_time'], time.time(),
                        metadata_json, model_id, model_version_id, name, description,
                        activation_text, sd_version, base_model
                    ))
                    conn.commit()
                
                processed_count += 1
                
                # Progress logging
                if processed_count % 10 == 0:
                    print(f"[Lora DB] Processed {processed_count}/{len(files_to_process)} files...")
                
            except Exception as e:
                print(f"[Lora DB] Error processing {file_path}: {e}")
        
        # Record scan statistics
        scan_duration = time.time() - start_time
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO scan_stats (
                        scan_date, lora_directory, total_files, processed_files,
                        files_with_metadata, files_with_hashes, scan_duration, scan_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    time.time(), lora_path, len(discovered_files), processed_count,
                    metadata_count, hash_count, scan_duration,
                    'full' if force_refresh else 'incremental'
                ))
                conn.commit()
        except Exception as e:
            print(f"[Lora DB] Error recording scan stats: {e}")
        
        print(f"[Lora DB] Scan complete:")
        print(f"  Total files discovered: {len(discovered_files)}")
        print(f"  Files processed: {processed_count}")
        print(f"  With metadata: {metadata_count}")
        print(f"  With hashes: {hash_count}")
        print(f"  Scan duration: {scan_duration:.2f} seconds")
        
        # Return database stats in legacy format for compatibility
        return self._get_legacy_cache_format()

    def _get_legacy_cache_format(self) -> Dict[str, Dict[str, Any]]:
        """Convert database entries to legacy cache format for compatibility"""
        legacy_cache = {}
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM loras')
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    metadata = json.loads(row_dict['metadata_json']) if row_dict['metadata_json'] else {}
                    
                    legacy_cache[row_dict['relative_path']] = {
                        'file_path': row_dict['file_path'],
                        'sha256': row_dict['sha256'],
                        'metadata': metadata,
                        'file_size': row_dict['file_size'],
                        'modified_time': row_dict['modified_time']
                    }
        except Exception as e:
            print(f"[Lora DB] Error converting to legacy format: {e}")
        
        return legacy_cache

    def get_checkpoint_count(self) -> int:
        """Get total number of Checkpoints in database"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM checkpoints')
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"[Checkpoint DB] Error getting Checkpoint count: {e}")
            return 0

    def scan_local_checkpoints(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Scan local Checkpoints and store in SQLite database with incremental updates"""
        start_time = time.time()
        
        print(f"[Checkpoint DB] Starting Checkpoint scan (force_refresh={force_refresh})...")
        checkpoint_path = self.get_checkpoint_directory_path()
        
        if not checkpoint_path:
            print(f"[Checkpoint DB] No valid Checkpoint directory found")
            return {}
        
        print(f"[Checkpoint DB] Scanning directory: {checkpoint_path}")
        
        # Get existing Checkpoints from database
        existing_checkpoints = {}
        if not force_refresh:
            try:
                with self.get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT * FROM checkpoints')
                    for row in cursor.fetchall():
                        existing_checkpoints[row['relative_path']] = dict(row)
                print(f"[Checkpoint DB] Found {len(existing_checkpoints)} existing Checkpoints in database")
            except Exception as e:
                print(f"[Checkpoint DB] Error loading existing Checkpoints: {e}")
        
        # Scan filesystem
        discovered_files = {}
        for root, dirs, files in os.walk(checkpoint_path):
            for file in files:
                if file.endswith(('.safetensors', '.pt', '.ckpt')):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, checkpoint_path)
                    
                    try:
                        file_stat = os.stat(file_path)
                        discovered_files[relative_path] = {
                            'file_path': file_path,
                            'file_size': file_stat.st_size,
                            'modified_time': file_stat.st_mtime
                        }
                    except Exception as e:
                        print(f"[Checkpoint DB] Error reading file stats for {file_path}: {e}")
        
        print(f"[Checkpoint DB] Discovered {len(discovered_files)} Checkpoint files on disk")
        
        # Determine what needs to be processed
        files_to_process = []
        files_to_remove = []
        
        if force_refresh:
            # Process all discovered files
            files_to_process = list(discovered_files.keys())
        else:
            # Only process new or modified files
            for relative_path, file_info in discovered_files.items():
                existing = existing_checkpoints.get(relative_path)
                if not existing or existing['modified_time'] != file_info['modified_time']:
                    files_to_process.append(relative_path)
            
            # Find files that no longer exist
            for relative_path in existing_checkpoints:
                if relative_path not in discovered_files:
                    files_to_remove.append(relative_path)
        
        print(f"[Checkpoint DB] Processing {len(files_to_process)} files, removing {len(files_to_remove)} orphaned entries")
        
        # Remove orphaned entries
        if files_to_remove:
            try:
                with self.get_db_connection() as conn:
                    cursor = conn.cursor()
                    for relative_path in files_to_remove:
                        cursor.execute('DELETE FROM checkpoints WHERE relative_path = ?', (relative_path,))
                    conn.commit()
                    print(f"[Checkpoint DB] Removed {len(files_to_remove)} orphaned entries")
            except Exception as e:
                print(f"[Checkpoint DB] Error removing orphaned entries: {e}")
        
        # Process files
        processed_count = 0
        metadata_count = 0
        hash_count = 0
        
        for relative_path in files_to_process:
            file_info = discovered_files[relative_path]
            file_path = file_info['file_path']
            file_size = file_info['file_size']
            
            # Format file size for display
            if file_size > 1024**3:
                size_str = f"{file_size / (1024**3):.1f} GB"
            elif file_size > 1024**2:
                size_str = f"{file_size / (1024**2):.1f} MB"
            else:
                size_str = f"{file_size / 1024:.1f} KB"
            
            print(f"[Checkpoint DB] Processing {processed_count + 1}/{len(files_to_process)}: {os.path.basename(file_path)} ({size_str})")
            
            try:
                # Load metadata (checkpoints use the same .json format as loras)
                metadata = self.load_lora_metadata(file_path)  # Reuse the same method
                metadata_json = json.dumps(metadata) if metadata else None
                if metadata:
                    metadata_count += 1
                
                # Calculate hash (this is the slow part for large files)
                file_hash = None
                
                # Skip hash calculation for very large files (>4GB) to speed up scanning
                # Hash calculation on 8GB files can take 5-10 minutes each
                skip_hash_threshold = 4 * 1024 * 1024 * 1024  # 4GB
                
                if file_size > skip_hash_threshold and not force_refresh:
                    print(f"[Checkpoint DB] Skipping hash calculation for large file {os.path.basename(file_path)} ({size_str}) - use 'Force Rescan' to include hashes")
                else:
                    print(f"[Checkpoint DB] Calculating SHA256 hash for {os.path.basename(file_path)}...")
                    hash_start_time = time.time()
                    file_hash = self.calculate_file_sha256(file_path)
                    hash_duration = time.time() - hash_start_time
                    
                    if file_hash:
                        hash_count += 1
                        print(f"[Checkpoint DB] Hash calculated in {hash_duration:.1f}s: {file_hash[:16]}...")
                    else:
                        print(f"[Checkpoint DB] Failed to calculate hash")
                
                # Extract metadata fields
                model_id = metadata.get('modelId', '') if metadata else ''
                model_version_id = metadata.get('modelVersionId', '') if metadata else ''
                name = metadata.get('name', '') if metadata else ''
                description = metadata.get('description', '') if metadata else ''
                sd_version = metadata.get('sd version', '') if metadata else ''
                base_model = metadata.get('baseModel', '') if metadata else ''
                
                # Insert or update in database
                with self.get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO checkpoints (
                            filename, file_path, relative_path, sha256, file_size, modified_time,
                            scan_time, metadata_json, model_id, model_version_id, name, description,
                            sd_version, base_model, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    ''', (
                        os.path.basename(file_path), file_path, relative_path, file_hash,
                        file_info['file_size'], file_info['modified_time'], time.time(),
                        metadata_json, model_id, model_version_id, name, description,
                        sd_version, base_model
                    ))
                    conn.commit()
                
                processed_count += 1
                elapsed = time.time() - start_time
                avg_time_per_file = elapsed / processed_count
                remaining_files = len(files_to_process) - processed_count
                estimated_remaining = avg_time_per_file * remaining_files
                
                print(f"[Checkpoint DB] OK Completed {processed_count}/{len(files_to_process)} files. ETA: {estimated_remaining/60:.1f} min")
                
            except Exception as e:
                print(f"[Checkpoint DB] Error processing {file_path}: {e}")
                processed_count += 1  # Still count it as processed to maintain accurate progress
        
        # Record scan statistics
        scan_duration = time.time() - start_time
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO scan_stats (
                        scan_date, checkpoint_directory, total_files, processed_files,
                        files_with_metadata, files_with_hashes, scan_duration, scan_type, content_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    time.time(), checkpoint_path, len(discovered_files), processed_count,
                    metadata_count, hash_count, scan_duration,
                    'full' if force_refresh else 'incremental', 'checkpoints'
                ))
                conn.commit()
        except Exception as e:
            print(f"[Checkpoint DB] Error recording scan stats: {e}")
        
        print(f"[Checkpoint DB] Scan complete:")
        print(f"  Total files discovered: {len(discovered_files)}")
        print(f"  Files processed: {processed_count}")
        print(f"  With metadata: {metadata_count}")
        print(f"  With hashes: {hash_count}")
        print(f"  Scan duration: {scan_duration:.2f} seconds")
        
        # Return database stats in legacy format for compatibility
        return self._get_legacy_checkpoint_cache_format()

    def _get_legacy_checkpoint_cache_format(self) -> Dict[str, Dict[str, Any]]:
        """Convert checkpoint database entries to legacy cache format for compatibility"""
        legacy_cache = {}
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM checkpoints')
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    metadata = json.loads(row_dict['metadata_json']) if row_dict['metadata_json'] else {}
                    
                    legacy_cache[row_dict['relative_path']] = {
                        'file_path': row_dict['file_path'],
                        'sha256': row_dict['sha256'],
                        'metadata': metadata,
                        'file_size': row_dict['file_size'],
                        'modified_time': row_dict['modified_time']
                    }
        except Exception as e:
            print(f"[Checkpoint DB] Error converting to legacy format: {e}")
        
        return legacy_cache

    def lookup_civitai_model_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Look up model information from Civitai using SHA256 hash"""
        if not file_hash:
            return None
            
        try:
            # Use the by-hash endpoint to get model version details
            api_url = f"https://civitai.com/api/v1/model-versions/by-hash/{file_hash}"
            
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            print(f"[Civitai API] Looking up model by hash: {file_hash[:16]}...")
            response = requests.get(api_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"[Civitai API] OK Found model: {data.get('model', {}).get('name', 'Unknown')}")
                return data
            elif response.status_code == 404:
                print(f"[Civitai API] X No model found for hash: {file_hash[:16]}")
                return None
            else:
                print(f"[Civitai API] ! API error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"[Civitai API] ERROR Error looking up hash {file_hash[:16]}: {e}")
            return None

    def extract_civitai_metadata(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize Civitai metadata from API response"""
        try:
            model_data = api_data.get('model', {})
            version_data = api_data
            
            # Extract model-level information
            extracted = {
                'civitai_model_id': model_data.get('id'),
                'civitai_version_id': version_data.get('id'),
                'civitai_model_name': model_data.get('name', ''),
                'civitai_model_description': model_data.get('description', ''),
                'civitai_creator_username': model_data.get('creator', {}).get('username', ''),
                'civitai_creator_image': model_data.get('creator', {}).get('image', ''),
                
                # Statistics
                'civitai_download_count': version_data.get('stats', {}).get('downloadCount', 0),
                'civitai_favorite_count': model_data.get('stats', {}).get('favoriteCount', 0),
                'civitai_comment_count': model_data.get('stats', {}).get('commentCount', 0),
                'civitai_rating_count': model_data.get('stats', {}).get('ratingCount', 0),
                'civitai_rating_average': model_data.get('stats', {}).get('rating', 0.0),
                'civitai_thumbs_up': model_data.get('stats', {}).get('thumbsUpCount', 0),
                'civitai_thumbs_down': model_data.get('stats', {}).get('thumbsDownCount', 0),
                
                # Tags and trigger words
                'civitai_tags_json': json.dumps(model_data.get('tags', [])),
                'civitai_trigger_words_json': json.dumps(version_data.get('trainedWords', [])),
                
                # Images (first 5 images)
                'civitai_images_json': json.dumps(version_data.get('images', [])[:5]),
                
                # Training details
                'civitai_training_details_json': json.dumps({
                    'base_model': version_data.get('baseModel', ''),
                    'base_model_type': version_data.get('baseModelType', ''),
                    'training_strength': version_data.get('trainingStrength'),
                    'training_details': version_data.get('trainingDetails', {}),
                    'description': version_data.get('description', ''),
                    'published_at': version_data.get('publishedAt', ''),
                    'updated_at': version_data.get('updatedAt', ''),
                }),
                
                # Download information
                'civitai_download_url': version_data.get('downloadUrl', ''),
                'civitai_last_updated': time.time(),
                'civitai_enriched_at': time.time()
            }
            
            return extracted
            
        except Exception as e:
            print(f"[Civitai Metadata] Error extracting metadata: {e}")
            return {}

    def enrich_models_with_civitai_data(self, table_name: str, file_hashes: List[str] = None, 
                                       force_refresh: bool = False) -> Dict[str, Any]:
        """Enrich models with Civitai data using their SHA256 hashes"""
        try:
            print(f"[Civitai Enrichment] Starting enrichment for {table_name} table...")
            
            # Get models that need enrichment
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                if file_hashes:
                    # Enrich specific hashes
                    placeholders = ','.join(['?' for _ in file_hashes])
                    query = f'''
                        SELECT id, filename, sha256, civitai_enriched_at 
                        FROM {table_name} 
                        WHERE sha256 IN ({placeholders}) AND sha256 IS NOT NULL AND sha256 != ""
                    '''
                    cursor.execute(query, file_hashes)
                else:
                    # Enrich all models without Civitai data or refresh old data
                    if force_refresh:
                        cursor.execute(f'''
                            SELECT id, filename, sha256, civitai_enriched_at 
                            FROM {table_name} 
                            WHERE sha256 IS NOT NULL AND sha256 != ""
                        ''')
                    else:
                        cursor.execute(f'''
                            SELECT id, filename, sha256, civitai_enriched_at 
                            FROM {table_name} 
                            WHERE sha256 IS NOT NULL AND sha256 != ""
                            AND (civitai_enriched_at IS NULL OR civitai_enriched_at < ?)
                        ''', (time.time() - 7*24*3600,))  # Refresh weekly
                
                models_to_enrich = cursor.fetchall()
            
            if not models_to_enrich:
                print(f"[Civitai Enrichment] No {table_name} models need enrichment")
                return {'enriched': 0, 'errors': 0, 'skipped': 0}
            
            print(f"[Civitai Enrichment] Found {len(models_to_enrich)} {table_name} models to enrich")
            
            enriched_count = 0
            error_count = 0
            skipped_count = 0
            
            for i, model in enumerate(models_to_enrich):
                model_id, filename, file_hash, last_enriched = model
                
                print(f"[Civitai Enrichment] Processing {i+1}/{len(models_to_enrich)}: {filename}")
                
                # Rate limiting - wait between API calls
                if i > 0:
                    time.sleep(1)  # 1 second between calls to be respectful
                
                # Look up model data
                api_data = self.lookup_civitai_model_by_hash(file_hash)
                
                if api_data:
                    # Extract metadata
                    civitai_fields = self.extract_civitai_metadata(api_data)
                    
                    if civitai_fields:
                        # Update database with Civitai data
                        with self.get_db_connection() as conn:
                            cursor = conn.cursor()
                            
                            # Build update query dynamically
                            field_names = list(civitai_fields.keys())
                            set_clause = ', '.join([f'{field} = ?' for field in field_names])
                            values = list(civitai_fields.values()) + [model_id]
                            
                            update_query = f'''
                                UPDATE {table_name} 
                                SET {set_clause}, updated_at = datetime('now')
                                WHERE id = ?
                            '''
                            
                            cursor.execute(update_query, values)
                            conn.commit()
                            
                        enriched_count += 1
                        print(f"[Civitai Enrichment] OK Enriched: {filename}")
                    else:
                        error_count += 1
                        print(f"[Civitai Enrichment] ! Failed to extract metadata: {filename}")
                else:
                    skipped_count += 1
                    print(f"[Civitai Enrichment] SKIP No Civitai data found: {filename}")
            
            print(f"[Civitai Enrichment] Completed for {table_name}:")
            print(f"  OK Enriched: {enriched_count}")
            print(f"  ! Errors: {error_count}")
            print(f"  SKIP Skipped: {skipped_count}")
            
            return {
                'enriched': enriched_count,
                'errors': error_count,
                'skipped': skipped_count,
                'total_processed': len(models_to_enrich)
            }
            
        except Exception as e:
            print(f"[Civitai Enrichment] ERROR Error during enrichment: {e}")
            import traceback
            traceback.print_exc()
            return {'enriched': 0, 'errors': 1, 'skipped': 0, 'total_processed': 0}

    def search_checkpoints_db(self, name_query: str = "", hash_query: str = "", 
                             folder_query: str = "", has_metadata: bool = False, 
                             has_hash: bool = False, selected_folders: List[str] = None, 
                             limit: int = 100) -> List[Dict[str, Any]]:
        """Search Checkpoints in database with various filters"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Build query dynamically based on filters
                where_conditions = []
                params = []
                
                if name_query:
                    where_conditions.append(
                        '(LOWER(name) LIKE ? OR LOWER(filename) LIKE ? OR LOWER(description) LIKE ?)'
                    )
                    search_term = f'%{name_query.lower()}%'
                    params.extend([search_term, search_term, search_term])
                
                if hash_query:
                    where_conditions.append('UPPER(sha256) LIKE ?')
                    params.append(f'%{hash_query.upper()}%')
                
                if folder_query:
                    where_conditions.append('LOWER(relative_path) LIKE ?')
                    params.append(f'%{folder_query.lower()}%')
                
                # Filter by selected folders (from dropdown)
                if selected_folders:
                    # Create folder filter conditions
                    folder_conditions = []
                    for folder in selected_folders:
                        if folder == "(root)":
                            # For root folder, look for files with no path separator
                            folder_conditions.append("relative_path NOT LIKE '%/%'")
                        else:
                            # For named folders, look for files that start with folder name
                            folder_conditions.append("LOWER(relative_path) LIKE ?")
                            params.append(f'{folder.lower()}/%')
                    
                    if folder_conditions:
                        where_conditions.append(f"({' OR '.join(folder_conditions)})")
                
                if has_metadata:
                    where_conditions.append('metadata_json IS NOT NULL AND metadata_json != ""')
                
                if has_hash:
                    where_conditions.append('sha256 IS NOT NULL AND sha256 != ""')
                
                # Construct final query
                base_query = 'SELECT * FROM checkpoints'
                if where_conditions:
                    base_query += ' WHERE ' + ' AND '.join(where_conditions)
                base_query += ' ORDER BY relative_path, filename LIMIT ?'
                params.append(limit)
                
                cursor.execute(base_query, params)
                results = []
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    # Parse metadata JSON
                    if row_dict['metadata_json']:
                        try:
                            row_dict['metadata'] = json.loads(row_dict['metadata_json'])
                        except:
                            row_dict['metadata'] = {}
                    else:
                        row_dict['metadata'] = {}
                    results.append(row_dict)
                
                return results
                
        except Exception as e:
            print(f"[Checkpoint DB] Error searching database: {e}")
            return []

    def get_checkpoint_folder_choices(self) -> List[str]:
        """Get all unique folder paths for the checkpoint filter dropdown"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT DISTINCT
                        CASE 
                            WHEN instr(relative_path, '/') > 0 
                            THEN substr(relative_path, 1, instr(relative_path, '/') - 1)
                            ELSE '(root)'
                        END as folder_path
                    FROM checkpoints 
                    ORDER BY 
                        CASE WHEN folder_path = '(root)' THEN 0 ELSE 1 END,
                        folder_path
                ''')
                
                folders = [row[0] for row in cursor.fetchall()]
                return folders
                
        except Exception as e:
            print(f"[Checkpoint DB] Error getting folder choices: {e}")
            return []

    def format_checkpoint_database_display(self, checkpoints: List[Dict[str, Any]], query_info: str = "") -> str:
        """Format Checkpoint database results for HTML display with rich Civitai integration"""
        if not checkpoints:
            return """
            <div style='padding: 30px; text-align: center; color: #ccc; background: #1a1a1a; border-radius: 8px;'>
                <h3>[EMPTY] No Checkpoints Found</h3>
                <p>No Checkpoints match your search criteria. Try adjusting your filters or scan your Checkpoint directory.</p>
            </div>
            """
        
        # Generate HTML for each Checkpoint
        checkpoint_items = []
        
        for checkpoint in checkpoints:
            # Format file size
            file_size = checkpoint.get('file_size', 0)
            if file_size > 1024 * 1024 * 1024:
                size_str = f"{file_size / (1024**3):.1f} GB"
            elif file_size > 1024 * 1024:
                size_str = f"{file_size / (1024**2):.1f} MB"
            elif file_size > 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size} bytes"
            
            # Format dates
            import datetime
            try:
                if checkpoint.get('modified_time'):
                    mod_date = datetime.datetime.fromtimestamp(checkpoint['modified_time']).strftime('%Y-%m-%d %H:%M')
                else:
                    mod_date = "Unknown"
            except:
                mod_date = "Unknown"
            
            # Status indicators
            has_hash = bool(checkpoint.get('sha256'))
            has_metadata = bool(checkpoint.get('metadata_json'))
            has_civitai = bool(checkpoint.get('civitai_enriched_at'))
            
            hash_indicator = "OK" if has_hash else "X"
            metadata_indicator = "OK" if has_metadata else "X"
            civitai_indicator = "*" if has_civitai else "o"
            
            # Metadata info
            metadata = checkpoint.get('metadata', {})
            model_id = metadata.get('modelId', '')
            description = metadata.get('description', '')
            
            # Civitai data
            civitai_name = checkpoint.get('civitai_model_name', '')
            civitai_description = checkpoint.get('civitai_model_description', '')
            civitai_creator = checkpoint.get('civitai_creator_username', '')
            civitai_downloads = checkpoint.get('civitai_download_count', 0)
            civitai_rating = checkpoint.get('civitai_rating_average', 0.0)
            civitai_rating_count = checkpoint.get('civitai_rating_count', 0)
            civitai_tags = []
            civitai_trigger_words = []
            civitai_images = []
            
            # Parse JSON fields
            try:
                if checkpoint.get('civitai_tags_json'):
                    civitai_tags = json.loads(checkpoint.get('civitai_tags_json', '[]'))
                if checkpoint.get('civitai_trigger_words_json'):
                    civitai_trigger_words = json.loads(checkpoint.get('civitai_trigger_words_json', '[]'))
                if checkpoint.get('civitai_images_json'):
                    civitai_images = json.loads(checkpoint.get('civitai_images_json', '[]'))
            except:
                pass
            
            # Generate Civitai images display
            checkpoint_images_html = ""
            if civitai_images:
                # Show first 3 images in a row
                image_items = []
                for img in civitai_images[:3]:
                    img_url = img.get('url', '')
                    if img_url:
                        image_items.append(f"""
                        <img src="{img_url}" 
                             style="width: 80px; height: 80px; object-fit: cover; border-radius: 4px; 
                                    cursor: pointer; transition: transform 0.2s;"
                             onclick="window.open('{img_url}', '_blank')"
                             onmouseover="this.style.transform='scale(1.1)'"
                             onmouseout="this.style.transform='scale(1)'"
                             title="Click to view full size">
                        """)
                
                if image_items:
                    checkpoint_images_html = f"""
                    <div style='margin-bottom: 8px;'>
                        <div style='font-size: 11px; color: #9ca3af; margin-bottom: 4px;'><strong>- Civitai Images:</strong></div>
                        <div style='display: flex; gap: 8px; flex-wrap: wrap;'>
                            {''.join(image_items)}
                        </div>
                    </div>
                    """
            
            # Create Checkpoint item HTML
            checkpoint_item = f"""
            <div style='margin-bottom: 20px; padding: 15px; border: 1px solid #444; border-radius: 8px; 
                       background: #2a2a2a; color: #fff;'>
                
                <!-- Header with filename and status -->
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                    <strong style='color: #60a5fa; font-size: 15px;'>{civitai_name if civitai_name else checkpoint.get('filename', 'Unknown')}</strong>
                    <div style='display: flex; gap: 8px; align-items: center;'>
                        <span style='font-size: 12px;'>{hash_indicator} Hash</span>
                        <span style='font-size: 12px;'>{metadata_indicator} Metadata</span>
                        <span style='font-size: 12px;'>{civitai_indicator} Civitai</span>
                    </div>
                </div>
                
                <!-- Civitai enriched content -->
                {'''
                <div style='display: flex; gap: 15px; margin-bottom: 12px;'>
                    <!-- Images column -->
                    <div style='flex-shrink: 0;'>
                        ''' + checkpoint_images_html + '''
                    </div>
                    
                    <!-- Info column -->
                    <div style='flex: 1;'>
                        <div style='margin-bottom: 8px;'>
                            <div style='font-size: 11px; color: #9ca3af; margin-bottom: 2px;'><strong>- Creator & Stats:</strong></div>
                            <div style='background: #1a1a1a; padding: 6px; border-radius: 4px; font-size: 11px; border: 1px solid #374151; line-height: 1.4;'>
                                <strong>Creator:</strong> ''' + (civitai_creator if civitai_creator else "Unknown") + '''<br>
                                <strong>Downloads:</strong> ''' + f"{civitai_downloads:,}" + ''' | <strong>Rating:</strong> ''' + ("*" * int(civitai_rating) + "☆" * (5-int(civitai_rating))) + f" ({civitai_rating:.1f}/5, {civitai_rating_count} votes)" + '''<br>
                                <strong>Description:</strong> ''' + ((civitai_description[:100] + "...") if len(civitai_description) > 100 else (civitai_description or "No description")) + '''
                            </div>
                        </div>
                        
                        ''' + ('''
                        <div style='margin-bottom: 8px;'>
                            <div style='font-size: 11px; color: #9ca3af; margin-bottom: 2px;'><strong>- Trigger Words:</strong></div>
                            <div style='background: #1a1a1a; padding: 6px; border-radius: 4px; font-size: 11px; border: 1px solid #374151;'>
                                ''' + (", ".join(civitai_trigger_words[:10]) if civitai_trigger_words else "No trigger words specified") + '''
                            </div>
                        </div>
                        ''' if civitai_trigger_words else "") + '''
                        
                        ''' + ('''
                        <div style='margin-bottom: 8px;'>
                            <div style='font-size: 11px; color: #9ca3af; margin-bottom: 2px;'><strong>- Tags:</strong></div>
                            <div style='background: #1a1a1a; padding: 6px; border-radius: 4px; font-size: 11px; border: 1px solid #374151;'>
                                ''' + (", ".join(['<span style="background: #374151; padding: 1px 4px; border-radius: 2px; margin: 1px;">' + tag + '</span>' for tag in civitai_tags[:8]]) if civitai_tags else "No tags") + '''
                            </div>
                        </div>
                        ''' if civitai_tags else "") + '''
                    </div>
                </div>
                ''' if has_civitai else ""}
                
                <!-- File info -->
                <div style='font-size: 11px; color: #bbb; margin-bottom: 8px;'>
                    <strong>Folder:</strong> <span style='color: #60a5fa; font-family: monospace;'>{os.path.dirname(checkpoint.get('relative_path', '')) or '(root)'}</span><br>
                    <strong>File:</strong> {os.path.basename(checkpoint.get('relative_path', 'Unknown'))}<br>
                    <strong>Size:</strong> {size_str} | <strong>Modified:</strong> {mod_date}
                </div>
                
                <!-- Hash info -->
                {'''
                <div style='margin-bottom: 8px;'>
                    <div style='font-size: 11px; color: #9ca3af; margin-bottom: 2px;'><strong>SHA256 Hash:</strong></div>
                    <code style='background: #1a1a1a; padding: 4px 6px; border-radius: 4px; font-size: 10px; 
                                color: #10b981; border: 1px solid #374151; word-break: break-all;'>''' + str(checkpoint.get('sha256')) + '''</code>
                </div>
                ''' if has_hash else ''}
                
                <!-- Metadata info -->
                {'''
                <div style='margin-bottom: 8px;'>
                    <div style='font-size: 11px; color: #9ca3af; margin-bottom: 2px;'><strong>Metadata:</strong></div>
                    <div style='background: #1a1a1a; padding: 6px; border-radius: 4px; font-size: 11px; 
                               border: 1px solid #374151; line-height: 1.4;'>
                        ''' + (f"<strong>Model ID:</strong> {model_id}<br>" if model_id else "") + '''
                        ''' + (f"<strong>Name:</strong> {metadata.get('name', '')}<br>" if metadata.get('name') else "") + '''
                        ''' + (f"<strong>Description:</strong> {description}<br>" if description else "") + '''
                        ''' + (f"<strong>Base Model:</strong> {metadata.get('baseModel', '')}<br>" if metadata.get('baseModel') else "") + '''
                        ''' + (f"<strong>SD Version:</strong> {metadata.get('sd version', '')}" if metadata.get('sd version') else "") + '''
                    </div>
                </div>
                ''' if has_metadata and metadata else ''}
                
            </div>
            """
            
            checkpoint_items.append(checkpoint_item)
        
        # Combine all items
        results_html = f"""
        <div style='max-height: 800px; overflow-y: auto; padding: 10px; background: #0d1117; border-radius: 8px;'>
            <div style='margin-bottom: 15px; padding: 10px; background: #1c2938; border-radius: 6px; 
                       text-align: center; color: #fff; border: 1px solid #444;'>
                <strong>Database Results:</strong> {len(checkpoints)} Checkpoints found
                {f"<br><small style='color: #ccc;'>{query_info}</small>" if query_info else ""}
            </div>
            {''.join(checkpoint_items)}
        </div>
        """
        
        return results_html

    def parse_loras_from_civitai_prompt(self, prompt_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Lora information from Civitai prompt metadata"""
        loras_found = []
        
        # Get the meta information
        meta = prompt_data.get('meta', {})
        if not isinstance(meta, dict):
            return loras_found
        
        # Look for Lora information in various possible keys
        lora_keys = ['Lora', 'LoRA', 'lora', 'AddNet', 'TI']  # Common keys where Lora info might be stored
        
        for key in lora_keys:
            if key in meta and meta[key]:
                lora_value = meta[key]
                if isinstance(lora_value, str):
                    # Parse Lora string format
                    # Common formats: "loraname:strength", "loraname:strength:hash", etc.
                    loras_found.extend(self._parse_lora_string(lora_value))
                elif isinstance(lora_value, dict):
                    # Direct Lora object
                    loras_found.append(lora_value)
                elif isinstance(lora_value, list):
                    # List of Loras
                    for lora_item in lora_value:
                        if isinstance(lora_item, dict):
                            loras_found.append(lora_item)
                        elif isinstance(lora_item, str):
                            loras_found.extend(self._parse_lora_string(lora_item))
        
        # Also look for Lora tags in the positive prompt text
        positive_prompt = prompt_data.get('positive', '')
        if positive_prompt:
            loras_found.extend(self._extract_loras_from_prompt_text(positive_prompt))
        
        print(f"[Lora Parser] Found {len(loras_found)} Loras in prompt ID {prompt_data.get('id', 'unknown')}")
        return loras_found

    def _parse_lora_string(self, lora_string: str) -> List[Dict[str, Any]]:
        """Parse Lora information from string format"""
        loras = []
        
        # Handle comma-separated Loras
        if ',' in lora_string:
            parts = [part.strip() for part in lora_string.split(',') if part.strip()]
        else:
            parts = [lora_string.strip()] if lora_string.strip() else []
        
        for part in parts:
            # Parse individual Lora entries
            # Format examples: "loraname:0.8", "loraname:0.8:hash123", etc.
            if ':' in part:
                components = part.split(':')
                lora_info = {
                    'name': components[0].strip(),
                    'strength': float(components[1]) if len(components) > 1 and components[1].replace('.', '').isdigit() else 1.0,
                    'hash': components[2].strip() if len(components) > 2 else None,
                    'source': 'metadata_string'
                }
                loras.append(lora_info)
            else:
                # Just a name
                loras.append({
                    'name': part.strip(),
                    'strength': 1.0,
                    'hash': None,
                    'source': 'metadata_string'
                })
        
        return loras

    def _extract_loras_from_prompt_text(self, prompt_text: str) -> List[Dict[str, Any]]:
        """Extract Lora tags from prompt text using regex"""
        loras = []
        
        # Pattern to match <lora:name:strength> format
        import re
        lora_pattern = r'<lora:([^:>]+):?([0-9]*\.?[0-9]*)>'
        
        matches = re.findall(lora_pattern, prompt_text, re.IGNORECASE)
        
        for match in matches:
            name = match[0].strip()
            strength = float(match[1]) if match[1] else 1.0
            
            loras.append({
                'name': name,
                'strength': strength,
                'hash': None,
                'source': 'prompt_text'
            })
        
        return loras

    def check_lora_availability(self, civitai_loras: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check which Civitai Loras are available locally using database"""
        availability_results = []
        
        for civitai_lora in civitai_loras:
            result = {
                'civitai_lora': civitai_lora,
                'available': False,
                'local_matches': [],
                'match_method': None
            }
            
            # Try different matching methods
            
            # Method 1: Exact hash match (most reliable)
            civitai_hash = civitai_lora.get('hash')
            if civitai_hash:
                hash_matches = self._find_loras_by_hash_db(civitai_hash)
                if hash_matches:
                    result['available'] = True
                    result['local_matches'] = hash_matches
                    result['match_method'] = 'hash'
                    availability_results.append(result)
                    continue
            
            # Method 2: Name matching (less reliable but still useful)
            civitai_name = civitai_lora.get('name', '').lower()
            if civitai_name:
                name_matches = self._find_loras_by_name_db(civitai_name)
                if name_matches:
                    result['available'] = True
                    result['local_matches'] = name_matches
                    result['match_method'] = 'name'
            
            availability_results.append(result)
        
        return availability_results

    def _find_loras_by_hash_db(self, target_hash: str) -> List[Dict[str, Any]]:
        """Find local Loras that match the given hash using database"""
        matches = []
        target_hash_upper = target_hash.upper() if target_hash else None
        
        if not target_hash_upper:
            return matches
        
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Check direct hash match
                cursor.execute('SELECT * FROM loras WHERE UPPER(sha256) = ?', (target_hash_upper,))
                for row in cursor.fetchall():
                    matches.append({
                        'filename': row['filename'],
                        'file_path': row['file_path'],
                        'match_type': 'sha256_direct',
                        'relative_path': row['relative_path']
                    })
                
                # Check hash from metadata
                cursor.execute('''
                    SELECT * FROM loras 
                    WHERE metadata_json IS NOT NULL 
                    AND json_extract(metadata_json, '$.sha256') IS NOT NULL
                    AND UPPER(json_extract(metadata_json, '$.sha256')) = ?
                ''', (target_hash_upper,))
                
                for row in cursor.fetchall():
                    # Avoid duplicates
                    if not any(m['file_path'] == row['file_path'] for m in matches):
                        matches.append({
                            'filename': row['filename'],
                            'file_path': row['file_path'],
                            'match_type': 'sha256_metadata',
                            'relative_path': row['relative_path']
                        })
                        
        except Exception as e:
            print(f"[Lora DB] Error finding Loras by hash: {e}")
        
        return matches

    def _find_loras_by_name_db(self, target_name: str) -> List[Dict[str, Any]]:
        """Find local Loras that match the given name using database"""
        matches = []
        target_name_lower = target_name.lower() if target_name else ""
        
        if not target_name_lower:
            return matches
        
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Exact filename match (without extension)
                cursor.execute('''
                    SELECT * FROM loras 
                    WHERE LOWER(REPLACE(REPLACE(REPLACE(filename, '.safetensors', ''), '.pt', ''), '.ckpt', '')) = ?
                ''', (target_name_lower,))
                
                for row in cursor.fetchall():
                    matches.append({
                        'filename': row['filename'],
                        'file_path': row['file_path'],
                        'match_type': 'filename_exact',
                        'relative_path': row['relative_path']
                    })
                
                # Partial filename match
                if not matches:  # Only do partial if no exact matches
                    cursor.execute('''
                        SELECT * FROM loras 
                        WHERE LOWER(filename) LIKE ? OR LOWER(filename) LIKE ?
                    ''', (f'%{target_name_lower}%', f'{target_name_lower}%'))
                    
                    for row in cursor.fetchall():
                        matches.append({
                            'filename': row['filename'],
                            'file_path': row['file_path'],
                            'match_type': 'filename_partial',
                            'relative_path': row['relative_path']
                        })
                
                # Check metadata name fields
                if not matches:  # Only if no filename matches
                    cursor.execute('''
                        SELECT * FROM loras 
                        WHERE LOWER(name) = ? OR LOWER(name) LIKE ?
                        OR json_extract(metadata_json, '$.name') IS NOT NULL
                        AND LOWER(json_extract(metadata_json, '$.name')) = ?
                    ''', (target_name_lower, f'%{target_name_lower}%', target_name_lower))
                    
                    for row in cursor.fetchall():
                        matches.append({
                            'filename': row['filename'],
                            'file_path': row['file_path'],
                            'match_type': 'metadata_name',
                            'relative_path': row['relative_path']
                        })
                        
        except Exception as e:
            print(f"[Lora DB] Error finding Loras by name: {e}")
        
        return matches

    def search_loras_db(self, name_query: str = "", hash_query: str = "", 
                       folder_query: str = "", has_metadata: bool = False, 
                       has_hash: bool = False, selected_folders: List[str] = None, 
                       limit: int = 100) -> List[Dict[str, Any]]:
        """Search Loras in database with various filters"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Build query dynamically based on filters
                where_conditions = []
                params = []
                
                if name_query:
                    where_conditions.append(
                        '(LOWER(name) LIKE ? OR LOWER(filename) LIKE ? OR LOWER(description) LIKE ?)'
                    )
                    search_term = f'%{name_query.lower()}%'
                    params.extend([search_term, search_term, search_term])
                
                if hash_query:
                    where_conditions.append('UPPER(sha256) LIKE ?')
                    params.append(f'%{hash_query.upper()}%')
                
                if folder_query:
                    where_conditions.append('LOWER(relative_path) LIKE ?')
                    params.append(f'%{folder_query.lower()}%')
                
                # Filter by selected folders (from dropdown)
                if selected_folders:
                    # Create folder filter conditions
                    folder_conditions = []
                    for folder in selected_folders:
                        if folder == "(root)":
                            # For root folder, look for files with no path separator
                            folder_conditions.append("relative_path NOT LIKE '%/%'")
                        else:
                            # For named folders, look for files that start with folder name
                            folder_conditions.append("LOWER(relative_path) LIKE ?")
                            params.append(f'{folder.lower()}/%')
                    
                    if folder_conditions:
                        where_conditions.append(f"({' OR '.join(folder_conditions)})")
                
                if has_metadata:
                    where_conditions.append('metadata_json IS NOT NULL AND metadata_json != ""')
                
                if has_hash:
                    where_conditions.append('sha256 IS NOT NULL AND sha256 != ""')
                
                # Construct final query
                base_query = 'SELECT * FROM loras'
                if where_conditions:
                    base_query += ' WHERE ' + ' AND '.join(where_conditions)
                base_query += ' ORDER BY relative_path, filename LIMIT ?'
                params.append(limit)
                
                cursor.execute(base_query, params)
                results = []
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    # Parse metadata JSON
                    if row_dict['metadata_json']:
                        try:
                            row_dict['metadata'] = json.loads(row_dict['metadata_json'])
                        except:
                            row_dict['metadata'] = {}
                    else:
                        row_dict['metadata'] = {}
                    results.append(row_dict)
                
                return results
                
        except Exception as e:
            print(f"[Lora DB] Error searching database: {e}")
            return []

    def get_folder_choices(self) -> List[str]:
        """Get all unique folder paths for the filter dropdown"""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT DISTINCT
                        CASE 
                            WHEN instr(relative_path, '/') > 0 
                            THEN substr(relative_path, 1, instr(relative_path, '/') - 1)
                            ELSE '(root)'
                        END as folder_path
                    FROM loras 
                    ORDER BY 
                        CASE WHEN folder_path = '(root)' THEN 0 ELSE 1 END,
                        folder_path
                ''')
                
                folders = [row[0] for row in cursor.fetchall()]
                return folders
                
        except Exception as e:
            print(f"[Lora DB] Error getting folder choices: {e}")
            return []

    def format_lora_database_display(self, loras: List[Dict[str, Any]], query_info: str = "") -> str:
        """Format Lora database results for HTML display with rich Civitai integration"""
        if not loras:
            return """
            <div style='padding: 30px; text-align: center; color: #ccc; background: #1a1a1a; border-radius: 8px;'>
                <h3>[EMPTY] No Loras Found</h3>
                <p>No Loras match your search criteria. Try adjusting your filters or scan your Lora directory.</p>
            </div>
            """
        
        # Generate HTML for each Lora
        lora_items = []
        
        for lora in loras:
            # Format file size
            file_size = lora.get('file_size', 0)
            if file_size > 1024 * 1024 * 1024:
                size_str = f"{file_size / (1024**3):.1f} GB"
            elif file_size > 1024 * 1024:
                size_str = f"{file_size / (1024**2):.1f} MB"
            elif file_size > 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size} bytes"
            
            # Format dates
            import datetime
            try:
                if lora.get('modified_time'):
                    mod_date = datetime.datetime.fromtimestamp(lora['modified_time']).strftime('%Y-%m-%d %H:%M')
                else:
                    mod_date = "Unknown"
            except:
                mod_date = "Unknown"
            
            # Status indicators
            has_hash = bool(lora.get('sha256'))
            has_metadata = bool(lora.get('metadata_json'))
            has_civitai = bool(lora.get('civitai_enriched_at'))
            
            hash_indicator = "OK" if has_hash else "X"
            metadata_indicator = "OK" if has_metadata else "X"
            civitai_indicator = "*" if has_civitai else "o"
            
            # Metadata info
            metadata = lora.get('metadata', {})
            model_id = metadata.get('modelId', '')
            description = metadata.get('description', '')
            
            # Civitai data
            civitai_name = lora.get('civitai_model_name', '')
            civitai_description = lora.get('civitai_model_description', '')
            civitai_creator = lora.get('civitai_creator_username', '')
            civitai_downloads = lora.get('civitai_download_count', 0)
            civitai_rating = lora.get('civitai_rating_average', 0.0)
            civitai_rating_count = lora.get('civitai_rating_count', 0)
            civitai_tags = []
            civitai_trigger_words = []
            civitai_images = []
            
            # Parse JSON fields
            try:
                if lora.get('civitai_tags_json'):
                    civitai_tags = json.loads(lora.get('civitai_tags_json', '[]'))
                if lora.get('civitai_trigger_words_json'):
                    civitai_trigger_words = json.loads(lora.get('civitai_trigger_words_json', '[]'))
                if lora.get('civitai_images_json'):
                    civitai_images = json.loads(lora.get('civitai_images_json', '[]'))
            except:
                pass
            
            # Generate Civitai images display
            images_html = ""
            if civitai_images:
                # Show first 3 images in a row
                image_items = []
                for img in civitai_images[:3]:
                    img_url = img.get('url', '')
                    if img_url:
                        image_items.append(f"""
                        <img src="{img_url}" 
                             style="width: 80px; height: 80px; object-fit: cover; border-radius: 4px; 
                                    cursor: pointer; transition: transform 0.2s;"
                             onclick="window.open('{img_url}', '_blank')"
                             onmouseover="this.style.transform='scale(1.1)'"
                             onmouseout="this.style.transform='scale(1)'"
                             title="Click to view full size">
                        """)
                
                if image_items:
                    images_html = f"""
                    <div style='margin-bottom: 8px;'>
                        <div style='font-size: 11px; color: #9ca3af; margin-bottom: 4px;'><strong>- Civitai Images:</strong></div>
                        <div style='display: flex; gap: 8px; flex-wrap: wrap;'>
                            {''.join(image_items)}
                        </div>
                    </div>
                    """

            # Create Lora item HTML
            lora_item = f"""
            <div style='margin-bottom: 20px; padding: 15px; border: 1px solid #444; border-radius: 8px; 
                       background: #2a2a2a; color: #fff;'>
                
                <!-- Header with filename and status -->
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                    <strong style='color: #4ade80; font-size: 15px;'>{civitai_name if civitai_name else lora.get('filename', 'Unknown')}</strong>
                    <div style='display: flex; gap: 8px; align-items: center;'>
                        <span style='font-size: 12px;'>{hash_indicator} Hash</span>
                        <span style='font-size: 12px;'>{metadata_indicator} Metadata</span>
                        <span style='font-size: 12px;'>{civitai_indicator} Civitai</span>
                    </div>
                </div>
                
                <!-- Civitai enriched content -->
                {'''
                <div style='display: flex; gap: 15px; margin-bottom: 12px;'>
                    <!-- Images column -->
                    <div style='flex-shrink: 0;'>
                        ''' + images_html + '''
                    </div>
                    
                    <!-- Info column -->
                    <div style='flex: 1;'>
                        <div style='margin-bottom: 8px;'>
                            <div style='font-size: 11px; color: #9ca3af; margin-bottom: 2px;'><strong>- Creator & Stats:</strong></div>
                            <div style='background: #1a1a1a; padding: 6px; border-radius: 4px; font-size: 11px; border: 1px solid #374151; line-height: 1.4;'>
                                <strong>Creator:</strong> ''' + (civitai_creator if civitai_creator else "Unknown") + '''<br>
                                <strong>Downloads:</strong> ''' + f"{civitai_downloads:,}" + ''' | <strong>Rating:</strong> ''' + ("*" * int(civitai_rating) + "☆" * (5-int(civitai_rating))) + f" ({civitai_rating:.1f}/5, {civitai_rating_count} votes)" + '''<br>
                                <strong>Description:</strong> ''' + ((civitai_description[:100] + "...") if len(civitai_description) > 100 else (civitai_description or "No description")) + '''
                            </div>
                        </div>
                        
                        ''' + ('''
                        <div style='margin-bottom: 8px;'>
                            <div style='font-size: 11px; color: #9ca3af; margin-bottom: 2px;'><strong>- Trigger Words:</strong></div>
                            <div style='background: #1a1a1a; padding: 6px; border-radius: 4px; font-size: 11px; border: 1px solid #374151;'>
                                ''' + (", ".join(civitai_trigger_words[:10]) if civitai_trigger_words else "No trigger words specified") + '''
                            </div>
                        </div>
                        ''' if civitai_trigger_words else "") + '''
                        
                        ''' + ('''
                        <div style='margin-bottom: 8px;'>
                            <div style='font-size: 11px; color: #9ca3af; margin-bottom: 2px;'><strong>- Tags:</strong></div>
                            <div style='background: #1a1a1a; padding: 6px; border-radius: 4px; font-size: 11px; border: 1px solid #374151;'>
                                ''' + (", ".join(['<span style="background: #374151; padding: 1px 4px; border-radius: 2px; margin: 1px;">' + tag + '</span>' for tag in civitai_tags[:8]]) if civitai_tags else "No tags") + '''
                            </div>
                        </div>
                        ''' if civitai_tags else "") + '''
                    </div>
                </div>
                ''' if has_civitai else ""}
                
                <!-- File info -->
                <div style='font-size: 11px; color: #bbb; margin-bottom: 8px;'>
                    <strong>Folder:</strong> <span style='color: #4ade80; font-family: monospace;'>{os.path.dirname(lora.get('relative_path', '')) or '(root)'}</span><br>
                    <strong>File:</strong> {os.path.basename(lora.get('relative_path', 'Unknown'))}<br>
                    <strong>Size:</strong> {size_str} | <strong>Modified:</strong> {mod_date}
                </div>
                
                <!-- Hash info -->
                {'''
                <div style='margin-bottom: 8px;'>
                    <div style='font-size: 11px; color: #9ca3af; margin-bottom: 2px;'><strong>SHA256 Hash:</strong></div>
                    <code style='background: #1a1a1a; padding: 4px 6px; border-radius: 4px; font-size: 10px; 
                                color: #10b981; border: 1px solid #374151; word-break: break-all;'>''' + str(lora.get('sha256')) + '''</code>
                </div>
                ''' if has_hash else ''}
                
                <!-- Metadata info -->
                {'''
                <div style='margin-bottom: 8px;'>
                    <div style='font-size: 11px; color: #9ca3af; margin-bottom: 2px;'><strong>Metadata:</strong></div>
                    <div style='background: #1a1a1a; padding: 6px; border-radius: 4px; font-size: 11px; 
                               border: 1px solid #374151; line-height: 1.4;'>
                        ''' + (f"<strong>Model ID:</strong> {model_id}<br>" if model_id else "") + '''
                        ''' + (f"<strong>Name:</strong> {metadata.get('name', '')}<br>" if metadata.get('name') else "") + '''
                        ''' + (f"<strong>Description:</strong> {description}<br>" if description else "") + '''
                        ''' + (f"<strong>Base Model:</strong> {metadata.get('baseModel', '')}<br>" if metadata.get('baseModel') else "") + '''
                        ''' + (f"<strong>SD Version:</strong> {metadata.get('sd version', '')}" if metadata.get('sd version') else "") + '''
                    </div>
                </div>
                ''' if has_metadata and metadata else ''}
                
            </div>
            """
            
            lora_items.append(lora_item)
        
        # Combine all items
        results_html = f"""
        <div style='max-height: 800px; overflow-y: auto; padding: 10px; background: #0d1117; border-radius: 8px;'>
            <div style='margin-bottom: 15px; padding: 10px; background: #1c2938; border-radius: 6px; 
                       text-align: center; color: #fff; border: 1px solid #444;'>
                <strong>Database Results:</strong> {len(loras)} Loras found
                {f"<br><small style='color: #ccc;'>{query_info}</small>" if query_info else ""}
            </div>
            {''.join(lora_items)}
        </div>
        """
        
        return results_html

    def refresh_lora_list_on_load(self, lora_component):
        """Refresh LORA list when component loads"""
        def update_loras():
            available_loras = self.get_available_loras()
            lora_component.choices = available_loras
            return available_loras
        
        return update_loras

    def apply_random_loras(self, p, selected_loras: List[str], min_strength: float, 
                          max_strength: float, max_count: int):
        """Apply random LORA files to generation with random strengths"""
        if not selected_loras:
            return
        
        # Randomly select LORAs
        num_loras = random.randint(1, min(max_count, len(selected_loras)))
        chosen_loras = random.sample(selected_loras, num_loras)
        
        # Apply each LORA with random strength
        for lora in chosen_loras:
            strength = round(random.uniform(min_strength, max_strength), 2)
            lora_prompt = f"<lora:{lora}:{strength}>"
            
            # Add to positive prompt
            if hasattr(p, 'prompt') and p.prompt:
                p.prompt += f", {lora_prompt}"
            else:
                p.prompt = lora_prompt
                
            print(f"[LORA] Applied: {lora} with strength {strength}")

    def generate_image_html(self, prompt_data):
        """Generate HTML for the image display"""
        image_url = prompt_data.get('image_url', '')
        if image_url:
            # Calculate display size (max 300px width while preserving aspect ratio)
            img_width = prompt_data.get('image_width', 512)
            img_height = prompt_data.get('image_height', 512)
            
            display_width = min(300, img_width)
            display_height = int((display_width / img_width) * img_height) if img_width > 0 else 300
            
            return f"""
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
            return """
            <div style='text-align: center; margin-bottom: 10px; padding: 40px; 
                       background: #2a2a2a; border-radius: 8px; color: #aaa; border: 1px solid #444;'>
                <strong>IMG</strong><br>No image available
            </div>
            """

    def format_prompt_metadata(self, prompt_data):
        """Extract and format metadata from prompt data"""
        # Build comprehensive metadata display
        image_info = []
        if prompt_data.get('image_width') and prompt_data.get('image_height'):
            image_info.append(f"{prompt_data.get('image_width', 0)} x {prompt_data.get('image_height', 0)}px")
        if prompt_data.get('id'):
            image_info.append(f"ID: {prompt_data.get('id')}")
        if prompt_data.get('username'):
            image_info.append(f"- {prompt_data.get('username')}")
        if prompt_data.get('created_at'):
            # Format date nicely
            import datetime
            try:
                dt = datetime.datetime.fromisoformat(prompt_data.get('created_at').replace('Z', '+00:00'))
                formatted_date = dt.strftime('%Y-%m-%d %H:%M')
                image_info.append(f"DATE: {formatted_date}")
            except:
                image_info.append(f"DATE: {prompt_data.get('created_at')}")
        return image_info

    def format_nsfw_indicators(self, prompt_data):
        """Format NSFW and reaction indicators"""
        indicators = []
        if prompt_data.get('nsfw', False):
            nsfw_level = prompt_data.get('nsfw_level', 'Unknown')
            indicators.append(f"<span style='background: #ff6b6b; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px;'>NSFW ({nsfw_level})</span>")
        
        # Reaction stats - show ALL reactions including zeros
        all_reactions = []
        if prompt_data.get('likes', 0) >= 0:
            all_reactions.append(f"LIKE {prompt_data.get('likes', 0)}")
        if prompt_data.get('hearts', 0) >= 0:
            all_reactions.append(f"HEART {prompt_data.get('hearts', 0)}")
        if prompt_data.get('laughs', 0) >= 0:
            all_reactions.append(f"LAUGH {prompt_data.get('laughs', 0)}")
        if prompt_data.get('cries', 0) >= 0:
            all_reactions.append(f"CRY {prompt_data.get('cries', 0)}")
        if prompt_data.get('dislikes', 0) >= 0:
            all_reactions.append(f"DISLIKE {prompt_data.get('dislikes', 0)}")
        if prompt_data.get('comments', 0) >= 0:
            all_reactions.append(f"COMMENT {prompt_data.get('comments', 0)}")
        
        if all_reactions:
            indicators.append(f"<span style='color: #ffd700; font-size: 11px;'>{' '.join(all_reactions)}</span>")
        
        return indicators

    def extract_generation_parameters(self, prompt_data):
        """Extract and categorize generation parameters"""
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
                              'ControlNet', 'TI', 'Hypernet', 'AddNet',
                              'First pass size', 'Schedule type', 'Schedule max sigma',
                              'Schedule min sigma', 'Schedule rho']
            for key in interesting_keys:
                if key in meta and meta[key] and key != 'Lora':  # Skip Lora as it's handled separately
                    extra_params.append(f"<strong>{key}:</strong> {meta[key]}")
            
            # Also capture any other keys that might be interesting
            skip_keys = {'prompt', 'negativePrompt', 'steps', 'sampler', 'cfgScale', 
                       'seed', 'Model', 'clipSkip', 'Size', 'Denoising strength',
                       'Hires upscaler', 'Hires steps', 'Hires upscale', 
                       'Model hash', 'VAE', 'Lora'}  # Skip Lora as it's handled separately
            for key, value in meta.items():
                if key not in skip_keys and value and str(value).strip():
                    extra_params.append(f"<strong>{key}:</strong> {value}")
        
        # Check Lora availability
        lora_info = self.format_lora_availability_info(prompt_data)
        
        return image_metadata, content_info, core_params, advanced_params, hires_params, extra_params, lora_info

    def format_lora_availability_info(self, prompt_data: Dict[str, Any]) -> List[str]:
        """Format Lora availability information for display"""
        try:
            # Safety check for prompt_data
            if not prompt_data or not isinstance(prompt_data, dict):
                return []
            
            # Parse Loras from the prompt
            civitai_loras = self.parse_loras_from_civitai_prompt(prompt_data)
            
            if not civitai_loras:
                return []
            
            # Check availability
            availability_results = self.check_lora_availability(civitai_loras)
            
            if not availability_results:
                return []
            
            lora_info = []
            
            for result in availability_results:
                try:
                    # Safety checks for result structure
                    if not result or not isinstance(result, dict):
                        continue
                    
                    civitai_lora = result.get('civitai_lora', {})
                    if not civitai_lora or not isinstance(civitai_lora, dict):
                        continue
                    
                    lora_name = civitai_lora.get('name', 'Unknown')
                    lora_strength = civitai_lora.get('strength', 1.0)
                    lora_hash = civitai_lora.get('hash', 'No hash')
                    lora_source = civitai_lora.get('source', 'unknown')
                    
                    # Determine status icon and color
                    if result.get('available', False):
                        if result.get('match_method') == 'hash':
                            status_icon = "OK"
                            status_color = "#10b981"  # Green
                            status_text = "Available (Hash Match)"
                        else:
                            status_icon = "!"
                            status_color = "#f59e0b"  # Orange
                            status_text = "Available (Name Match)"
                    else:
                        status_icon = "X"
                        status_color = "#ef4444"  # Red
                        status_text = "Not Found"
                    
                    # Build match details safely
                    match_details = []
                    local_matches = result.get('local_matches', [])
                    if local_matches and isinstance(local_matches, list):
                        for match in local_matches[:2]:  # Show max 2 matches to save space
                            if match and isinstance(match, dict):
                                match_filename = match.get('filename', 'Unknown file')
                                match_type = match.get('match_type', 'unknown')
                                match_details.append(f"{match_filename} ({match_type})")
                        
                        if len(local_matches) > 2:
                            match_details.append(f"... and {len(local_matches) - 2} more")
                    
                    # Create formatted info string
                    lora_detail = f"""
                    <div style='margin: 4px 0; padding: 6px; background: #1a1a1a; border-radius: 4px; border-left: 3px solid {status_color};'>
                        <div style='display: flex; align-items: center; gap: 6px; margin-bottom: 2px;'>
                            <span style='font-size: 12px;'>{status_icon}</span>
                            <strong style='color: {status_color}; font-size: 11px;'>{lora_name}</strong>
                            <span style='color: #888; font-size: 10px;'>({lora_strength})</span>
                        </div>
                        <div style='font-size: 10px; color: #bbb; line-height: 1.3;'>
                            Status: {status_text}<br>
                            Hash: <code style='background: #2a2a2a; padding: 1px 2px; border-radius: 2px;'>{str(lora_hash)[:16]}{'...' if len(str(lora_hash)) > 16 else ''}</code><br>
                            Source: {lora_source}
                            {f'<br>Matches: {", ".join(match_details)}' if match_details else ''}
                        </div>
                    </div>
                    """
                    
                    lora_info.append(lora_detail.strip())
                
                except Exception as inner_e:
                    print(f"[Lora Availability] Error processing individual Lora result: {inner_e}")
                    # Add a fallback entry for this Lora
                    lora_info.append(f"<div style='color: #ef4444; font-size: 11px;'>Error processing Lora: {str(inner_e)}</div>")
            
            return lora_info
            
        except Exception as e:
            print(f"[Lora Availability] Error formatting Lora info: {e}")
            import traceback
            traceback.print_exc()
            return [f"<div style='color: #ef4444; font-size: 11px;'>Error checking Lora availability: {str(e)}</div>"]

    def format_queue_item_html(self, i, prompt_data, current_index, image_html, basic_image_info,
                              image_metadata, content_info, core_params, advanced_params, 
                              hires_params, extra_params, lora_info, indicators, positive_preview, 
                              negative_preview, negative_text):
        """Format HTML for a single queue item"""
        status_icon = "OK" if i < current_index else "~"
        status_text = "Used" if i < current_index else "Pending"
        
        return f"""
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
                        {' | '.join(basic_image_info) if basic_image_info else 'No basic metadata available'}
                    </div>
                    
                    <!-- Image Metadata Section -->
                    {'''
                    <div style='margin-bottom: 10px;'>
                        <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>- Image Metadata:</div>
                        <div style='font-size: 11px; color: #d1d5db; line-height: 1.4; background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #3b82f6;'>
                            ''' + (' - '.join(image_metadata)) + '''
                        </div>
                    </div>
                    ''' if image_metadata else ''}
                    
                    <!-- Content Rating Info -->
                    {'''
                    <div style='margin-bottom: 10px;'>
                        <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>NSFW Content Rating:</div>
                        <div style='font-size: 11px; color: #d1d5db; line-height: 1.4; background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #ef4444;'>
                            ''' + (' - '.join(content_info)) + '''
                        </div>
                    </div>
                    ''' if content_info else ''}
                    
                    <!-- Lora Availability Section -->
                    {'''
                    <div style='margin-bottom: 10px;'>
                        <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>- Lora Availability:</div>
                        <div style='background: #111827; padding: 4px; border-radius: 4px; border-left: 3px solid #8b5cf6;'>
                            ''' + (''.join(lora_info)) + '''
                        </div>
                    </div>
                    ''' if lora_info else ''}
                    
                    <!-- Core Generation Parameters -->
                    {'''
                    <div style='margin-bottom: 10px;'>
                        <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>CFG Core Generation Settings:</div>
                        <div style='font-size: 11px; color: #d1d5db; line-height: 1.4; background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #10b981;'>
                            ''' + (' - '.join(core_params[:4])) + '''
                            ''' + (('<br>' + ' - '.join(core_params[4:])) if len(core_params) > 4 else '') + '''
                        </div>
                    </div>
                    ''' if core_params else ''}
                    
                    <!-- Advanced Parameters -->
                    {'''
                    <div style='margin-bottom: 10px;'>
                        <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>ADV Advanced Settings:</div>
                        <div style='font-size: 11px; color: #d1d5db; line-height: 1.4; background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #f59e0b;'>
                            ''' + (' - '.join(advanced_params[:3])) + '''
                            ''' + (('<br>' + ' - '.join(advanced_params[3:])) if len(advanced_params) > 3 else '') + '''
                        </div>
                    </div>
                    ''' if advanced_params else ''}
                    
                    <!-- Hires/Upscaling Parameters -->
                    {'''
                    <div style='margin-bottom: 10px;'>
                        <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>SEARCH Hires/Upscaling:</div>
                        <div style='font-size: 11px; color: #d1d5db; line-height: 1.4; background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #8b5cf6;'>
                            ''' + (' - '.join(hires_params)) + '''
                        </div>
                    </div>
                    ''' if hires_params else ''}
                    
                    <!-- Extra/Misc Parameters -->
                    {'''
                    <div style='margin-bottom: 8px;'>
                        <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>EXTRA Additional Parameters:</div>
                        <div style='font-size: 11px; color: #d1d5db; line-height: 1.4; background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #6b7280;'>
                            ''' + (' - '.join(extra_params[:4])) + '''
                            ''' + (('<br>' + ' - '.join(extra_params[4:8])) if len(extra_params) > 4 else '') + '''
                            ''' + (('<br>' + ' - '.join(extra_params[8:])) if len(extra_params) > 8 else '') + '''
                        </div>
                    </div>
                    ''' if extra_params else ''}
                </div>
            </div>
            
            <!-- Prompts Section -->
            <div style='margin-bottom: 10px;'>
                <strong style='color: #4ade80; font-size: 13px;'>POS Positive Prompt:</strong><br>
                <span style='background: #1a3b1a; padding: 8px; border-radius: 4px; display: block; margin-top: 4px; line-height: 1.4; color: #e6ffe6; border: 1px solid #2d5a2d; font-size: 12px;'>{positive_preview}</span>
            </div>
            
            <div>
                <strong style='color: #ff6b6b; font-size: 13px;'>NEG Negative Prompt:</strong><br>
                <span style='background: #3b1a1a; padding: 8px; border-radius: 4px; display: block; margin-top: 4px; line-height: 1.4; color: #ffe6e6; border: 1px solid #5a2d2d; font-style: {"italic" if not negative_text else "normal"}; font-size: 12px;'>{negative_preview}</span>
            </div>
        </div>
        """

    def format_search_item_html(self, i, prompt_data, image_html, basic_image_info,
                               image_metadata, content_info, core_params, advanced_params, 
                               hires_params, extra_params, lora_info, indicators, positive_preview, 
                               negative_preview, negative_text):
        """Format HTML for a single search result item (no queue status)"""
        
        # Extract key information for prominent display
        username = prompt_data.get('username', 'Unknown Creator')
        post_id = prompt_data.get('post_id', '')
        image_id = prompt_data.get('id', '')
        created_at = prompt_data.get('created_at', '')
        
        # Format creation date
        formatted_date = 'Unknown Date'
        if created_at:
            try:
                import datetime
                dt = datetime.datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                formatted_date = dt.strftime('%Y-%m-%d %H:%M')
            except:
                formatted_date = created_at
        
        return f"""
        <div style='margin-bottom: 20px; padding: 15px; border: 1px solid #444; border-radius: 8px;
                   background: #2a2a2a; color: #fff;'>
            <!-- Header with creator info and indicators -->
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;'>
                <div style='display: flex; flex-direction: column; gap: 2px;'>
                    <strong style='color: #fff; font-size: 14px;'>Result #{i + 1}</strong>
                    <div style='font-size: 12px; color: #60a5fa;'>
                        <strong>👤 {username}</strong>
                        {f' • 📝 Post {post_id}' if post_id else ''}
                        {f' • 🆔 {image_id}' if image_id else ''}
                    </div>
                    <div style='font-size: 11px; color: #9ca3af;'>📅 {formatted_date}</div>
                </div>
                <div style='display: flex; gap: 8px; align-items: center; flex-wrap: wrap;'>
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
                        {' | '.join(basic_image_info) if basic_image_info else 'No basic metadata available'}
                    </div>
                    
                    <!-- Image Metadata Section -->
                    {'''
                    <div style='margin-bottom: 10px;'>
                        <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>Image Metadata:</div>
                        <div style='font-size: 11px; color: #d1d5db; line-height: 1.4; background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #3b82f6;'>
                            ''' + (' - '.join(image_metadata)) + '''
                        </div>
                    </div>
                    ''' if image_metadata else ''}
                    
                    <!-- Content Rating Info -->
                    {'''
                    <div style='margin-bottom: 10px;'>
                        <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>Content Rating:</div>
                        <div style='font-size: 11px; color: #d1d5db; line-height: 1.4; background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #ef4444;'>
                            ''' + (' - '.join(content_info)) + '''
                        </div>
                    </div>
                    ''' if content_info else ''}
                    
                    <!-- Lora Availability Section -->
                    <div style='margin-bottom: 10px;'>
                        <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>🔧 LoRA Information:</div>
                        <div style='background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #8b5cf6;'>
                            {(''.join(lora_info)) if lora_info else '<div style="color: #6b7280; font-size: 11px; font-style: italic;">No LoRAs detected in this prompt</div>'}
                        </div>
                    </div>
                    
                    <!-- Core Generation Parameters -->
                    {'''
                    <div style='margin-bottom: 10px;'>
                        <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>Core Generation Settings:</div>
                        <div style='font-size: 11px; color: #d1d5db; line-height: 1.4; background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #10b981;'>
                            ''' + (' - '.join(core_params[:4])) + '''
                            ''' + (('<br>' + ' - '.join(core_params[4:])) if len(core_params) > 4 else '') + '''
                        </div>
                    </div>
                    ''' if core_params else ''}
                    
                    <!-- Advanced Parameters -->
                    {'''
                    <div style='margin-bottom: 10px;'>
                        <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>Advanced Settings:</div>
                        <div style='font-size: 11px; color: #d1d5db; line-height: 1.4; background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #f59e0b;'>
                            ''' + (' - '.join(advanced_params[:3])) + '''
                            ''' + (('<br>' + ' - '.join(advanced_params[3:])) if len(advanced_params) > 3 else '') + '''
                        </div>
                    </div>
                    ''' if advanced_params else ''}
                    
                    <!-- Hires/Upscaling Parameters -->
                    {'''
                    <div style='margin-bottom: 10px;'>
                        <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>Hires/Upscaling:</div>
                        <div style='font-size: 11px; color: #d1d5db; line-height: 1.4; background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #8b5cf6;'>
                            ''' + (' - '.join(hires_params)) + '''
                        </div>
                    </div>
                    ''' if hires_params else ''}
                    
                    <!-- Extra/Misc Parameters -->
                    {'''
                    <div style='margin-bottom: 8px;'>
                        <div style='font-size: 12px; color: #9ca3af; margin-bottom: 4px; font-weight: bold;'>Additional Parameters:</div>
                        <div style='font-size: 11px; color: #d1d5db; line-height: 1.4; background: #111827; padding: 6px; border-radius: 4px; border-left: 3px solid #6b7280;'>
                            ''' + (' - '.join(extra_params[:4])) + '''
                            ''' + (('<br>' + ' - '.join(extra_params[4:8])) if len(extra_params) > 4 else '') + '''
                            ''' + (('<br>' + ' - '.join(extra_params[8:])) if len(extra_params) > 8 else '') + '''
                        </div>
                    </div>
                    ''' if extra_params else ''}
                </div>
            </div>
            
            <!-- Prompts Section -->
            <div style='margin-bottom: 10px;'>
                <strong style='color: #4ade80; font-size: 13px;'>Positive Prompt:</strong><br>
                <span style='background: #1a3b1a; padding: 8px; border-radius: 4px; display: block; margin-top: 4px; line-height: 1.4; color: #e6ffe6; border: 1px solid #2d5a2d; font-size: 12px;'>{positive_preview}</span>
            </div>
            
            <div>
                <strong style='color: #ff6b6b; font-size: 13px;'>Negative Prompt:</strong><br>
                <span style='background: #3b1a1a; padding: 8px; border-radius: 4px; display: block; margin-top: 4px; line-height: 1.4; color: #ffe6e6; border: 1px solid #5a2d2d; font-style: {"italic" if not negative_text else "normal"}; font-size: 12px;'>{negative_preview}</span>
            </div>
            
            <!-- Send to StableQueue Button -->
            <div style='margin-top: 15px; text-align: right;'>
                <button onclick='sendToStableQueue({i})' 
                        style='background-color: #2563eb; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: bold; transition: background-color 0.2s;'
                        onmouseover='this.style.backgroundColor="#1d4ed8"'
                        onmouseout='this.style.backgroundColor="#2563eb"'>
                    📤 Send to StableQueue
                </button>
            </div>
        </div>
        """

    def _extract_images_from_models(self, models: List[dict]) -> List[dict]:
        """Extract images from models API response and convert to image format"""
        images = []
        
        for model in models:
            if not model or not isinstance(model, dict):
                continue
                
            model_versions = model.get('modelVersions', [])
            for version in model_versions:
                if not version or not isinstance(version, dict):
                    continue
                    
                version_images = version.get('images', [])
                for image in version_images:
                    if not image or not isinstance(image, dict):
                        continue
                    
                    # Convert model image to standard image format
                    # Add model and version info to meta for prompt extraction
                    image_copy = image.copy()
                    
                    # Ensure meta exists and has the prompt information
                    if 'meta' not in image_copy or not image_copy['meta']:
                        image_copy['meta'] = {}
                    
                    # If the image meta doesn't have a prompt, try to use model/version info
                    if not image_copy['meta'].get('prompt'):
                        # Create a basic prompt from model name and trained words
                        model_name = model.get('name', '')
                        trained_words = version.get('trainedWords', [])
                        if trained_words:
                            image_copy['meta']['prompt'] = ', '.join(trained_words)
                        elif model_name:
                            image_copy['meta']['prompt'] = model_name
                    
                    # Add model metadata for context
                    image_copy['model_name'] = model.get('name', '')
                    image_copy['model_type'] = model.get('type', '')
                    image_copy['version_name'] = version.get('name', '')
                    
                    images.append(image_copy)
        
        print(f"[Search] Converted {len(models)} models to {len(images)} images")
        return images

    def _validate_api_response(self, response) -> dict:
        """Validate and parse the API response"""
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            print(f"Authentication failed - check your API key")
            return None
        elif response.status_code == 403:
            print(f"Access forbidden - API key may not have required permissions")
            return None
        elif response.status_code == 429:
            print(f"Rate limited - please wait before making more requests")
            return None
        else:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response text: {response.text}")
            return None

    def _create_prompt_pair(self, item: dict, meta: dict) -> dict:
        """Create a comprehensive prompt pair dictionary from API item and metadata"""
        positive_prompt = meta.get('prompt', '')
        negative_prompt = meta.get('negativePrompt', '')
        
        # Get image URL from the item itself (not meta)
        image_url = item.get('url', '')
        
        is_nsfw = item.get('nsfw', False)
        nsfw_level = item.get('nsfwLevel', 'None')
        
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
        
        return prompt_pair

# Global reference to the script instance for tab access
script_instance = None

def _create_main_controls_tab():
    """Create the main controls tab UI components"""
    with gr.TabItem("Main Controls"):
        gr.HTML("<h3>Random Prompt Generation</h3>")
        gr.HTML("<p>Fetch prompts from Civitai and manage your prompt queue</p>")
        
        # Filter Note Section
        with gr.Group():
            gr.HTML("<h4>Search Configuration</h4>")
            gr.HTML("<p><em>Configure search filters in the <strong>Prompt Search</strong> tab, then use 'Fetch Prompts' there to load results.</em></p>")
            
            with gr.Row():
                clear_cache_btn = gr.Button("Clear Cache", variant="secondary")
        
        # Custom Prompt Controls
        with gr.Group():
            gr.HTML("<h4>Custom Prompt Settings</h4>")
            custom_prompt_start = gr.Textbox(
                placeholder="Text to add at the beginning of each prompt",
                label="Custom Start Text",
                lines=2
            )
            
            custom_prompt_end = gr.Textbox(
                placeholder="Text to add at the end of each prompt",
                label="Custom End Text",
                lines=2
            )
            
            custom_negative_prompt = gr.Textbox(
                placeholder="Custom negative prompt to add",
                label="Custom Negative Prompt",
                lines=2
            )
        
        # Generate Controls
        with gr.Group():
            gr.HTML("<h4>- Generate Prompts</h4>")
            with gr.Row():
                populate_btn = gr.Button("Populate Prompt Fields", variant="primary", size="lg")
            
            # Status displays
            cache_status = gr.HTML("Cache: No prompts loaded")
            prompt_queue_status = gr.HTML("Queue: No prompts available")
        
        # Hidden bridge textboxes for JavaScript communication
        with gr.Group(visible=False):
            hidden_positive_prompt = gr.Textbox(elem_id="civitai_hidden_positive", label="Hidden Positive")
            hidden_negative_prompt = gr.Textbox(elem_id="civitai_hidden_negative", label="Hidden Negative")
    
    return {
        'clear_cache_btn': clear_cache_btn,
        'custom_prompt_start': custom_prompt_start,
        'custom_prompt_end': custom_prompt_end,
        'custom_negative_prompt': custom_negative_prompt,
        'populate_btn': populate_btn,
        'cache_status': cache_status,
        'prompt_queue_status': prompt_queue_status,
        'hidden_positive_prompt': hidden_positive_prompt,
        'hidden_negative_prompt': hidden_negative_prompt
    }

def _create_search_tab():
    """Create the search tab UI components"""
    with gr.TabItem("Prompt Search"):
        gr.HTML("<h3>Prompt Search & Browse</h3>")
        gr.HTML("<p>Search and browse prompts from Civitai with detailed metadata and Lora availability</p>")
        
        # Filter Settings Section
        with gr.Group():
            gr.HTML("<h4>Search Filters</h4>")
            with gr.Row():
                nsfw_filter = gr.Radio(
                    choices=["Include All", "Exclude NSFW", "Only NSFW"],
                    value="Include All",
                    label="NSFW Filter"
                )
                sort_method = gr.Radio(
                    choices=["Most Reactions", "Most Comments", "Most Collected", "Newest"],
                    value="Most Reactions",
                    label="Sort Method"
                )
                period_filter = gr.Radio(
                    choices=["AllTime", "Year", "Month", "Week", "Day"],
                    value="AllTime",
                    label="Time Period"
                )
            
            with gr.Row():
                keyword_filter = gr.Textbox(
                    placeholder="Enter keywords (comma-separated, optional)",
                    label="Keyword Filter",
                    lines=1,
                    scale=2
                )
                username_filter = gr.Textbox(
                    placeholder="Filter by username (optional)",
                    label="Username Filter",
                    lines=1,
                    scale=2
                )
            
            with gr.Row():
                post_id_filter = gr.Number(
                    placeholder="Enter Post ID (optional)",
                    label="Post ID",
                    value=None,
                    precision=0,
                    scale=1
                )
                model_id_filter = gr.Number(
                    placeholder="Enter Model ID (optional)",
                    label="Model ID",
                    value=None,
                    precision=0,
                    scale=1
                )
                gr.HTML("<div style='flex: 1;'></div>", scale=2)  # Spacer
        
        # Search controls
        with gr.Row():
            refresh_results_btn = gr.Button("🔄", variant="secondary", scale=0, min_width=40)
            fetch_prompts_btn = gr.Button("Search", variant="primary", size="sm")
            load_more_btn = gr.Button("Load More Results", variant="secondary", size="sm")
            clear_results_btn = gr.Button("Clear Results", variant="stop", size="sm")
        
        # Page navigation controls
        with gr.Row():
            prev_page_btn = gr.Button("‹ Previous Page", variant="secondary", size="sm")
            current_page_num = gr.Number(value=1, label="Page", minimum=1, precision=0, scale=1, interactive=True)
            next_page_btn = gr.Button("Next Page ›", variant="secondary", size="sm")
            go_to_page_btn = gr.Button("Go", variant="primary", size="sm", scale=0)
        
        # Search information
        search_info = gr.HTML("Search: No prompts loaded")
        
        # Search results display
        with gr.Group():
            gr.HTML("<h4>Search Results</h4>")
            search_display = gr.HTML(
                value="<div style='padding: 20px; text-align: center; color: #888;'>No prompts found. Fetch some prompts to get started!</div>",
                elem_id="civitai_search_display"
            )
    
    return {
        'nsfw_filter': nsfw_filter,
        'sort_method': sort_method,
        'keyword_filter': keyword_filter,
        'period_filter': period_filter,
        'username_filter': username_filter,
        'post_id_filter': post_id_filter,
        'model_id_filter': model_id_filter,
        'refresh_results_btn': refresh_results_btn,
        'fetch_prompts_btn': fetch_prompts_btn,
        'load_more_btn': load_more_btn,
        'clear_results_btn': clear_results_btn,
        'prev_page_btn': prev_page_btn,
        'current_page_num': current_page_num,
        'next_page_btn': next_page_btn,
        'go_to_page_btn': go_to_page_btn,
        'search_info': search_info,
        'search_display': search_display
    }

def _create_checkpoint_management_tab():
    """Create the Checkpoint management tab UI components"""
    with gr.TabItem("Checkpoint Database"):
        gr.HTML("<h3>Local Checkpoint Database Management</h3>")
        gr.HTML("<p>Manage and browse your local Checkpoint collection with persistent SQLite storage.</p>")
        
        # Database status and controls
        with gr.Row():
            checkpoint_db_stats = gr.HTML("Database: Not loaded")
            checkpoint_refresh_stats_btn = gr.Button("Refresh Stats", variant="secondary", size="sm")
        
        with gr.Row():
            checkpoint_scan_db_btn = gr.Button("Scan Checkpoints", variant="primary", size="sm")
            checkpoint_force_scan_btn = gr.Button("Force Rescan", variant="secondary", size="sm")
            enrich_checkpoints_btn = gr.Button("* Enrich with Civitai", variant="primary", size="sm")
            checkpoint_clear_db_btn = gr.Button("Clear Database", variant="stop", size="sm")
            checkpoint_vacuum_db_btn = gr.Button("Optimize DB", variant="secondary", size="sm")
        

        
        # Search and filter controls
        gr.HTML("<h4>Search & Filter</h4>")
        with gr.Row():
            checkpoint_search_name = gr.Textbox(placeholder="Search by name...", label="Name Filter", scale=2)
            checkpoint_search_hash = gr.Textbox(placeholder="Search by hash...", label="Hash Filter", scale=2)
            checkpoint_search_folder = gr.Textbox(placeholder="Search by folder path...", label="Folder Filter", scale=2)
            checkpoint_search_btn = gr.Button("Search", variant="primary", size="sm", scale=1)
        
        with gr.Row():
            checkpoint_filter_has_metadata = gr.Checkbox(label="Has Metadata", value=False)
            checkpoint_filter_has_hash = gr.Checkbox(label="Has Hash", value=False)
            checkpoint_show_all_btn = gr.Button("Show All", variant="secondary", size="sm")
            checkpoint_refresh_folders_btn = gr.Button("Refresh Folders", variant="secondary", size="sm")
        
        # Folder filter dropdown (Excel-style)
        checkpoint_folder_filter = gr.CheckboxGroup(
            choices=[],
            value=[],
            label="Filter by Folders (uncheck to hide)",
            elem_id="checkpoint_folder_filter",
            interactive=True
        )
        
        # Results display
        checkpoint_results_info = gr.HTML("Results: No search performed")
        checkpoint_display = gr.HTML("<div style='padding: 20px; text-align: center; color: #888;'>No results to display</div>")
    
    return {
        'checkpoint_db_stats': checkpoint_db_stats,
        'checkpoint_refresh_stats_btn': checkpoint_refresh_stats_btn,
        'checkpoint_scan_db_btn': checkpoint_scan_db_btn,
        'checkpoint_force_scan_btn': checkpoint_force_scan_btn,
        'enrich_checkpoints_btn': enrich_checkpoints_btn,
        'checkpoint_clear_db_btn': checkpoint_clear_db_btn,
        'checkpoint_vacuum_db_btn': checkpoint_vacuum_db_btn,
        'checkpoint_search_name': checkpoint_search_name,
        'checkpoint_search_hash': checkpoint_search_hash,
        'checkpoint_search_folder': checkpoint_search_folder,
        'checkpoint_search_btn': checkpoint_search_btn,
        'checkpoint_filter_has_metadata': checkpoint_filter_has_metadata,
        'checkpoint_filter_has_hash': checkpoint_filter_has_hash,
        'checkpoint_show_all_btn': checkpoint_show_all_btn,
        'checkpoint_refresh_folders_btn': checkpoint_refresh_folders_btn,
        'checkpoint_folder_filter': checkpoint_folder_filter,
        'checkpoint_results_info': checkpoint_results_info,
        'checkpoint_display': checkpoint_display
    }

def _create_lora_management_tab():
    """Create the Lora management tab UI components"""
    with gr.TabItem("Lora Database"):
        gr.HTML("<h3>Local Lora Database Management</h3>")
        gr.HTML("<p>Manage and browse your local Lora collection with persistent SQLite storage.</p>")
        
        # Database status and controls
        with gr.Row():
            db_stats = gr.HTML("Database: Not loaded")
            refresh_stats_btn = gr.Button("Refresh Stats", variant="secondary", size="sm")
        
        with gr.Row():
            scan_db_btn = gr.Button("Scan Loras", variant="primary", size="sm")
            force_scan_btn = gr.Button("Force Rescan", variant="secondary", size="sm")
            enrich_loras_btn = gr.Button("* Enrich with Civitai", variant="primary", size="sm")
            clear_db_btn = gr.Button("Clear Database", variant="stop", size="sm")
            vacuum_db_btn = gr.Button("Optimize DB", variant="secondary", size="sm")
        

        
        # Search and filter controls
        gr.HTML("<h4>Search & Filter</h4>")
        with gr.Row():
            search_name = gr.Textbox(placeholder="Search by name...", label="Name Filter", scale=2)
            search_hash = gr.Textbox(placeholder="Search by hash...", label="Hash Filter", scale=2)
            search_folder = gr.Textbox(placeholder="Search by folder path...", label="Folder Filter", scale=2)
            search_btn = gr.Button("Search", variant="primary", size="sm", scale=1)
        
        with gr.Row():
            filter_has_metadata = gr.Checkbox(label="Has Metadata", value=False)
            filter_has_hash = gr.Checkbox(label="Has Hash", value=False)
            show_all_btn = gr.Button("Show All", variant="secondary", size="sm")
            refresh_folders_btn = gr.Button("Refresh Folders", variant="secondary", size="sm")
        
        # Folder filter dropdown (Excel-style)
        folder_filter = gr.CheckboxGroup(
            choices=[],
            value=[],
            label="Filter by Folders (uncheck to hide)",
            elem_id="lora_folder_filter",
            interactive=True
        )
        
        # Results display
        results_info = gr.HTML("Results: No search performed")
        lora_display = gr.HTML("<div style='padding: 20px; text-align: center; color: #888;'>No results to display</div>")
    
    return {
        'db_stats': db_stats,
        'refresh_stats_btn': refresh_stats_btn,
        'scan_db_btn': scan_db_btn,
        'force_scan_btn': force_scan_btn,
        'enrich_loras_btn': enrich_loras_btn,
        'clear_db_btn': clear_db_btn,
        'vacuum_db_btn': vacuum_db_btn,
        'search_name': search_name,
        'search_hash': search_hash,
        'search_folder': search_folder,
        'search_btn': search_btn,
        'filter_has_metadata': filter_has_metadata,
        'filter_has_hash': filter_has_hash,
        'show_all_btn': show_all_btn,
        'refresh_folders_btn': refresh_folders_btn,
        'folder_filter': folder_filter,
        'results_info': results_info,
        'lora_display': lora_display
    }

def _create_event_handlers():
    """Create event handlers for the UI components"""
    
    def test_api_connection():
        import modules.shared as shared
        api_key = getattr(shared.opts, 'civitai_api_key', '')
        result = script_instance.test_civitai_api(api_key)
        return result
    
    def refresh_api_key_status():
        """Refresh and display current API key status"""
        import modules.shared as shared
        api_key = getattr(shared.opts, 'civitai_api_key', '')
        
        if api_key:
            masked_key = '***' + api_key[-4:] if len(api_key) > 4 else '***'
            status = f"<span style='color: green;'>API Key configured: {masked_key}</span>"
        else:
            status = "<span style='color: orange;'>No API key configured</span>"
        
        return status
    
    def get_current_paths():
        """Get current directory path configurations"""
        import modules.shared as shared
        
        lora_path = getattr(shared.opts, 'civitai_lora_path', '')
        checkpoint_path = getattr(shared.opts, 'civitai_checkpoint_path', '')
        
        lora_status = f"<strong>Lora Path:</strong> {lora_path}" if lora_path else "<strong>Lora Path:</strong> Auto-detected"
        checkpoint_status = f"<strong>Checkpoint Path:</strong> {checkpoint_path}" if checkpoint_path else "<strong>Checkpoint Path:</strong> Auto-detected"
        
        return lora_status, checkpoint_status
    
    def refresh_lora_list():
        loras = script_instance.get_available_loras()
        return gr.CheckboxGroup.update(choices=loras)
    
    def scan_local_loras():
        """Scan local Loras and return status"""
        try:
            script_instance.scan_local_loras(force_refresh=False)
            count = script_instance.get_lora_count()
            status_msg = f"Lora database: {count} files scanned"
            return status_msg
        except Exception as e:
            return f"Lora database: Error - {str(e)}"
    
    def scan_and_refresh_folders():
        """Scan local Loras and refresh folder choices"""
        try:
            script_instance.scan_local_loras(force_refresh=False)
            count = script_instance.get_lora_count()
            status_msg = f"Lora database: {count} files scanned"
            
            # Get updated folder choices
            folders = script_instance.get_folder_choices()
            folder_update = gr.CheckboxGroup.update(choices=folders, value=folders)
            
            return status_msg, folder_update
        except Exception as e:
            error_msg = f"Lora database: Error - {str(e)}"
            return error_msg, gr.CheckboxGroup.update(choices=[], value=[])
    
    def force_rescan_loras():
        """Force rescan of local Loras"""
        try:
            script_instance.scan_local_loras(force_refresh=True)
            count = script_instance.get_lora_count()
            status_msg = f"Lora database: {count} files rescanned (forced)"
            return status_msg
        except Exception as e:
            return f"Lora database: Error - {str(e)}"
    
    def force_rescan_and_refresh_folders():
        """Force rescan of local Loras and refresh folder choices"""
        try:
            script_instance.scan_local_loras(force_refresh=True)
            count = script_instance.get_lora_count()
            status_msg = f"Lora database: {count} files rescanned (forced)"
            
            # Get updated folder choices
            folders = script_instance.get_folder_choices()
            folder_update = gr.CheckboxGroup.update(choices=folders, value=folders)
            
            return status_msg, folder_update
        except Exception as e:
            error_msg = f"Lora database: Error - {str(e)}"
            return error_msg, gr.CheckboxGroup.update(choices=[], value=[])
    
    def clear_lora_cache():
        """Clear the Lora database"""
        if script_instance.clear_database():
            return "Lora database: Cleared successfully"
        else:
            return "Lora database: Error clearing database"
    
    def vacuum_database():
        """Vacuum the database for optimization"""
        if script_instance.vacuum_database():
            return "Database vacuumed successfully"
        else:
            return "Error vacuuming database"
    
    def backup_database():
        """Create a backup of the database"""
        result = script_instance.backup_database()
        if result.get("success"):
            return result.get("message", "Backup created successfully")
        else:
            return f"Backup failed: {result.get('message', 'Unknown error')}"
    
    def list_database_backups():
        """List all available database backups"""
        try:
            backups = script_instance.list_database_backups()
            
            if not backups:
                return """
                <div style='padding: 20px; text-align: center; color: #888; background: #1a1a1a; border-radius: 8px;'>
                    <h3>No Backups Found</h3>
                    <p>No database backups exist yet. Create a backup first.</p>
                </div>
                """
            
            backup_items = []
            backup_items.append(f"""
            <div style='background: #1a1a1a; padding: 15px; border-radius: 8px; font-size: 13px; color: #ccc;'>
                <h4 style='color: #fff; margin-top: 0;'>Available Database Backups ({len(backups)} total)</h4>
            """)
            
            for backup in backups:
                size_str = f"{backup['size'] / (1024**2):.1f} MB" if backup['size'] > 1024**2 else f"{backup['size'] / 1024:.1f} KB"
                
                backup_items.append(f"""
                <div style='margin: 10px 0; padding: 10px; background: #2a2a2a; border-radius: 6px; border-left: 3px solid #4CAF50;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <strong style='color: #4CAF50;'>{backup['filename']}</strong><br>
                            <span style='color: #aaa;'>Created: {backup['modified_date']}</span>
                        </div>
                        <div style='text-align: right; color: #ccc;'>
                            <strong>{size_str}</strong>
                        </div>
                    </div>
                </div>
                """)
            
            backup_items.append("</div>")
            return ''.join(backup_items)
            
        except Exception as e:
            return f"<div style='color: #ff6b6b;'>Error listing backups: {str(e)}</div>"
    
    def restore_database_from_backup(backup_name):
        """Restore database from specified backup"""
        if not backup_name:
            return "Please enter a backup filename to restore"
        
        if not backup_name.strip():
            return "Please enter a valid backup filename"
        
        result = script_instance.restore_database(backup_name.strip())
        if result.get("success"):
            message = result.get("message", "Database restored successfully")
            if result.get("current_backup"):
                message += f"\n\nPrevious database backed up as: {result.get('current_backup')}"
            return message
        else:
            return f"Restore failed: {result.get('message', 'Unknown error')}"
    
    def get_database_stats():
        """Get and format database statistics"""
        stats = script_instance.get_database_stats()
        
        if not stats:
            return "No statistics available"
        
        try:
            total_loras = stats.get('total_loras', 0)
            with_hash = stats.get('with_hash', 0)
            with_metadata = stats.get('with_metadata', 0)
            total_size = stats.get('total_size', 0)
            last_scan = stats.get('last_scan')
            
            # Format file size
            if total_size > 1024**3:
                size_str = f"{total_size / (1024**3):.1f} GB"
            elif total_size > 1024**2:
                size_str = f"{total_size / (1024**2):.1f} MB"
            else:
                size_str = f"{total_size / 1024:.1f} KB"
            
            # Format last scan
            scan_info = "Never"
            if last_scan:
                try:
                    import datetime
                    scan_time = datetime.datetime.fromtimestamp(last_scan['scan_date'])
                    scan_info = f"{scan_time.strftime('%Y-%m-%d %H:%M')} ({last_scan.get('scan_type', 'unknown')})"
                except:
                    scan_info = "Error parsing date"
            
            stats_html = f"""
            <div style='background: #1a1a1a; padding: 15px; border-radius: 8px; font-size: 13px; color: #ccc;'>
                <h4 style='color: #fff; margin-top: 0;'>STATS Database Statistics</h4>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
                    <div>
                        <strong>Total Loras:</strong> {total_loras}<br>
                        <strong>With Hashes:</strong> {with_hash}<br>
                        <strong>With Metadata:</strong> {with_metadata}
                    </div>
                    <div>
                        <strong>Total Size:</strong> {size_str}<br>
                        <strong>Last Scan:</strong> {scan_info}
                    </div>
                </div>
            </div>
            """
            return stats_html
            
        except Exception as e:
            return f"Error formatting statistics: {e}"
    
    def search_loras():
        """Search Loras in database"""
        try:
            # For now, just show all Loras
            results = script_instance.search_loras_db(limit=100)
            results_html = script_instance.format_lora_database_display(results, "Showing recent Loras")
            return results_html
        except Exception as e:
            return f"<div style='color: #ff6b6b;'>Search error: {str(e)}</div>"
    
    def search_loras_with_filters(name_query, hash_query, folder_query, has_metadata, has_hash, selected_folders):
        """Search Loras in database with all filters"""
        try:
            # Perform search with all filters
            results = script_instance.search_loras_db(
                name_query=name_query or "",
                hash_query=hash_query or "",
                folder_query=folder_query or "",
                has_metadata=has_metadata,
                has_hash=has_hash,
                selected_folders=selected_folders if selected_folders else None,
                limit=200
            )
            
            # Build query info
            query_parts = []
            if name_query:
                query_parts.append(f"name contains '{name_query}'")
            if hash_query:
                query_parts.append(f"hash contains '{hash_query}'")
            if folder_query:
                query_parts.append(f"folder contains '{folder_query}'")
            if selected_folders:
                if len(selected_folders) == 1:
                    query_parts.append(f"folder: {selected_folders[0]}")
                else:
                    query_parts.append(f"folders: {', '.join(selected_folders[:3])}{'...' if len(selected_folders) > 3 else ''}")
            if has_metadata:
                query_parts.append("has metadata")
            if has_hash:
                query_parts.append("has hash")
            
            query_info = "Filters: " + ", ".join(query_parts) if query_parts else "No filters applied"
            
            # Format results
            results_html = script_instance.format_lora_database_display(results, query_info)
            
            return results_html
            
        except Exception as e:
            error_msg = f"Search error: {str(e)}"
            error_html = f"""
            <div style='padding: 20px; text-align: center; color: #ff6b6b; background: #2a1a1a; border-radius: 8px;'>
                <h3>Search Error</h3>
                <p>{error_msg}</p>
            </div>
            """
            return error_html
    
    def show_all_loras():
        """Show all Loras in database"""
        try:
            results = script_instance.search_loras_db(limit=200)
            results_html = script_instance.format_lora_database_display(results, "Showing all Loras (limited to 200)")
            return results_html
        except Exception as e:
            return f"<div style='color: #ff6b6b;'>Error loading Loras: {str(e)}</div>"
    
    def refresh_folder_choices():
        """Refresh the folder filter choices and select all by default"""
        try:
            folders = script_instance.get_folder_choices()
            # Return both choices and values (all selected by default)
            return gr.CheckboxGroup.update(choices=folders, value=folders)
        except Exception as e:
            print(f"Error refreshing folder choices: {e}")
            return gr.CheckboxGroup.update(choices=[], value=[])
    
    def show_folders():
        """Show all unique folders in the database"""
        try:
            with script_instance.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        CASE 
                            WHEN instr(relative_path, '/') > 0 
                            THEN substr(relative_path, 1, instr(relative_path, '/') - 1)
                            ELSE '(root)'
                        END as folder_path,
                        COUNT(*) as file_count
                    FROM loras 
                    GROUP BY folder_path
                    ORDER BY folder_path
                ''')
                
                folders = cursor.fetchall()
                
                if not folders:
                    return """
                    <div style='padding: 30px; text-align: center; color: #ccc; background: #1a1a1a; border-radius: 8px;'>
                        <h3>No Folders Found</h3>
                        <p>No Loras in database. Scan your Lora directory first.</p>
                    </div>
                    """
                
                folder_items = []
                total_folders = len(folders)
                total_files = sum(folder[1] for folder in folders)
                
                # Header
                folder_items.append(f"""
                <div style='margin-bottom: 15px; padding: 10px; background: #1c2938; border-radius: 6px; 
                           text-align: center; color: #fff; border: 1px solid #444;'>
                    <strong>Folder Structure:</strong> {total_folders} folders, {total_files} files total
                </div>
                """)
                
                # Folder list
                for folder_path, file_count in folders:
                    folder_item = f"""
                    <div style='margin-bottom: 8px; padding: 8px 12px; border: 1px solid #444; border-radius: 6px; 
                               background: #2a2a2a; color: #fff; display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <span style='color: #4ade80; font-family: monospace; font-size: 13px;'>{folder_path}</span>
                        </div>
                        <div>
                            <span style='color: #ffd700; font-size: 11px; background: #333; padding: 2px 6px; border-radius: 3px;'>
                                {file_count} files
                            </span>
                        </div>
                    </div>
                    """
                    folder_items.append(folder_item)
                
                results_html = f"""
                <div style='max-height: 600px; overflow-y: auto; padding: 10px; background: #0d1117; border-radius: 8px;'>
                    {''.join(folder_items)}
                </div>
                """
                
                return results_html
                
        except Exception as e:
            return f"""
            <div style='padding: 20px; text-align: center; color: #ff6b6b; background: #2a1a1a; border-radius: 8px;'>
                <h3>Error Loading Folders</h3>
                <p>{str(e)}</p>
            </div>
            """
    
    # Checkpoint Event Handlers
    def scan_local_checkpoints():
        """Scan local Checkpoints and return status"""
        try:
            script_instance.scan_local_checkpoints(force_refresh=False)
            count = script_instance.get_checkpoint_count()
            status_msg = f"Checkpoint database: {count} files scanned"
            return status_msg
        except Exception as e:
            return f"Checkpoint database: Error - {str(e)}"
    
    def scan_checkpoints_and_refresh_folders():
        """Scan local Checkpoints and refresh folder choices"""
        try:
            script_instance.scan_local_checkpoints(force_refresh=False)
            count = script_instance.get_checkpoint_count()
            status_msg = f"Checkpoint database: {count} files scanned"
            
            # Get updated folder choices
            folders = script_instance.get_checkpoint_folder_choices()
            folder_update = gr.CheckboxGroup.update(choices=folders, value=folders)
            
            return status_msg, folder_update
        except Exception as e:
            error_msg = f"Checkpoint database: Error - {str(e)}"
            return error_msg, gr.CheckboxGroup.update(choices=[], value=[])
    
    def force_rescan_checkpoints():
        """Force rescan of local Checkpoints"""
        try:
            script_instance.scan_local_checkpoints(force_refresh=True)
            count = script_instance.get_checkpoint_count()
            status_msg = f"Checkpoint database: {count} files rescanned (forced)"
            return status_msg
        except Exception as e:
            return f"Checkpoint database: Error - {str(e)}"
    
    def force_rescan_checkpoints_and_refresh_folders():
        """Force rescan of local Checkpoints and refresh folder choices"""
        try:
            script_instance.scan_local_checkpoints(force_refresh=True)
            count = script_instance.get_checkpoint_count()
            status_msg = f"Checkpoint database: {count} files rescanned (forced)"
            
            # Get updated folder choices
            folders = script_instance.get_checkpoint_folder_choices()
            folder_update = gr.CheckboxGroup.update(choices=folders, value=folders)
            
            return status_msg, folder_update
        except Exception as e:
            error_msg = f"Checkpoint database: Error - {str(e)}"
            return error_msg, gr.CheckboxGroup.update(choices=[], value=[])
    
    def clear_checkpoint_cache():
        """Clear the Checkpoint database"""
        try:
            with script_instance.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM checkpoints')
                conn.commit()
                return "Checkpoint database: Cleared successfully"
        except Exception as e:
            return f"Checkpoint database: Error clearing database - {str(e)}"
    
    def get_checkpoint_database_stats():
        """Get and format checkpoint database statistics"""
        try:
            with script_instance.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Get basic stats
                cursor.execute('SELECT COUNT(*) as total FROM checkpoints')
                total_checkpoints = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) as with_hash FROM checkpoints WHERE sha256 IS NOT NULL AND sha256 != ""')
                with_hash = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) as with_metadata FROM checkpoints WHERE metadata_json IS NOT NULL AND metadata_json != ""')
                with_metadata = cursor.fetchone()[0]
                
                cursor.execute('SELECT SUM(file_size) as total_size FROM checkpoints')
                total_size = cursor.fetchone()[0] or 0
                
                # Get last scan info
                cursor.execute('SELECT * FROM scan_stats WHERE content_type = "checkpoints" ORDER BY scan_date DESC LIMIT 1')
                last_scan = cursor.fetchone()
                
                # Format file size
                if total_size > 1024**3:
                    size_str = f"{total_size / (1024**3):.1f} GB"
                elif total_size > 1024**2:
                    size_str = f"{total_size / (1024**2):.1f} MB"
                else:
                    size_str = f"{total_size / 1024:.1f} KB"
                
                # Format last scan
                scan_info = "Never"
                if last_scan:
                    try:
                        import datetime
                        scan_time = datetime.datetime.fromtimestamp(last_scan['scan_date'])
                        scan_info = f"{scan_time.strftime('%Y-%m-%d %H:%M')} ({last_scan.get('scan_type', 'unknown')})"
                    except:
                        scan_info = "Error parsing date"
                
                stats_html = f"""
                <div style='background: #1a1a1a; padding: 15px; border-radius: 8px; font-size: 13px; color: #ccc;'>
                    <h4 style='color: #fff; margin-top: 0;'>- Checkpoint Database Statistics</h4>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
                        <div>
                            <strong>Total Checkpoints:</strong> {total_checkpoints}<br>
                            <strong>With Hashes:</strong> {with_hash}<br>
                            <strong>With Metadata:</strong> {with_metadata}
                        </div>
                        <div>
                            <strong>Total Size:</strong> {size_str}<br>
                            <strong>Last Scan:</strong> {scan_info}
                        </div>
                    </div>
                </div>
                """
                return stats_html
                
        except Exception as e:
            return f"Error formatting checkpoint statistics: {e}"
    
    def search_checkpoints_with_filters(name_query, hash_query, folder_query, has_metadata, has_hash, selected_folders):
        """Search Checkpoints in database with all filters"""
        try:
            # Perform search with all filters
            results = script_instance.search_checkpoints_db(
                name_query=name_query or "",
                hash_query=hash_query or "",
                folder_query=folder_query or "",
                has_metadata=has_metadata,
                has_hash=has_hash,
                selected_folders=selected_folders if selected_folders else None,
                limit=200
            )
            
            # Build query info
            query_parts = []
            if name_query:
                query_parts.append(f"name contains '{name_query}'")
            if hash_query:
                query_parts.append(f"hash contains '{hash_query}'")
            if folder_query:
                query_parts.append(f"folder contains '{folder_query}'")
            if selected_folders:
                if len(selected_folders) == 1:
                    query_parts.append(f"folder: {selected_folders[0]}")
                else:
                    query_parts.append(f"folders: {', '.join(selected_folders[:3])}{'...' if len(selected_folders) > 3 else ''}")
            if has_metadata:
                query_parts.append("has metadata")
            if has_hash:
                query_parts.append("has hash")
            
            query_info = "Filters: " + ", ".join(query_parts) if query_parts else "No filters applied"
            
            # Format results
            results_html = script_instance.format_checkpoint_database_display(results, query_info)
            
            return results_html
            
        except Exception as e:
            error_msg = f"Search error: {str(e)}"
            error_html = f"""
            <div style='padding: 20px; text-align: center; color: #ff6b6b; background: #2a1a1a; border-radius: 8px;'>
                <h3>Search Error</h3>
                <p>{error_msg}</p>
            </div>
            """
            return error_html
    
    def show_all_checkpoints():
        """Show all Checkpoints in database"""
        try:
            results = script_instance.search_checkpoints_db(limit=200)
            results_html = script_instance.format_checkpoint_database_display(results, "Showing all Checkpoints (limited to 200)")
            return results_html
        except Exception as e:
            return f"<div style='color: #ff6b6b;'>Error loading Checkpoints: {str(e)}</div>"
    
    def refresh_checkpoint_folder_choices():
        """Refresh the checkpoint folder filter choices and select all by default"""
        try:
            folders = script_instance.get_checkpoint_folder_choices()
            # Return both choices and values (all selected by default)
            return gr.CheckboxGroup.update(choices=folders, value=folders)
        except Exception as e:
            print(f"Error refreshing checkpoint folder choices: {e}")
            return gr.CheckboxGroup.update(choices=[], value=[])

    def enrich_loras_with_civitai():
        """Enrich Loras with Civitai data"""
        try:
            print(f"[Civitai Enrichment] Starting Lora enrichment...")
            results = script_instance.enrich_models_with_civitai_data('loras')
            
            if results['total_processed'] == 0:
                return "* No Loras found with hashes to enrich. Scan your Loras first!"
            
            enriched = results['enriched']
            errors = results['errors']
            skipped = results['skipped']
            total = results['total_processed']
            
            status_msg = f"* Civitai Enrichment Complete: {enriched}/{total} enriched, {skipped} skipped, {errors} errors"
            print(f"[Civitai Enrichment] {status_msg}")
            return status_msg
            
        except Exception as e:
            error_msg = f"* Civitai Enrichment Error: {str(e)}"
            print(f"[Civitai Enrichment] {error_msg}")
            return error_msg

    def enrich_checkpoints_with_civitai():
        """Enrich Checkpoints with Civitai data"""
        try:
            print(f"[Civitai Enrichment] Starting Checkpoint enrichment...")
            results = script_instance.enrich_models_with_civitai_data('checkpoints')
            
            if results['total_processed'] == 0:
                return "* No Checkpoints found with hashes to enrich. Scan your Checkpoints first!"
            
            enriched = results['enriched']
            errors = results['errors']
            skipped = results['skipped']
            total = results['total_processed']
            
            status_msg = f"* Civitai Enrichment Complete: {enriched}/{total} enriched, {skipped} skipped, {errors} errors"
            print(f"[Civitai Enrichment] {status_msg}")
            return status_msg
            
        except Exception as e:
            error_msg = f"* Civitai Enrichment Error: {str(e)}"
            print(f"[Civitai Enrichment] {error_msg}")
            return error_msg

    # Main Controls Event Handlers
    def clear_prompt_cache():
        """Clear the prompt cache"""
        script_instance.cached_prompts = []
        script_instance.prompt_queue = []
        script_instance.queue_index = 0
        cache_status = "Cache: Cleared"
        queue_status = "Queue: Empty"
        return cache_status, queue_status
    
    def fetch_new_prompts(nsfw_filter, keyword_filter, sort_method, period_filter, username_filter, post_id_filter, model_id_filter):
        """Fetch new prompts from Civitai"""
        try:
            prompts = script_instance.fetch_civitai_prompts(nsfw_filter, keyword_filter, sort_method, 
                                                           period_filter, username_filter, post_id_filter, model_id_filter,
                                                           limit=100, is_fetch_more=False)
            cache_status = f"Cache: {len(script_instance.cached_prompts)} prompts loaded"
            queue_status = f"Queue: {len(script_instance.prompt_queue)} prompts, index: {script_instance.queue_index}"
            return cache_status, queue_status, "", ""
        except Exception as e:
            error_msg = f"Error fetching prompts: {str(e)}"
            return error_msg, "Queue: Error", "", ""
    
    def get_prompts_and_update_queue(custom_start, custom_end, custom_negative):
        """Generate prompts and update search display"""
        try:
            status_msg, positive, negative = script_instance.generate_prompts_with_outputs(custom_start, custom_end, custom_negative)
            
            # Update search display
            search_status, search_display_content, page, total_pages = refresh_search_display()
            
            return status_msg, positive, negative, search_status, search_display_content
        except Exception as e:
            error_msg = f"Error generating prompts: {str(e)}"
            return error_msg, "", "", "Search: Error", "Error loading search results"
    
    # Queue Event Handlers
    def refresh_search_display(page: int = 1):
        """Refresh the search results display with pagination support"""
        try:
            results_length = len(script_instance.prompt_queue)
            
            if results_length == 0:
                search_status = "Search: No results"
                search_display_content = "<div style='padding: 20px; text-align: center; color: #888;'>No prompts found. Fetch some prompts to get started!</div>"
                return search_status, search_display_content, page, 1
            
            search_status = f"Search: {results_length} results found"
            
            # Pagination logic
            items_per_page = 10
            total_pages = (results_length + items_per_page - 1) // items_per_page  # Ceiling division
            page = max(1, min(page, total_pages))  # Clamp page to valid range
            start_index = (page - 1) * items_per_page
            end_index = min(start_index + items_per_page, results_length)
            
            display_items = []
            
            # Add pagination header with enhanced clickable page numbers
            if total_pages > 1:
                # Create page navigation with JavaScript functionality
                page_buttons = []
                
                # Create elegant page indicators (status only - no fake interactivity)
                start_page = max(1, page - 3)
                end_page = min(total_pages, page + 3)
                
                # First page + ellipsis if needed
                if start_page > 1:
                    page_buttons.append(f'<span style="padding: 5px 10px; margin: 0 2px; background: #444; color: #aaa; border: 1px solid #555; border-radius: 4px; display: inline-block; font-size: 13px;">1</span>')
                    if start_page > 2:
                        page_buttons.append('<span style="margin: 0 5px; color: #888; font-weight: bold;">...</span>')
                
                # Page number indicators
                for p in range(start_page, end_page + 1):
                    if p == page:
                        page_buttons.append(f'<span style="padding: 5px 10px; margin: 0 2px; background: #007acc; color: white; border: 1px solid #0099ff; border-radius: 4px; font-weight: bold; box-shadow: 0 2px 4px rgba(0,122,204,0.3); display: inline-block; font-size: 13px;">{p}</span>')
                    else:
                        page_buttons.append(f'<span style="padding: 5px 10px; margin: 0 2px; background: #444; color: #aaa; border: 1px solid #555; border-radius: 4px; display: inline-block; font-size: 13px;">{p}</span>')
                
                # Last page + ellipsis if needed
                if end_page < total_pages:
                    if end_page < total_pages - 1:
                        page_buttons.append('<span style="margin: 0 5px; color: #888; font-weight: bold;">...</span>')
                    page_buttons.append(f'<span style="padding: 5px 10px; margin: 0 2px; background: #444; color: #aaa; border: 1px solid #555; border-radius: 4px; display: inline-block; font-size: 13px;">{total_pages}</span>')
                
                pagination_html = ' '.join(page_buttons)
                
                display_items.append(f"""
                <div style='padding: 20px; text-align: center; color: #ccc; background: linear-gradient(135deg, #1a1a1a, #2a2a2a); border-radius: 8px; margin-bottom: 20px; border: 1px solid #444; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
                    <div style='margin-bottom: 15px;'>
                        <div style='font-size: 16px; font-weight: bold; margin-bottom: 5px;'>
                            <span style='color: #007acc;'>{results_length}</span> search results found
                        </div>
                        <div style='color: #888; font-size: 14px;'>
                            Showing results {start_index + 1}-{end_index} • Page {page} of {total_pages}
                        </div>
                    </div>
                    <div style='margin: 15px 0;'>
                        {pagination_html}
                    </div>
                    <div style='font-size: 12px; color: #666; margin-top: 10px;'>
                        📍 <strong>Page Status:</strong> You're viewing page {page} of {total_pages}. Use the navigation controls above to browse.
                    </div>
                </div>
                """)
            
            # Generate items for current page
            for i in range(start_index, end_index):
                prompt_data = script_instance.prompt_queue[i]
                
                # Generate all the formatting data
                image_html = script_instance.generate_image_html(prompt_data)
                basic_image_info = script_instance.format_prompt_metadata(prompt_data)
                indicators = script_instance.format_nsfw_indicators(prompt_data)
                
                image_metadata, content_info, core_params, advanced_params, hires_params, extra_params, lora_info = script_instance.extract_generation_parameters(prompt_data)
                
                # Format prompts for display
                positive_text = prompt_data.get('positive', '')
                negative_text = prompt_data.get('negative', '')
                
                positive_preview = positive_text[:200] + '...' if len(positive_text) > 200 else positive_text
                negative_preview = negative_text[:100] + '...' if len(negative_text) > 100 else (negative_text if negative_text else "(No negative prompt)")
                
                # Format the complete item (no current_index needed for search results)
                item_html = script_instance.format_search_item_html(
                    i, prompt_data, image_html, basic_image_info,
                    image_metadata, content_info, core_params, advanced_params,
                    hires_params, extra_params, lora_info, indicators,
                    positive_preview, negative_preview, negative_text
                )
                
                display_items.append(item_html)
            
            # Add pagination footer with enhanced styling
            if total_pages > 1:
                display_items.append(f"""
                <div style='padding: 15px; text-align: center; color: #ccc; background: linear-gradient(135deg, #1a1a1a, #2a2a2a); border-radius: 8px; margin-top: 20px; border: 1px solid #444; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
                    <div style='margin-bottom: 8px; font-size: 14px; color: #888;'>
                        Page <span style='color: #007acc; font-weight: bold;'>{page}</span> of <span style='color: #007acc; font-weight: bold;'>{total_pages}</span> — {end_index - start_index} items on this page
                    </div>
                    <div style='font-size: 12px; color: #666;'>
                        Page indicator • Use Previous/Next buttons or Page input above for navigation
                    </div>
                </div>
                """)
            
            search_display_content = ''.join(display_items)
            
            return search_status, search_display_content, page, total_pages
            
        except Exception as e:
            print(f"[Search Display] Error: {e}")
            return f"Search: Error - {str(e)}", "<div style='color: #ff6b6b;'>Error loading search results</div>", 1, 1
    
    def clear_search_results():
        """Clear search results and update displays"""
        script_instance.cached_prompts = []
        script_instance.prompt_queue = []
        
        cache_status = "Cache: Cleared"
        search_status = "Search: No results"
        search_info = "Search: Results cleared"
        search_display = "<div style='padding: 20px; text-align: center; color: #888;'>Search results cleared!</div>"
        
        return cache_status, search_status, search_info, search_display, 1
    
    def fetch_prompts_smart(nsfw_filter, keyword_filter, sort_method, period_filter, username_filter, post_id_filter, model_id_filter):
        """Search function: always starts fresh with new search criteria"""
        try:
            # Always start fresh for searches - clear existing results
            print(f"[Search] Starting fresh search with filters: {nsfw_filter}, {sort_method}, keyword: '{keyword_filter}'")
            
            # Clear previous results
            script_instance.cached_prompts = []
            script_instance.prompt_queue = []
            script_instance.queue_index = 0
            
            # Fetch prompts with current filter settings
            prompts = script_instance.fetch_civitai_prompts(
                nsfw_filter,
                keyword_filter,
                sort_method,
                period_filter,
                username_filter,
                post_id_filter,
                model_id_filter,
                limit=100,
                is_fetch_more=False  # Always fresh search
            )
            
            print(f"[Search] Search complete: Found {len(script_instance.prompt_queue)} results")
            
            cache_status = f"Cache: {len(script_instance.cached_prompts)} prompts loaded"
            search_status = f"Search: {len(script_instance.prompt_queue)} results found"
            
            # Update search display (page 1 for new searches)
            search_status_info, search_display, current_page, total_pages = refresh_search_display(1)
            
            return cache_status, search_status, "", "", search_status_info, search_display, current_page
            
        except Exception as e:
            error_msg = f"Error searching prompts: {str(e)}"
            print(f"[Search] Exception: {e}")
            return error_msg, "Search: Error", "", "", "Search: Error", "Error loading search results", 1

    # StableQueue Event Handler
    def stablequeue_output(prompt_index):
        """Handle StableQueue request from JavaScript"""
        return script_instance.stablequeue_output(prompt_index)

    def navigate_to_page(page_num, current_page):
        """Navigate to a specific page in the search results"""
        try:
            page = int(page_num) if page_num else current_page
            search_status_info, search_display, new_page, total_pages = refresh_search_display(page)
            return search_status_info, search_display, new_page
        except Exception as e:
            print(f"[Navigate] Error: {e}")
            return "Search: Error", "Error loading page", current_page

    def go_to_previous_page(current_page):
        """Go to previous page"""
        new_page = max(1, current_page - 1)
        return navigate_to_page(new_page, current_page)

    def go_to_next_page(current_page):
        """Go to next page"""
        # Calculate max page based on current results length
        results_length = len(script_instance.prompt_queue)
        max_page = (results_length + 9) // 10  # Ceiling division for 10 items per page
        new_page = min(max_page, current_page + 1)
        return navigate_to_page(new_page, current_page)

    def load_more_results():
        """Load more results using the same search criteria"""
        try:
            if len(script_instance.prompt_queue) == 0:
                return "Search: No initial search performed", "Please perform a search first", 1
            
            # Use last saved search settings to fetch more
            print(f"[Load More] Loading more results with saved filters")
            
            # Fetch more prompts using saved settings
            prompts = script_instance.fetch_civitai_prompts(
                script_instance.last_nsfw_filter,
                script_instance.last_keyword_filter,
                script_instance.last_sort_method,
                "AllTime",  # Default period
                "",  # Default username
                None,  # Default post_id
                None,  # Default model_id
                limit=100,
                is_fetch_more=True  # Important: this preserves existing results
            )
            
            print(f"[Load More] Total results now: {len(script_instance.prompt_queue)}")
            
            search_status = f"Search: {len(script_instance.prompt_queue)} results found"
            
            # Update search display (stay on current page)
            current_results_length = len(script_instance.prompt_queue)
            items_per_page = 10
            total_pages = (current_results_length + items_per_page - 1) // items_per_page
            
            search_status_info, search_display, current_page, total_pages = refresh_search_display(total_pages)  # Go to last page to show new results
            
            return search_status, search_display, current_page
            
        except Exception as e:
            error_msg = f"Error loading more results: {str(e)}"
            print(f"[Load More] Exception: {e}")
            return "Search: Error", "Error loading more results", 1
    
    # Return all handlers as a dictionary
    return {
        'test_api_connection': test_api_connection,
        'refresh_api_key_status': refresh_api_key_status,
        'get_current_paths': get_current_paths,
        'refresh_lora_list': refresh_lora_list,
        'scan_local_loras': scan_local_loras,
        'scan_and_refresh_folders': scan_and_refresh_folders,
        'force_rescan_loras': force_rescan_loras,
        'force_rescan_and_refresh_folders': force_rescan_and_refresh_folders,
        'clear_lora_cache': clear_lora_cache,
        'vacuum_database': vacuum_database,
        'backup_database': backup_database,
        'list_database_backups': list_database_backups,
        'restore_database_from_backup': restore_database_from_backup,
        'get_database_stats': get_database_stats,
        'search_loras': search_loras,
        'search_loras_with_filters': search_loras_with_filters,
        'show_all_loras': show_all_loras,
        'refresh_folder_choices': refresh_folder_choices,
        'show_folders': show_folders,
        'clear_prompt_cache': clear_prompt_cache,
        # Checkpoint Event Handlers
        'scan_local_checkpoints': scan_local_checkpoints,
        'scan_checkpoints_and_refresh_folders': scan_checkpoints_and_refresh_folders,
        'force_rescan_checkpoints': force_rescan_checkpoints,
        'force_rescan_checkpoints_and_refresh_folders': force_rescan_checkpoints_and_refresh_folders,
        'clear_checkpoint_cache': clear_checkpoint_cache,
        'get_checkpoint_database_stats': get_checkpoint_database_stats,
        'search_checkpoints_with_filters': search_checkpoints_with_filters,
        'show_all_checkpoints': show_all_checkpoints,
        'refresh_checkpoint_folder_choices': refresh_checkpoint_folder_choices,
        'enrich_loras_with_civitai': enrich_loras_with_civitai,
        'enrich_checkpoints_with_civitai': enrich_checkpoints_with_civitai,
        'fetch_new_prompts': fetch_new_prompts,
        'get_prompts_and_update_queue': get_prompts_and_update_queue,
        'refresh_search_display': refresh_search_display,
        'clear_search_results': clear_search_results,
        'fetch_prompts_smart': fetch_prompts_smart,
        'load_more_results': load_more_results,
        'navigate_to_page': navigate_to_page,
        'go_to_previous_page': go_to_previous_page,
        'go_to_next_page': go_to_next_page,
        'stablequeue_output': stablequeue_output
    }

def on_ui_tabs():
    """Create the Civitai Randomizer tab"""
    global script_instance
    
    if script_instance is None:
        script_instance = CivitaiRandomizerScript()
        # Load API key from settings when tab is created
        script_instance.load_config()
    
    with gr.Blocks() as civitai_tab:
        gr.HTML("<h2>Civitai Prompt & LORA Randomizer</h2>")
        gr.HTML("<p>Automatically fetch random prompts from Civitai and randomize LORAs for endless creative generation</p>")
        
        # Hidden input for StableQueue communication (at main UI level like working extension)
        civitai_stablequeue_input = gr.Textbox(elem_id="civitai_stablequeue_input", visible=False)
        
        with gr.Tabs():
            # Create all five tabs
            main_controls_tab = _create_main_controls_tab()
            search_tab = _create_search_tab()
            checkpoint_management_tab = _create_checkpoint_management_tab()
            lora_management_tab = _create_lora_management_tab()
            settings_tab = _create_settings_tab()
        
        # Event handlers
        event_handlers = _create_event_handlers()
        
        # Main Controls Tab Event Bindings        
        main_controls_tab['clear_cache_btn'].click(
            event_handlers['clear_prompt_cache'],
            outputs=[main_controls_tab['cache_status'], main_controls_tab['prompt_queue_status']]
        )
        
        main_controls_tab['populate_btn'].click(
            event_handlers['get_prompts_and_update_queue'],
            inputs=[main_controls_tab['custom_prompt_start'], main_controls_tab['custom_prompt_end'], main_controls_tab['custom_negative_prompt']],
            outputs=[main_controls_tab['prompt_queue_status'], main_controls_tab['hidden_positive_prompt'], main_controls_tab['hidden_negative_prompt'], search_tab['search_info'], search_tab['search_display']],
            _js="""
            function(custom_start, custom_end, custom_negative) {
                console.log('[Civitai Randomizer] Populate button clicked with JS!');
                
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
                        
                        console.log('[Civitai Randomizer] OK Main fields populated via bridge!');
                    } else {
                        console.log('[Civitai Randomizer] X Could not find main prompt fields');
                    }
                }, 500);
                
                return [custom_start, custom_end, custom_negative];
            }
            """
        )
        
        # Search Tab Event Bindings
        search_tab['refresh_results_btn'].click(
            fn=lambda page: event_handlers['refresh_search_display'](page),
            inputs=[search_tab['current_page_num']],
            outputs=[search_tab['search_info'], search_tab['search_display'], search_tab['current_page_num']]
        )
        
        search_tab['fetch_prompts_btn'].click(
            event_handlers['fetch_prompts_smart'],
            inputs=[search_tab['nsfw_filter'], search_tab['keyword_filter'], search_tab['sort_method'], 
                   search_tab['period_filter'], search_tab['username_filter'], search_tab['post_id_filter'], search_tab['model_id_filter']],
            outputs=[main_controls_tab['cache_status'], main_controls_tab['prompt_queue_status'], main_controls_tab['hidden_positive_prompt'], main_controls_tab['hidden_negative_prompt'], search_tab['search_info'], search_tab['search_display'], search_tab['current_page_num']]
        )
        
        search_tab['load_more_btn'].click(
            event_handlers['load_more_results'],
            outputs=[search_tab['search_info'], search_tab['search_display'], search_tab['current_page_num']]
        )
        
        search_tab['clear_results_btn'].click(
            event_handlers['clear_search_results'],
            outputs=[main_controls_tab['cache_status'], main_controls_tab['prompt_queue_status'], search_tab['search_info'], search_tab['search_display'], search_tab['current_page_num']]
        )
        
        # Page navigation bindings
        search_tab['prev_page_btn'].click(
            event_handlers['go_to_previous_page'],
            inputs=[search_tab['current_page_num']],
            outputs=[search_tab['search_info'], search_tab['search_display'], search_tab['current_page_num']]
        )
        
        search_tab['next_page_btn'].click(
            event_handlers['go_to_next_page'],
            inputs=[search_tab['current_page_num']],
            outputs=[search_tab['search_info'], search_tab['search_display'], search_tab['current_page_num']]
        )
        
        search_tab['go_to_page_btn'].click(
            event_handlers['navigate_to_page'],
            inputs=[search_tab['current_page_num'], search_tab['current_page_num']],
            outputs=[search_tab['search_info'], search_tab['search_display'], search_tab['current_page_num']]
        )
        
        # StableQueue event binding
        civitai_stablequeue_input.change(
            fn=event_handlers['stablequeue_output'],
            inputs=civitai_stablequeue_input
        )
        
        # Checkpoint Management Tab Event Bindings
        checkpoint_management_tab['checkpoint_refresh_stats_btn'].click(
            event_handlers['get_checkpoint_database_stats'],
            outputs=[checkpoint_management_tab['checkpoint_db_stats']]
        )
        
        checkpoint_management_tab['checkpoint_scan_db_btn'].click(
            event_handlers['scan_checkpoints_and_refresh_folders'],
            outputs=[checkpoint_management_tab['checkpoint_results_info'], checkpoint_management_tab['checkpoint_folder_filter']]
        )
        
        checkpoint_management_tab['checkpoint_force_scan_btn'].click(
            event_handlers['force_rescan_checkpoints_and_refresh_folders'],
            outputs=[checkpoint_management_tab['checkpoint_results_info'], checkpoint_management_tab['checkpoint_folder_filter']]
        )
        
        checkpoint_management_tab['checkpoint_clear_db_btn'].click(
            event_handlers['clear_checkpoint_cache'],
            outputs=[checkpoint_management_tab['checkpoint_results_info']]
        )
        
        checkpoint_management_tab['checkpoint_vacuum_db_btn'].click(
            event_handlers['vacuum_database'],
            outputs=[checkpoint_management_tab['checkpoint_results_info']]
        )
        

        
        checkpoint_management_tab['enrich_checkpoints_btn'].click(
            event_handlers['enrich_checkpoints_with_civitai'],
            outputs=[checkpoint_management_tab['checkpoint_results_info']]
        )
        
        checkpoint_management_tab['checkpoint_search_btn'].click(
            event_handlers['search_checkpoints_with_filters'],
            inputs=[checkpoint_management_tab['checkpoint_search_name'], checkpoint_management_tab['checkpoint_search_hash'], checkpoint_management_tab['checkpoint_search_folder'], checkpoint_management_tab['checkpoint_filter_has_metadata'], checkpoint_management_tab['checkpoint_filter_has_hash'], checkpoint_management_tab['checkpoint_folder_filter']],
            outputs=[checkpoint_management_tab['checkpoint_display']]
        )
        
        checkpoint_management_tab['checkpoint_show_all_btn'].click(
            event_handlers['show_all_checkpoints'],
            outputs=[checkpoint_management_tab['checkpoint_display']]
        )
        
        checkpoint_management_tab['checkpoint_refresh_folders_btn'].click(
            event_handlers['refresh_checkpoint_folder_choices'],
            outputs=[checkpoint_management_tab['checkpoint_folder_filter']]
        )
        
        # Auto-trigger search when checkpoint folder filter changes
        checkpoint_management_tab['checkpoint_folder_filter'].change(
            event_handlers['search_checkpoints_with_filters'],
            inputs=[checkpoint_management_tab['checkpoint_search_name'], checkpoint_management_tab['checkpoint_search_hash'], checkpoint_management_tab['checkpoint_search_folder'], checkpoint_management_tab['checkpoint_filter_has_metadata'], checkpoint_management_tab['checkpoint_filter_has_hash'], checkpoint_management_tab['checkpoint_folder_filter']],
            outputs=[checkpoint_management_tab['checkpoint_display']]
        )
        
        # Lora Management Tab Event Bindings
        lora_management_tab['refresh_stats_btn'].click(
            event_handlers['get_database_stats'],
            outputs=[lora_management_tab['db_stats']]
        )
        
        lora_management_tab['scan_db_btn'].click(
            event_handlers['scan_and_refresh_folders'],
            outputs=[lora_management_tab['results_info'], lora_management_tab['folder_filter']]
        )
        
        lora_management_tab['force_scan_btn'].click(
            event_handlers['force_rescan_and_refresh_folders'],
            outputs=[lora_management_tab['results_info'], lora_management_tab['folder_filter']]
        )
        
        lora_management_tab['clear_db_btn'].click(
            event_handlers['clear_lora_cache'],
            outputs=[lora_management_tab['results_info']]
        )
        
        lora_management_tab['vacuum_db_btn'].click(
            event_handlers['vacuum_database'],
            outputs=[lora_management_tab['results_info']]
        )
        

        
        lora_management_tab['enrich_loras_btn'].click(
            event_handlers['enrich_loras_with_civitai'],
            outputs=[lora_management_tab['results_info']]
        )
        
        lora_management_tab['search_btn'].click(
            event_handlers['search_loras_with_filters'],
            inputs=[lora_management_tab['search_name'], lora_management_tab['search_hash'], lora_management_tab['search_folder'], lora_management_tab['filter_has_metadata'], lora_management_tab['filter_has_hash'], lora_management_tab['folder_filter']],
            outputs=[lora_management_tab['lora_display']]
        )
        
        lora_management_tab['show_all_btn'].click(
            event_handlers['show_all_loras'],
            outputs=[lora_management_tab['lora_display']]
        )
        
        lora_management_tab['refresh_folders_btn'].click(
            event_handlers['refresh_folder_choices'],
            outputs=[lora_management_tab['folder_filter']]
        )
        
        # Auto-trigger search when folder filter changes
        lora_management_tab['folder_filter'].change(
            event_handlers['search_loras_with_filters'],
            inputs=[lora_management_tab['search_name'], lora_management_tab['search_hash'], lora_management_tab['search_folder'], lora_management_tab['filter_has_metadata'], lora_management_tab['filter_has_hash'], lora_management_tab['folder_filter']],
            outputs=[lora_management_tab['lora_display']]
        )
        
        # Settings Tab Event Bindings
        settings_tab['test_api_btn'].click(
            event_handlers['test_api_connection'],
            outputs=[settings_tab['api_status']]
        )
        
        settings_tab['refresh_api_key_btn'].click(
            event_handlers['refresh_api_key_status'],
            outputs=[settings_tab['current_api_key']]
        )
        
        # Settings Tab Backup Event Bindings
        settings_tab['backup_db_btn'].click(
            event_handlers['backup_database'],
            outputs=[settings_tab['backup_status']]
        )
        
        settings_tab['list_backups_btn'].click(
            event_handlers['list_database_backups'],
            outputs=[settings_tab['backup_display']]
        )
        
        settings_tab['restore_db_btn'].click(
            event_handlers['restore_database_from_backup'],
            inputs=[settings_tab['restore_backup_name']],
            outputs=[settings_tab['backup_status']]
        )
        
        # Initialize folder filter choices for Lora tab
        try:
            folders = script_instance.get_folder_choices()
            lora_management_tab['folder_filter'].choices = folders
            lora_management_tab['folder_filter'].value = folders  # Select all by default
            
            # Show all Loras by default
            if folders:
                initial_results = script_instance.search_loras_db(selected_folders=folders, limit=200)
                initial_display = script_instance.format_lora_database_display(initial_results, "Showing all Loras")
                lora_management_tab['lora_display'].value = initial_display
        except Exception as e:
            print(f"[Civitai Randomizer] Error initializing Lora folder filter: {e}")
        
        # Initialize folder filter choices for Checkpoint tab
        try:
            checkpoint_folders = script_instance.get_checkpoint_folder_choices()
            checkpoint_management_tab['checkpoint_folder_filter'].choices = checkpoint_folders
            checkpoint_management_tab['checkpoint_folder_filter'].value = checkpoint_folders  # Select all by default
            
            # Show all Checkpoints by default
            if checkpoint_folders:
                initial_checkpoint_results = script_instance.search_checkpoints_db(selected_folders=checkpoint_folders, limit=200)
                initial_checkpoint_display = script_instance.format_checkpoint_database_display(initial_checkpoint_results, "Showing all Checkpoints")
                checkpoint_management_tab['checkpoint_display'].value = initial_checkpoint_display
        except Exception as e:
            print(f"[Civitai Randomizer] Error initializing Checkpoint folder filter: {e}")
        
        # Initialize settings tab with current values
        try:
            # Load initial API key status
            import modules.shared as shared
            api_key = getattr(shared.opts, 'civitai_api_key', '')
            if api_key:
                masked_key = '***' + api_key[-4:] if len(api_key) > 4 else '***'
                api_key_status = f"<span style='color: green;'>API Key configured: {masked_key}</span>"
            else:
                api_key_status = "<span style='color: orange;'>No API key configured</span>"
            settings_tab['current_api_key'].value = api_key_status
            
            # Load initial path configurations
            lora_path = getattr(shared.opts, 'civitai_lora_path', '')
            checkpoint_path = getattr(shared.opts, 'civitai_checkpoint_path', '')
            
            lora_status = f"<strong>Lora Path:</strong> {lora_path}" if lora_path else "<strong>Lora Path:</strong> Auto-detected"
            checkpoint_status = f"<strong>Checkpoint Path:</strong> {checkpoint_path}" if checkpoint_path else "<strong>Checkpoint Path:</strong> Auto-detected"
            
            settings_tab['current_lora_path'].value = lora_status
            settings_tab['current_checkpoint_path'].value = checkpoint_status
            
        except Exception as e:
            print(f"[Civitai Randomizer] Error initializing settings tab: {e}")
        
        print(f"[Civitai Randomizer] OK Tab interface with subtabs created successfully")
    
    return [(civitai_tab, "Civitai Randomizer", "civitai_randomizer")]

def _create_settings_tab():
    """Create the settings tab UI components"""
    with gr.TabItem("Settings"):
        gr.HTML("<h3>Configuration & Testing</h3>")
        gr.HTML("<p>Configure your Civitai API settings and test connectivity</p>")
        
        # API Configuration Section
        with gr.Group():
            gr.HTML("<h4>API Configuration</h4>")
            
            # Display current API key setting
            current_api_key = gr.HTML("Loading API key status...")
            
            gr.HTML("<p>To configure your API key, go to <strong>Settings > Civitai Randomizer</strong> in the main WebUI settings.</p>")
            
            # API Test Section
            with gr.Row():
                test_api_btn = gr.Button("Test API Connection", variant="primary", size="sm")
                refresh_api_key_btn = gr.Button("Refresh Key", variant="secondary", size="sm")
            
            api_status = gr.HTML("Click 'Test API Connection' to verify your API key")
        
        # Path Configuration Section
        with gr.Group():
            gr.HTML("<h4>Directory Paths</h4>")
            current_lora_path = gr.HTML("Loading Lora path...")
            current_checkpoint_path = gr.HTML("Loading Checkpoint path...")
            
            gr.HTML("<p>To configure directory paths, go to <strong>Settings > Civitai Randomizer</strong> in the main WebUI settings.</p>")
        
        # Database Backup Section
        with gr.Group():
            gr.HTML("<h4>Database Backup & Restore</h4>")
            gr.HTML("<p>Create and restore backups of your model database (contains both Loras and Checkpoints)</p>")
            
            with gr.Row():
                backup_db_btn = gr.Button("Create Backup", variant="secondary", size="sm")
                list_backups_btn = gr.Button("List Backups", variant="secondary", size="sm")
            
            with gr.Row():
                restore_backup_name = gr.Textbox(placeholder="backup_filename.db", label="Backup to Restore", scale=3)
                restore_db_btn = gr.Button("Restore", variant="secondary", size="sm", scale=1)
            
            backup_status = gr.HTML("<span style='color: #888;'>Database backup management</span>")
            backup_display = gr.HTML("<div style='padding: 20px; text-align: center; color: #888;'>Click 'List Backups' to see available backups</div>")
    
    return {
        'current_api_key': current_api_key,
        'test_api_btn': test_api_btn,
        'refresh_api_key_btn': refresh_api_key_btn,
        'api_status': api_status,
        'current_lora_path': current_lora_path,
        'current_checkpoint_path': current_checkpoint_path,
        'backup_db_btn': backup_db_btn,
        'list_backups_btn': list_backups_btn,
        'restore_backup_name': restore_backup_name,
        'restore_db_btn': restore_db_btn,
        'backup_status': backup_status,
        'backup_display': backup_display
    }

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
    
    shared.opts.add_option(
        "civitai_lora_path",
        shared.OptionInfo(
            "",
            "Local Lora Directory Path (leave empty for auto-detection)",
            gr.Textbox,
            {"placeholder": "/path/to/your/lora/folder"},
            section=section
        )
    )
    
    shared.opts.add_option(
        "civitai_checkpoint_path",
        shared.OptionInfo(
            "",
            "Local Checkpoint Directory Path (leave empty for auto-detection)",
            gr.Textbox,
            {"placeholder": "/path/to/your/checkpoint/folder"},
            section=section
        )
    )
    
    # StableQueue Integration Settings
    shared.opts.add_option(
        "stablequeue_url",
        shared.OptionInfo(
            "http://192.168.73.124:8083",
            "StableQueue Server URL",
            gr.Textbox,
            {"placeholder": "http://your-stablequeue-server:8083"},
            section=section
        ).info("URL of your StableQueue server")
    )
    
    shared.opts.add_option(
        "stablequeue_api_key",
        shared.OptionInfo(
            "",
            "StableQueue API Key",
            gr.Textbox,
            {"type": "password", "placeholder": "Enter your StableQueue API key"},
            section=section
        ).info("API key from your StableQueue instance")
    )
    
    shared.opts.add_option(
        "stablequeue_api_secret",
        shared.OptionInfo(
            "",
            "StableQueue API Secret",
            gr.Textbox,
            {"type": "password", "placeholder": "Enter your StableQueue API secret"},
            section=section
        ).info("API secret from your StableQueue instance")
    )
    
    shared.opts.add_option(
        "stablequeue_default_server",
        shared.OptionInfo(
            "ArchLinux",
            "StableQueue Default Server",
            gr.Textbox,
            {"placeholder": "ArchLinux"},
            section=section
        ).info("Default server alias to send jobs to (configure servers in StableQueue web UI)")
    )

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings) 