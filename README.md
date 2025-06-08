# Civitai Randomizer for Stable Diffusion Forge

A powerful extension for Stable Diffusion Forge that enhances the "Generate Forever" functionality with Civitai API integration, providing automatic prompt randomization and LORA management.

## Features

### 🎯 **Core Functionality**
- **Civitai Prompt Retrieval**: Automatically fetch prompts from Civitai's vast community database
- **Generate Forever Integration**: Seamlessly works with Forge's continuous generation feature
- **Smart Caching**: Efficient prompt caching system to minimize API calls
- **Custom Prompt Enhancement**: Add your own text at the beginning and/or end of fetched prompts

### 🔞 **NSFW Content Management**
- **Flexible Filtering**: Choose to include all content, exclude NSFW, or show only NSFW content
- **Safe Defaults**: Starts with inclusive settings, easily configurable
- **Accurate Classification**: Uses Civitai's own NSFW classification system

### 🎨 **Advanced Prompt Control**
- **Keyword Filtering**: Filter prompts by specific keywords or phrases
- **Multiple Sort Options**: Sort by Most Reactions, Most Comments, or Newest
- **Bypass Mode**: Use only custom prompts and LORA randomization without Civitai fetching
- **Smart Cleaning**: Automatically cleans and formats prompts for optimal results

### 🧩 **LORA Management**
- **Automatic Detection**: Scans your LORA directories and lists available models
- **Random Selection**: Randomly applies LORAs with customizable strength ranges
- **Multi-LORA Support**: Apply multiple LORAs simultaneously with configurable limits
- **Strength Randomization**: Each LORA gets a random strength within your specified range

### 🛠️ **User Experience**
- **Intuitive Interface**: Clean, organized Gradio interface with collapsible sections
- **Real-time Feedback**: Live status updates and API connection testing
- **Error Handling**: Robust error handling with helpful error messages
- **Cache Management**: Easy cache clearing and manual prompt fetching

## Installation

### Method 1: Direct Installation (Recommended)

1. Navigate to your Stable Diffusion Forge installation directory
2. Go to the `extensions` folder (create it if it doesn't exist)
3. Copy the `civitai_randomizer.py` file to the `extensions` folder
4. Restart Stable Diffusion Forge

### Method 2: Git Clone

```bash
cd /path/to/stable-diffusion-forge/extensions
git clone <repository-url> civitai-randomizer
```

### Method 3: Manual Setup

1. Create a new folder in your `extensions` directory called `civitai-randomizer`
2. Copy all files from this repository into that folder
3. Restart Stable Diffusion Forge

## Configuration

### Civitai API Key (Optional)

While the extension works without an API key for public content, having one provides several benefits:

1. **Higher Rate Limits**: Avoid rate limiting on heavy usage
2. **Access to More Content**: Some content may require authentication
3. **Better Performance**: Faster API responses for authenticated requests

To get your API key:
1. Visit [Civitai.com](https://civitai.com)
2. Log in to your account
3. Go to Account Settings → API Keys
4. Generate a new API key
5. Copy and paste it into the extension's API Key field

## Usage Guide

### Basic Setup

1. **Enable the Extension**: Open the "Civitai Randomizer" accordion in the Forge interface
2. **Test API Connection**: Enter your API key (optional) and click "Test API" to verify connectivity
3. **Configure Filters**: Set your NSFW preferences and any keyword filters
4. **Enable Randomizer**: Check "Enable Civitai Randomizer" to activate automatic prompt fetching

### Generate Forever Integration

The extension automatically integrates with Forge's "Generate Forever" feature:

1. Enable "Generate Forever" in the main interface
2. Enable "Civitai Randomizer" in the extension panel
3. Each new generation will automatically get a fresh prompt from Civitai

### LORA Randomization

1. **Refresh LORA List**: Click "Refresh LORA List" to scan for available LORAs
2. **Select LORAs**: Choose which LORAs you want to include in randomization
3. **Configure Strength**: Set minimum and maximum strength values
4. **Set Limits**: Choose how many LORAs to apply per generation
5. **Enable**: Check "Enable LORA Randomizer" to activate

### Advanced Features

#### Custom Prompts
- **Beginning Text**: Added before the Civitai prompt
- **End Text**: Added after the Civitai prompt
- **Bypass Mode**: Skip Civitai fetching entirely, use only custom text and LORAs

#### Keyword Filtering
- Enter comma-separated keywords to filter prompts
- Only prompts containing at least one keyword will be used
- Case-insensitive matching

#### Cache Management
- **View Cache Status**: See how many prompts are currently cached
- **Clear Cache**: Remove all cached prompts to force fresh fetching
- **Manual Fetch**: Fetch new prompts immediately with current settings

## Interface Overview

### Main Controls
- **Enable Civitai Randomizer**: Master switch for the extension
- **Bypass Prompt Fetching**: Use only custom prompts and LORA randomization
- **NSFW Content Filter**: Control NSFW content inclusion
- **Keyword Filter**: Filter prompts by specific terms
- **Sort Method**: Choose how to sort fetched prompts

### Custom Prompt Settings
- **Custom Prompt (Beginning)**: Text added to the start of each prompt
- **Custom Prompt (End)**: Text added to the end of each prompt

### LORA Management
- **Enable LORA Randomizer**: Toggle LORA randomization
- **Available LORAs**: Select which LORAs to include
- **Strength Range**: Min/max strength for random application
- **Max LORAs per Generation**: Limit simultaneous LORA usage

### Status & Cache
- **Cache Status**: Shows current number of cached prompts
- **Clear Cache**: Remove all cached prompts
- **Fetch New Prompts**: Manually fetch fresh prompts

## Troubleshooting

### Common Issues

#### "API Connection Failed"
- **Check Internet Connection**: Ensure you have a stable internet connection
- **Verify API Key**: If using an API key, make sure it's correctly entered
- **Check Civitai Status**: Visit civitai.com to ensure the service is operational

#### "No Prompts Fetched"
- **Adjust Filters**: Your keyword or NSFW filters might be too restrictive
- **Check Rate Limits**: You might be hitting API rate limits (wait a few minutes)
- **Verify Settings**: Ensure your sort method and other settings are appropriate

#### "LORAs Not Detected"
- **Check LORA Directory**: Ensure LORAs are in the correct directories:
  - `models/Lora/`
  - `models/LyCORIS/`
  - `{models_path}/Lora/`
  - `{models_path}/LyCORIS/`
- **File Formats**: Only `.safetensors`, `.ckpt`, and `.pt` files are detected
- **Refresh List**: Click "Refresh LORA List" after adding new LORAs

#### "Extension Not Showing"
- **Restart Forge**: Restart the entire Stable Diffusion Forge application
- **Check File Location**: Ensure `civitai_randomizer.py` is in the `extensions` directory
- **Check Logs**: Look at the console output for error messages

### Performance Tips

1. **Cache Management**: Clear cache periodically to get fresh content
2. **Reasonable Limits**: Don't set too many LORAs per generation (2-3 is usually optimal)
3. **API Key Usage**: Use an API key to avoid rate limiting
4. **Keyword Specificity**: Use specific keywords to get more relevant prompts

### Debug Mode

To enable debug output, set `show_debug_info` to `true` in `config.json`. This will provide additional console output for troubleshooting.

## Configuration File

The `config.json` file allows you to customize default settings:

```json
{
    "civitai_api": {
        "base_url": "https://civitai.com/api/v1",
        "default_params": {
            "limit": 100,
            "sort": "Most Reactions"
        },
        "rate_limit": {
            "requests_per_minute": 60,
            "cooldown_seconds": 5
        }
    },
    "extension_settings": {
        "default_nsfw_filter": "Include All",
        "max_cached_prompts": 1000
    },
    "lora_settings": {
        "default_strength_range": [0.5, 1.0],
        "max_loras_per_generation": 3
    }
}
```

## API Reference

### Civitai API Endpoints Used

- **GET /api/v1/images**: Fetch image metadata and prompts
  - Parameters: `limit`, `nsfw`, `sort`, `username`, `period`
  - Authentication: Bearer token (optional for public content)

### Extension Methods

The extension provides several key methods:

- `fetch_civitai_prompts()`: Fetches prompts from Civitai API
- `generate_random_prompt()`: Creates randomized prompts
- `apply_random_loras()`: Applies random LORA selections
- `get_available_loras()`: Scans for available LORA files

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Make your changes
4. Test with Stable Diffusion Forge
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This extension is not officially affiliated with Civitai or Stability AI. It uses the public Civitai API in accordance with their terms of service. Please ensure you comply with Civitai's terms of use when using this extension.

## Support

If you encounter issues or have questions:

1. Check the troubleshooting section above
2. Review the console output for error messages
3. Check that all dependencies are installed
4. Ensure you're using a compatible version of Stable Diffusion Forge

## Changelog

### Version 1.0.0
- Initial release
- Civitai API integration
- NSFW filtering
- LORA randomization
- Generate Forever integration
- Custom prompt management
